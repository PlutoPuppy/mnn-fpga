# Larger hardware jobs, prepared INT8 operands and DMA

The driver supports larger matrix jobs and both PIO and DMA transports on
the validated 90 MHz matmul-soc image. The original small-job executable is
preserved as `tinyvit_offload`; the new executable is deployed separately as
`tinyvit_offload_batched`.

## Latest upstream review

Reviewed latest upstream commit
[`b3ea448a36a5db7bd301173d4e5fff027eca2382`](https://github.com/j3rryhu/matmul-soc/commit/b3ea448a36a5db7bd301173d4e5fff027eca2382)
on 2026-11-11. Compared with the previously inspected `4b749ced`, it adds
`test_program/matmul_accel_cli.c`, `test_program/matmul_baseline.c` and a
SignalTap configuration change. It does not change accelerator RTL or the
register ABI. The current validated 90 MHz image remains compatible.

The new upstream CLI explicitly describes per-bank capacity rules and larger
hardware tiles: a 256x64 by 64x256 job fills both 16 KiB operand buffers and
the 64 KiB output canvas. This integration follows those capacity rules and
the upstream DMA addresses/control protocol.

## PIO versus DMA

PIO means the CPU executes the loads and stores that carry each buffer word
over the HPS-to-FPGA bridge. The previous driver used 128-bit NEON PIO and
ordering barriers around each beat. This is a hardware offload of arithmetic,
but the DMA engines did not carry its data.

The bulk PIO transport now sends whole operand images with full 128-bit
transactions and barriers around the complete image. It remains useful as a
simple fallback and an independent check of the DMA path.

The DMA transport uses the two engines already present in the FPGA image:

- DMA0 reads an operand image from HPS DDR and writes accelerator W/X RAM.
- DMA1 reads the result canvas and writes it to HPS DDR.
- The CPU prepares/stages INT8 bytes and programs the descriptors, but does
  not issue the accelerator-buffer loads/stores for the payload itself.
- Transfers are split into at most 8192-byte descriptors, following the
  upstream transfer-length limitation. This includes full-canvas readback.

This board has no udmabuf/CMA userspace allocation interface. The prototype
therefore uses the same excluded DDR region as the upstream test programs:
`0x30000000..0x3001ffff`. Linux on the tested image uses RAM only through
`0x2fffffff`; the board bootloader reports 1 GiB physical DDR. Before mapping
the staging region, the transport reads `/proc/iomem`, requires an observable
System RAM map and rejects overlap with any reported Linux memory resource.
It never uses a malloc pointer as a DMA address. Only one cooperating client
may own the accelerator; `/dev/mem` is locked for the device lifetime.

DMA is an explicit board-specific opt-in. Do not use this fixed window on a
different memory layout or share it with another FPGA program. Production
integration should replace it with a coherent DMA buffer allocation driver.
The current staging mappings are uncached/device memory: CPU staging and
host readback still have a cost and are included in the full-model benchmark.
Using DMA is not a promise of a speedup for small jobs.

## Larger jobs

These are the current driver's panel choices, not absolute hardware shape
limits. The [shape and ABI review](matmul-soc-shape-and-abi-review.md) verifies
the updated upstream register map, partial dimensions and capacity rules.
In particular, 128x128 by 128x128 fits both operand buffers; K=64 is needed
for the current maximum 256x256 output-panel choice, not every possible tile.

The software planner uses these aligned physical job limits:

| Dimension | Maximum per submitted job |
| --- | ---: |
| Output rows M | 256 |
| Output columns N | 256 |
| Reduction length K | 64 |
| A bytes | 16,384 |
| B bytes | 16,384 |
| Output bytes | 65,536 |

The FPGA's physical array is still 16x16. One hardware start now covers all
internal row/column tiles and 16-wide reduction bands for that job. Software
uses additional panels for dimensions beyond these limits and pads tails to
16. It accumulates reduction panels in host double precision and publishes
FP32 output only when the entire GEMM succeeds.

The submission count is:

`ceil(M/256) * ceil(N/256) * ceil(K/64)`

| Workload | Previous driver | Batched driver |
| --- | ---: | ---: |
| One 49x32 by 32x49 head | 32 starts | 1 start |
| First QK: 16 windows x 4 heads | 2048 starts | 64 starts |
| One 256x256 by 256x256 matrix product | 4096 starts | 4 starts |

For the first QK, aligned job dimensions are 64x32 by 32x64. Each head sends
4096 operand bytes and reads 4096 result bytes. Across 64 heads that is
512 KiB of logical payload versus 1.5 MiB previously, excluding DMA staging,
registers and polling. Padding MACs are unchanged; the improvement comes from
fewer submissions, less repeated preparation/transfer, and hardware-managed
internal tiling. This planner provides safe capacity bounds rather than an
exhaustive optimal search for every rectangular shape.

## Prepared INT8 input API

`MatmulSocOperand` describes a dense row-major signed INT8 matrix and a scale:

`real_value = values[index] * scale`, with zero point zero.

Callers that already have prepared bytes can submit them directly:

```cpp
fpga::MatmulSocOperand a, b;
a.rows = M; a.cols = K; a.values = prepared_a_bytes; a.scale = a_scale;
b.rows = K; b.cols = N; b.values = prepared_b_bytes; b.scale = b_scale;
auto stats = fpga::matmul_soc_gemm_prepared(io, a, b, output_fp32);
// stats.quantized_values == 0: this path does not reconvert FP32 inputs.
```

Prepared operands may be reused when their values and scales remain valid.
The driver packs/pads them for the buffer layout but does not requantize them.
The API supports optional bias and transposed output. Input operands are
already in their logical row-major order.

For FP32 input, `prepare_matmul_soc_operand(data, rows, cols, transpose)`
validates all values, finds the maximum absolute value, and quantizes once
using the symmetric scale `max(abs(data))/127` (all-zero scale 1). Rounding
uses nearest with ties away from zero. `matmul_soc_gemm` is the FP32 adapter
that prepares A/B and then calls the prepared API.

The selected TinyViT Q and K matrices are **dynamic activations**, produced
while the FP32 model runs. They cannot be prepared offline as fixed weights.
The current MNN bridge prepares each head's operands once at runtime. This
reduces the first QK's quantized-value count from 1,048,576 padded/repeated
tile values to 200,704 actual values. The hardware receives INT8 in both
transports; the rest of this TinyViT model remains FP32. This work does not
convert the complete MNN model to an INT8 model or add an INT8 MNN backend hook.

Direct inspection of the deployed `tiny-vit.mnn` (SHA256
`f7aee628e2cef2d726979cc259529b3af0778b429c52ba8b76f3c609a2996de4`)
confirmed 1463 serialized operators, 68 convolution/depthwise records with
5,349,888 FP32 weight elements, zero quantized-weight records, zero INT8/UINT8
constant elements, zero tensor-quantization records and no INT8/quantized
operator types. Constant blobs use FP32 or INT32. The model file itself is
not INT8-quantized; temporary INT8 operands are introduced by the offload code.
Inspection output is saved in `build/model-inspection/quantization.txt`.

## Numerical policy

The hardware floors and saturates each 16-wide reduction band's result to
INT8, then accumulates those bytes with saturation. The driver chooses one
Q0.16 output multiplier per job from an integer row-L1 bound and B's peak,
reserving one rounding LSB for each band to prevent intermediate/final
saturation. Output is dequantized using the actual encoded multiplier.

This changes the approximation from the old independently quantized small
jobs. It is not bit-identical to the previous driver or FP32 arithmetic.
Larger multi-band smoke cases can have larger absolute error; full-model
score comparisons are needed. Saturating `use_prev_out` is not used between
host reduction panels; those panels are reconstructed and summed on CPU.

## Commands

Load the validated 90 MHz image using the existing interrupted-U-Boot/JTAG
procedure; a normal SD boot replaces it with the stock image. Then, from
`/home/ubuntu/tinyvit`:

```sh
# Physical tests, including prepared INT8 and a full 256x256 output canvas:
sudo ./matmul_soc_smoke_batched --devmem
sudo ./matmul_soc_smoke_batched --devmem --dma

# Full model, first QK, real DMA transport:
sudo env MNN_MATMUL_SOC_DEVMEM=1 MNN_MATMUL_SOC_DMA=1 \
  ./tinyvit_offload_batched --model tiny-vit.mnn --height 224 --width 224 \
  --input-f32 input.f32 --threads 2 --runs 1 \
  --fpga-op /layers.1/blocks.0/attn/MatMul_output_0 --require-fpga \
  --output-dir outputs/first-qk-batched-dma
```

Omit `MNN_MATMUL_SOC_DMA=1` for bulk PIO. The runtime log explicitly identifies
`DMA` or `PIO`, prepared-value and byte counts; the runner reports completed
FPGA jobs. Use fresh processes until the separate MNN session-reuse output
drift is resolved.

## Validation evidence

Host tests passed for prepared INT8, transpose combinations, tails, full
canvas use, matrices beyond one canvas, multi-band floor arithmetic, invalid
scales/nonfinite input, busy hardware and failure-atomic output.

Physical prepared-INT8 tests passed in both PIO and DMA modes: 14 cases,
18 hardware submissions per transport, including 49x32 by 32x49, a full
256x64 by 64x256 job, a 65-element reduction tail and 272 output rows. Both
transports reported identical errors for all cases. Logs are under
`build/batched-validation/`.

Full-model validation passed with 64 jobs for first QK. The new binary was
deployed as `tinyvit_offload_batched`; SHA256:
`edd091d50d3c1fc2ea579935b786eb4c35acc6c4681ce7e0eb3c2c74170a10a8`.
The original `tinyvit_offload` was preserved. On each of 20 paired random-image
tests per transport, all runs completed and CPU scores matched the earlier
CPU baseline hashes. All 20 PIO/DMA score hashes matched each other exactly.

| Mean full-model inference | CPU baseline | Offload | Added latency |
| --- | ---: | ---: | ---: |
| Previous small-job PIO (earlier test) | 1164.107 ms | 1348.315 ms | 184.208 ms / 15.824% |
| New larger-block PIO | 1175.294 ms | 1225.104 ms | 49.810 ms / 4.238% |
| New larger-block DMA | 1179.150 ms | 1224.162 ms | 45.013 ms / 3.817% |

Each new row uses 20 paired images, alternating CPU/offload order, fresh
processes and two CPU threads. Both modes completed 1280 measured hardware
jobs (20 x 64). Their own CPU baselines are used because these are separate
benchmark sessions. The prior baseline is shown for context, not as a direct
clock-controlled comparison. DMA and PIO latency were similar; these data do
not establish a meaningful DMA advantage for this workload. Both still add
inference latency versus CPU despite the substantial reduction in overhead.

Paired-bootstrap 95% intervals for added inference latency were 42.81–56.17 ms
for PIO and 25.21–58.05 ms for DMA. PIO was slower in 20/20 pairs; DMA in
19/20 pairs, with one unusually slow CPU run. No samples were discarded.

The complete process wall times were noisy in these later runs:

| Mean complete process | CPU baseline | Offload | Mean increase |
| --- | ---: | ---: | ---: |
| Larger-block PIO | 2415.377 ms | 2438.590 ms | 23.213 ms / 0.961% |
| Larger-block DMA | 2460.438 ms | 2494.767 ms | 34.329 ms / 1.395% |

These include model loading, decode/preprocessing, output saving and teardown.
Their paired 95% intervals cross zero (PIO: -53.07 to +96.18 ms added; DMA:
-65.39 to +146.10 ms added). They do not establish a process-level speedup or
slowdown; the measured inference-time penalty is much more consistent.

The new approximation had top-1 agreement with FP32 on all 20 noise images,
top-5 ranking agreement on 18/20, maximum score error 0.015166, and minimum
cosine similarity 0.99999035. These are numerical checks on noise, not dataset
accuracy. The changed quantization policy is not bit-identical to the previous
driver. A fully INT8 model/producer could bypass runtime FP32 preparation;
that is not part of this FP32 TinyViT integration.

Evidence is saved in `build/benchmark-batched-pio-90mhz/` and
`build/benchmark-batched-dma-90mhz/`: per-pair CSV/JSON, metadata, statistics
and plots. Raw logs, outputs and physical tests are archived in
`build/batched-validation/evidence-without-images.tar.gz`. All 80 measured
raw score-file hashes were checked against the result JSON after download.
The archive includes explicit `prepared INT8, DMA` logs for all DMA model runs.
