# TinyViT co-design results on the DE1-SoC

The optimized 90 MHz pipeline meets the 1.5× target after one warm-up
forward. Across 20 matched random inputs, it averages **757.013 ms**
versus **1,207.126 ms** on original FP32, a **1.595×** speedup. The
ratio of medians is **1.550×**. The separate first-forward sweep reaches
**1.494×** with a 3.047 ms gap to its target. On 200 labeled Imagenette
images, top-1 accuracy is **155/200** on FPGA and **153/200** on FP32.

## Partition and arithmetic

The model is TinyViT-5M at 224 × 224, batch 1. The Cortex-A9 runs MNN with
two threads. The reusable 90 MHz NI83 systolic engine serves the selected
pointwise convolutions and static-weight projections. The adapter uses signed
INT8 activations and weights and accumulates in INT32. It writes an INT8
result after applying per-channel fixed-point scale and bias. The driver
chooses a full-reduction tile from the 32 KiB activation and 64 KiB weight
buffers and the 64 KiB output space. The 16 × 16 array processes each job's
internal tiles. For K = 640 with N = 160, the largest ordinary tile in this
implementation is M = 48, K = 640, N = 96, with tails padded to 16 lanes.

The previous mixed model has 30 native transformer linears. Nine additional
FP32 graph operators now use the same GEMM engine. These operators add one
FC1, five FC2, two QKV projections, and one attention output projection. In
total, the new policy covers ten FC1, nine FC2, ten QKV and ten attention
output projections across ten transformer blocks. A single sensitive FC2
stays FP32 on the HPS. The reported experiment uses the existing mixed MNN
model and a separately hashed per-channel calibration table.

Weights for the nine added projections are quantized and packed during
runner setup and cached in reserved HPS DDR3. The activation scan computes
one centered INT8 scale per live tensor. Each output channel has its own
calibration center and scale from 128 independent training images. The
driver folds the input center into the bias using the quantized weight sums.
It converts returned INT8 values into FP32 for the surrounding MNN operators.
The runtime reads the tensor's storage format from MNN. The public dimension
type reports CAFFE for both NCHW and NC4HW4, which caused incorrect results in
an early candidate until the storage-layout check was corrected.

For four existing MLP pairs, the FC1 output stays in FPGA RAM, receives its
GELU from a lookup table, and feeds FC2 over local DMA. HPS DMA continues to
move the other activations and packed weights between FPGA and reserved DDR3.
The Cortex-A9 keeps the stem, MBConv, depthwise and downsample convolutions,
remaining nonlinearities, residuals, normalization, attention score and value
matmuls, and classifier. The model graph has folded convolution parameters
and no separate BatchNorm operations.

The fused CPU QK/scale/bias/softmax/AV path uses vector key packing, INT8-to-FP32
unpacking, FP32 logit scale/bias, max reduction, and normalization. Attention
uses actual model window sizes of 49 or 196 tokens. A separate four-image
pilot matched the prior scores byte for byte with both projection policies.
Two-core attention compute with the original 30-linear selection averaged
154.998 ms before and 114.614 ms afterward in this pilot.

## Accuracy experiments

On the first 20 labeled diagnostic images, the corrected broad ten-projection
policy reached cosine 0.97379 and 19/20 FP32 top-1 agreement. Adding all five
candidate MBConv pointwise operations increased mean latency from 832.827 ms
to 875.170 ms and reduced FP32 agreement to 18/20. A subsequent six-image layer
ablation isolated stage 3 block 0 FC2. With that FC2 on the CPU, cosine rose
to 0.99586 and FP32 agreement was 6/6. Computing only its most outlying output
channel in FP32 still yielded cosine 0.97659 and agreement 5/6. The nine-layer
policy uses the full CPU FC2 for that layer. The runtime layout fix also passed
same-input projection checks, with relative L2 errors of approximately 0.94%
to 4.58% for the ten transformer candidates on one labeled input.

The labeled 200-image split comes from the ten-class Imagenette validation
set. Its image hashes are disjoint from the 128 calibration inputs and all
20 layer-selection inputs. The acceptance thresholds are mean logit cosine
at least 0.995, top-1 agreement at least 95%, and at most one percentage point
accuracy loss versus original FP32. The ten-class result covers this split.

| 200-image accuracy check | Result |
|---|---:|
| Original FP32 correct | 153/200 (76.5%) |
| Selected FPGA pipeline correct | 155/200 (77.5%) |
| FPGA and FP32 top-1 agreement | 195/200 (97.5%) |
| Mean logit cosine | 0.99617336 |

The selected pipeline passes all three acceptance checks. The benchmark reused
hash-verified FP32 reference scores for the same images, preprocessing, and
original model. The first accuracy attempt on the older NI83 image stopped
at its 60-image console checkpoint. The completed accuracy run uses the
new 90 MHz DMC1 image and persistent checkpoints.


## Hardware implementation

The DMC1 variant adds a sequencer at each of the three DMA CSR interfaces.
The HPS starts a transfer of up to 64 KiB with one command. The sequencer
splits it into the same legal 2 KiB HPS or 256-byte local bursts as the
original driver, so the existing Avalon data ports and FIFO remain in use.
It reports completed or rejected commands and provides a capability check.
A modified driver refuses chained mode with an older image. Host command
count and physical burst count are separate metrics.

Quartus fitted the DMC1 design in 23,160 ALMs, 35 DSP blocks, and 368 of
397 M10K blocks. The prior design used 22,710 ALMs with the same DSP and M10K
counts. This change targets control overhead. The existing payload still uses
the FPGA-to-HPS AXI bridge, with the lightweight bridge used for CSRs.

## Verification methodology

Each timing configuration uses the same 20 saved uniform-random RGB inputs
and two HPS threads. The benchmark hashes images, models, the runner and
calibration tables, and alternates the configuration order. For each image,
it runs each configuration once with timing control and once with operator
profiling. The control value covers a complete forward pass after model and
session setup. Fresh-process wall time also includes process startup, model
I/O, preparation and prepacking. Profiling adds instrumentation overhead.

Host native-driver checks cover 58 matrix and resident-pair cases. ARM NEON
attention passes 18 standalone cases, GELU passes 48 host oracle cases, and
the chained DMA sequencer passes simulated 256- and 2,048-byte engines with
aligned tails, 64 KiB transfers, rejection, cancellation and legacy mode.
## DMA chaining A/B on the same bitstream

| Measurement | Unchained | Chained |
|---|---:|---:|
| Host input DMA commands | 4,884 | 408 |
| Host output DMA commands | 1,597 | 327 |
| Host local DMA commands | 3,998 | 55 |
| Total host commands | 10,479 | 790 |
| DMA command setup (ms) | 30.385 | 2.334 |
| Input upload, including staging (ms) | 112.753 | 92.748 |
| Output download and unpacking (ms) | 44.849 | 39.365 |
| Resident FC1-to-FC2 local feedback (ms) | 23.259 | 2.459 |
| GEMM completion wait including polling (ms) | 49.299 | 49.740 |
| Complete profiled adapter (ms) | 259.025 | 214.656 |
| Loaded-model forward mean (ms) | 805.172 | 793.873 |
| Loaded-model forward median (ms) | 806.268 | 783.369 |

The host command count fell by 92.46%. Both modes executed 400 GEMM jobs
and moved 10,179,456 input bytes, 2,988,608 output bytes, and 1,023,488 local
feedback bytes per image. All 20 paired score outputs match byte for byte.
The DMA test included a 979.935 ms chained run on input 3. The measured mean
includes that run and reaches 1.463× versus 1,161.534 ms original FP32. The
unchained and chained profiles use the same arithmetic and buffer geometry.

All 88 reported Quartus timing checks pass at 90 MHz. The 85°C core setup
slack is +0.910 ns. The smallest reported slack is +0.077 ns on an HPS SDRAM
hold check at the fast 0°C corner. JTAG programming succeeded and the board
passed all 13 matrix tests with chaining enabled, including the 64 KiB
output case, K up to 2,048, and three resident MLP geometries.

### Shapes introduced by the nine-projection policy

| Operation | Live M × K × N | Maximum tile M × K × N | Jobs |
|---|---:|---:|---:|
| Stage 1 block 0 QKV | 784 × 128 × 384 | 256 × 128 × 128 | 12 |
| Stage 1 block 0 FC2 | 784 × 512 × 128 | 64 × 512 × 128 | 13 |
| Stage 2 block 3, 4 or 5 FC2, each | 196 × 640 × 160 | 48 × 640 × 96 | 10 |
| Stage 3 block 0 FC1 | 49 × 320 × 1,280 | 96 × 320 × 128 | 10 |
| Stage 3 block 1 attention output | 49 × 320 × 320 | 96 × 320 × 128 | 3 |
| Stage 3 block 1 QKV | 49 × 320 × 960 | 96 × 320 × 128 | 8 |
| Stage 3 block 1 FC2 | 49 × 1,280 × 320 | 16 × 1,280 × 48 | 28 |

Maximum tile size is capacity-derived. Each job uses the live span and pads
its compute tails to a multiple of 16. The nine layers add 104 GEMM jobs.
## First-forward latency and staging overlap

This sweep used the same binary and DMC1 bitstream for both FPGA modes.
The staging switch schedules a CPU worker after the current output DMA
completes. The HPS can then upload or launch the next tile. A wait protects
reuse of the DDR destination and the completed tensor before MNN reads it.
The worker uses the existing reserved DDR window and adds no FPGA RAM.

| Configuration | Forward mean (ms) | Median (ms) | SD (ms) | Fresh-process wall mean (ms) |
|---|---:|---:|---:|---:|
| Original FP32 CPU | 1,172.569 | 1,167.345 | 15.639 | 2,780.392 |
| FPGA with chained DMA | 790.244 | 789.311 | 10.021 | 2,185.880 |
| FPGA with chained DMA and overlapping output staging | 784.760 | 773.274 | 41.058 | 2,011.971 |

The staged configuration is **1.494× faster** than original FP32, or 33.07%
lower mean forward latency. All 20 FPGA runs beat their paired CPU times.
The 95% paired-bootstrap speedup interval is [1.455, 1.522]×. The target
for this suite is 781.712 ms, leaving a 3.047 ms gap. The mean includes the
941.898 ms staged run on input 8. All 20 staged scores and separate profiled
scores match the serialized-copy version exactly. The startup-inclusive
speedup is 1.382×. Preparation for all 39 selected linears averages 343 ms per fresh FPGA
process. The prior 30-linears path averaged 88 ms in the DMA A/B suite.

### Full pipeline profile, milliseconds per image

| Component | Original FP32 | Chained DMA | Chained DMA + staging overlap |
|---|---:|---:|---:|
| Selected linear callbacks | 495.592 | 218.429 | 202.196 |
| Remaining CPU convolutions | 270.609 | 270.473 | 275.184 |
| CPU QK, softmax and AV attention | 200.960 | 122.850 | 122.759 |
| Remaining CPU GELU | 125.486 | 89.225 | 91.652 |
| Remaining activation casts | 0.000 | 24.072 | 23.574 |
| Other layout copies | 54.014 | 46.546 | 46.585 |
| Residual, normalization, and framework work | 32.318 | 34.799 | 35.337 |
| **Instrumented inference** | **1,178.978** | **806.395** | **797.288** |

| Adapter component | Chained DMA | Chained DMA + staging overlap |
|---|---:|---:|
| Activation range scan | 5.739 | 5.423 |
| FP32 to INT8 activation quantization | 6.890 | 7.046 |
| Input layout conversion | 0.015 | 0.014 |
| Tile configuration and launch | 10.461 | 10.438 |
| Input upload, including packing and DMA | 93.386 | 94.152 |
| FPGA completion wait and polling | 49.420 | 50.077 |
| Local feedback for resident MLPs | 2.461 | 2.469 |
| Blocking output download and staging | 39.555 | 21.673 |
| INT8 to FP32 output reconstruction | 3.929 | 3.997 |
| Other adapter work | 2.643 | 2.784 |
| **Complete adapter** | **214.500** | **198.074** |

The staging worker spends 33.249 ms unpacking outputs, overlapping upload
and GEMM for the next tile. The additive table uses the blocking download
phase, so the worker time belongs only to the overlap analysis. The profile
shows a 17.882 ms drop in blocking download and a 16.426 ms drop in total
adapter time. The full inference difference also includes host scheduling
and the remaining CPU kernels. The mean staging gain was 5.484 ms. The paired
95% interval for the gain was [-15.833, 20.577] ms, and the median paired gain
was 15.984 ms.

The largest remaining CPU group is convolution at 275 ms. Input upload
uses about 27.3 ms for staging and 63.7 ms for DMA wait. The measured FPGA
uses signed INT8 GEMM, with CPU attention nonlinearities and CPU stem and
depthwise convolution. The existing data bridge and RAM allocation leave
chaining and host-side overlap as the changes evaluated here. Intel describes
the dedicated SDRAM path as a separate interface. [Intel Cyclone V SoC design guide](https://cdrdv2-public.intel.com/666598/an-cv-av-soc-ddg-683360-666598.pdf)
## Warmed-session performance

Each of the same 20 image pairs ran in alternating order. A session ran one
untimed forward on that image before its timed forward. Each warmed score
file matched its first-forward counterpart byte for byte.

| Configuration | Forward mean (ms) | Median (ms) | SD (ms) |
|---|---:|---:|---:|
| Original FP32 CPU | 1,207.126 | 1,160.485 | 187.239 |
| FPGA with chained DMA and staging overlap | 757.013 | 748.826 | 24.702 |

The warmed mean speedup is **1.595×** with a 95% paired-bootstrap interval
of [1.517, 1.719]×. The ratio of medians is **1.550×**. All 20 of 20
FPGA forwards completed faster than their matched CPU forwards. Both means
include all measurements. The slowest FP32 forward took 1,993.920 ms
and the slowest FPGA forward took 852.481 ms.

## Rebuild and rerun

The tested hardware starts from the generated 90 MHz NI83 project. Build the
chained variant in a fresh directory using Quartus Prime Lite 23.1std.1.

```sh
python platform/prepare_matmul_soc_dma_chain.py --source build/matmul-soc-native-ni83-90 --output build/matmul-soc-native-dmc1-rebuild
python platform/build_matmul_soc.py --friend-root build/matmul-soc-native-dmc1-rebuild --skip-generation
python platform/check_quartus_timing.py build/matmul-soc-native-dmc1-rebuild/output_files/adder_prototype.sta.summary
```

Use the selected projection allowlist at `mnn/app/tinyvit_native_projection_allowlist.txt`
with `mnn/app/calibrate_native_projection.py` to regenerate ranges from the
calibration split. The measured package uses the saved ranges and provenance.
The FPGA runner sets `MNN_MATMUL_SOC_REUSE_A=1`, `MNN_MATMUL_SOC_LIVE_ROWS=1`,
`MNN_MATMUL_SOC_DMA_POLL_BATCH=1`, `MNN_MATMUL_SOC_DMA_CHAIN=1`, and
`MNN_MATMUL_SOC_PIPELINE_C=1`. The benchmark gates transport and projection
choices with required FPGA job counts, output hash parity, and real-image
classification checks.

The local deployment bundle is `build/tinyvit-codesign-90mhz.tar.gz`. It
contains the validated SOF, ARM executable, mixed model, ranges, launch
script and SHA256 manifest. The packaged launcher ran 400 FPGA jobs on the
board and produced the exact score hash from the matching benchmark input.

The complete timing summaries, per-input latency, Quartus reports, and
matrix test output are indexed in the [evidence directory](benchmarks/codesign-optimization-2026-11-21/README.md).
