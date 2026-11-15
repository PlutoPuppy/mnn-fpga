# TinyViT first-QK offload: 20 paired random-image tests at 90 MHz

Measured on the connected DE1-SoC on 2026-11-11.

## Result

Offloading the first QK attention multiplication to the current 90 MHz FPGA
implementation increased full-model inference latency in **all 20 pairs**.
Mean inference time increased by **184.21 ms, or 15.82%**. Mean complete
process time increased by **185.63 ms, or 11.46%**.

| Timing statistic | HPS CPU only | First QK on FPGA; rest on CPU |
| --- | ---: | ---: |
| Mean full-model inference | 1164.107 ms | 1348.315 ms |
| Median full-model inference | 1162.965 ms | 1346.485 ms |
| Sample standard deviation | 2.951 ms | 5.414 ms |
| Minimum / maximum inference | 1160.35 / 1169.35 ms | 1343.09 / 1365.73 ms |
| Mean complete process wall time | 1619.335 ms | 1804.964 ms |
| Median complete process wall time | 1619.054 ms | 1803.026 ms |
| Sample process-time standard deviation | 3.405 ms | 6.216 ms |

| Paired comparison | Full-model inference | Complete process |
| --- | ---: | ---: |
| Mean added latency | +184.208 ms | +185.629 ms |
| Added time relative to CPU mean | +15.824% | +11.463% |
| CPU/offload ratio of mean times | 0.8634x | 0.8972x |
| Offload faster / slower pairs | 0 / 20 | 0 / 20 |
| Paired bootstrap 95% interval, added latency | +181.70 to +187.09 ms | +182.42 to +189.00 ms |
| Paired bootstrap 95% interval, added time | +15.60% to +16.08% | +11.26% to +11.68% |

Intervals use 20,000 paired resamples with seed 20261111. Both members of a
pair are resampled together. Percentage change is calculated from the ratio
of mean times, not by averaging differently weighted percentages. These
intervals describe this benchmark, not all possible board workloads.

The additional cost is similar with either order: CPU-first pairs added
183.485 ms mean inference time; FPGA-first pairs added 184.931 ms. Alternating
order therefore did not remove the observed slowdown.

## Selected operation and amount of work

The runtime trace identified the actual **first** QK operation as:

`/layers.1/blocks.0/attn/MatMul_output_0`

Inputs: `[16,4,49,32]` and `[16,4,32,49]`. Output shape is
`[16,4,49,49]`. There are 16 attention windows and four heads per window:
64 independent 49x32 by 32x49 matrix products.

| Work statistic | Per full inference |
| --- | ---: |
| Independent matrix products | 64 |
| M/N/K tile counts per product | 4 / 4 / 2 |
| 16x16x16 FPGA jobs per product | 32 |
| FPGA jobs | **2048** |
| Useful multiply-accumulates | 4,917,248 |
| Submitted MACs including padding | 8,388,608 |
| Useful fraction of submitted MACs | 58.62% |
| Logical operand/result buffer payload | 1,572,864 bytes (1.5 MiB) |

Every FPGA-mode inference was checked for exactly 2048 completed jobs, and
every CPU-mode inference for zero. All runs returned exit code zero and
produced 1000 finite scores. Across the 20 measured offload inferences,
**40,960 physical FPGA jobs** completed. Two additional unmeasured offload
warm-up processes completed 4096 jobs, giving 45,056 jobs including warm-ups.

Earlier reports selected `/layers.3/blocks.1/attn/MatMul_output_0`, a later
QK operation with only 320 jobs. Those results must not be substituted for
this first-QK benchmark: this operation submits 6.4 times as many jobs.

## Test method and timing boundaries

- 20 distinct random 224x224 RGB images, stored as lossless P6 PPM files.
- Each channel is independently sampled as a uniform integer from 0 to 255;
  seed base 2026111100, measured image seeds 2026111101 through 2026111120.
- Each CPU/FPGA pair uses the exact same image bytes. Image SHA256 values
  are recorded in `results.json`.
- Both modes decode the image and normalize `(pixel/255 - mean)/std`, using
  mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`.
- Two CPU threads, batch one, full `tiny-vit.mnn` model and the same static
  `tinyvit_offload` executable for both modes.
- Twenty measured runs per mode: **40 measured full-model inferences**.
- Two unmeasured warm-up pairs use fresh processes to warm OS/model caches.
- Every inference uses a fresh process and session, one run and no session
  warm-up. This avoids the previously observed session-reuse output drift.
- Pair order alternates: 10 CPU-first and 10 FPGA-first pairs. No other
  inference process runs concurrently.
- Both modes run with the same root privileges. CPU mode clears FPGA device
  environment variables and uses its normal `runSession` path. FPGA mode
  enables `/dev/mem`, selects the first QK and requires completed FPGA jobs.
- CPU-frequency/governor values were not exposed by the attempted sysfs
  paths. No CPU-frequency or affinity setting was changed for this test.

**Full-model inference time** is the runner's measured session execution.
It includes all model operations and FPGA selection callbacks, quantization,
PIO transactions, polling, diagnostic logging and host accumulation in offload
mode. It excludes model loading, image preprocessing and output copying.

**Complete process wall time** measures from starting the runner subprocess
until it exits. It additionally includes model/session creation, image decode,
normalization, output copying, score/metadata saving and process teardown.
Both modes save the same output types. This process timer uses a blocking
wait rather than timeout polling, avoiding polling quantization of the wall
measurement. Random-image generation, post-run score comparison and serial
downloads occur outside both timers. The complete process figures describe
this standalone executable workflow, not a persistent inference service.

The roughly 455 ms outside the inference timer is common to both modes:
mean process-minus-inference time is 455.228 ms for CPU and 456.649 ms for
offload. That is why the same approximately 185 ms penalty becomes a smaller
percentage when model loading/preprocessing/output handling are included.

## Numerical checks

| Final score comparison over 20 images | Result |
| --- | ---: |
| CPU/FPGA top-1 agreement | 20 / 20 |
| Identical top-5 ranking | 19 / 20 |
| Mean maximum absolute score error | 0.003661 |
| Worst maximum absolute score error | 0.006736 |
| Mean cosine similarity | 0.999999327 |
| Minimum cosine similarity | 0.999997681 |

The FPGA path is approximate INT8 arithmetic; the CPU path is FP32. These
are random-noise images without labels. Score agreement is a numerical check,
not an image-classification accuracy result. The normalization values are
explicit benchmark settings, not a new validation of the model's export pipeline.

## Interpretation

For this implementation and first QK operation, offloading is a reproducible
performance degradation. The integrated FPGA is functionally operating at
90 MHz, but that does not make the full software/hardware path faster than
the optimized HPS CPU implementation.

The offload uses 2048 small jobs, each with fresh quantization, register setup,
two operand transfers, polling, result reads and host accumulation. The fixed
padding discards 41.38% of submitted MAC positions. These are real costs of
the current mechanism. The benchmark establishes their combined net impact;
it does not separately time the selected CPU MatMul or each offload stage,
so it cannot assign the 184 ms penalty to a particular component.

Useful next optimizations to evaluate are reducing job count, reusing operand
tiles, reducing software quantization/conversion overhead, and a transfer or
output interface that requires less CPU work. Their benefit still needs
measurement. Raising the clock from 80 to 90 MHz alone has not eliminated
the overhead in this integration.

## Every measured pair

Negative time reduction means increased latency with offload.

| Pair | First | CPU inference ms | FPGA inference ms | Time reduction | CPU process ms | FPGA process ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | CPU | 1162.43 | 1349.75 | -16.11% | 1616.15 | 1803.11 |
| 2 | FPGA | 1160.35 | 1355.80 | -16.84% | 1614.21 | 1810.21 |
| 3 | CPU | 1161.20 | 1343.09 | -15.66% | 1624.11 | 1798.84 |
| 4 | FPGA | 1160.54 | 1344.28 | -15.83% | 1615.19 | 1798.04 |
| 5 | CPU | 1161.12 | 1343.88 | -15.74% | 1614.86 | 1798.22 |
| 6 | FPGA | 1166.56 | 1343.74 | -15.19% | 1620.59 | 1808.29 |
| 7 | CPU | 1161.51 | 1345.96 | -15.88% | 1615.65 | 1802.95 |
| 8 | FPGA | 1162.12 | 1350.16 | -16.18% | 1615.94 | 1809.48 |
| 9 | CPU | 1169.26 | 1348.04 | -15.29% | 1625.06 | 1801.79 |
| 10 | FPGA | 1167.48 | 1345.33 | -15.23% | 1622.03 | 1799.38 |
| 11 | CPU | 1164.41 | 1345.31 | -15.54% | 1618.75 | 1800.23 |
| 12 | FPGA | 1162.77 | 1365.73 | -17.45% | 1617.73 | 1820.16 |
| 13 | CPU | 1162.67 | 1345.15 | -15.69% | 1617.80 | 1806.02 |
| 14 | FPGA | 1168.47 | 1350.73 | -15.60% | 1623.28 | 1805.67 |
| 15 | CPU | 1162.91 | 1347.01 | -15.83% | 1619.52 | 1802.34 |
| 16 | FPGA | 1163.60 | 1344.34 | -15.53% | 1618.33 | 1818.50 |
| 17 | CPU | 1167.14 | 1349.73 | -15.64% | 1621.98 | 1804.67 |
| 18 | FPGA | 1163.02 | 1347.69 | -15.88% | 1622.77 | 1802.80 |
| 19 | CPU | 1165.23 | 1354.81 | -16.27% | 1619.36 | 1809.03 |
| 20 | FPGA | 1169.35 | 1345.77 | -15.09% | 1623.38 | 1799.55 |

## Build identity, artifacts and reproduction

The validated 90 MHz image was explicitly reloaded over JTAG before these
tests. Linux booted while skipping the stock FPGA-loading script.

- FPGA image SHA256:
  `0af66d4d24302b1ff9bb960ab189d41850603be1e8fd16edae29206fedfea058`.
- Runner SHA256:
  `a22ffd1bae738a250dc0910ef41c1370b60d5a180a81d8e3112e99dcdaf748c7`.
- Model SHA256:
  `f7aee628e2cef2d726979cc259529b3af0778b429c52ba8b76f3c609a2996de4`.
- Board data: `/home/ubuntu/tinyvit/benchmark-random-qk-90mhz/run-20pairs/`.
- Downloaded data: `build/benchmark-random-qk-90mhz/`, including `results.json`,
  `metadata.json`, `pairs.csv`, `summary.json`, `pairs.md`, `board-run.log`,
  and standalone charts `timings.png` / `timings.svg`.
- Raw evidence archive: `evidence-without-images.tar.gz`, including individual
  CPU/FPGA logs and scores, warm-up runs and runtime operation discovery.
  Images remain on the board and are reproducible from the recorded seeds.
- JTAG programming log: `build/random-qk-90mhz-program.log`.

The board's RTC reports 2016; benchmark durations use monotonic timers and
are independent of that incorrect wall-clock date. The actual test date is
2026-11-11.

After loading the matching image and deploying the script, run on the board
with a new output directory:

```sh
sudo python3 benchmark_board_random_qk.py \
  --output-dir /home/ubuntu/tinyvit/benchmark-random-qk-90mhz/NEW_RUN \
  --pairs 20 --seed 2026111100 \
  --bitstream-sha256 0af66d4d24302b1ff9bb960ab189d41850603be1e8fd16edae29206fedfea058
```

On the host, after downloading the JSON files:

```sh
python mnn/app/summarize_paired_benchmark.py build/benchmark-random-qk-90mhz --plot
```

Related: [90 MHz build verification](matmul-soc-90mhz-verification.md) and
[integration guide](matmul-soc-integration.md).
