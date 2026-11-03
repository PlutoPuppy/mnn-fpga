# TinyViT FPGA offload integration report

Date: 2026-11-03. Target: connected DE1-SoC, HPS Linux and Cyclone V FPGA.

## Outcome

The full TinyViT model successfully runs with one selected attention MatMul
offloaded to the FPGA and the remaining model operations on the HPS CPU.
Both tested inputs completed with 320 physical FPGA submissions and exit code 0.
Six independent physical matrix smoke cases also passed, totaling 48 submissions.

This establishes functional integration. It does not establish a consistent
full-model speedup or image-dataset accuracy. The offload uses approximate INT8
arithmetic; the CPU reference is FP32.

## 1. Hardware and software configuration

| Item | Configuration |
| --- | --- |
| Board | DE1-SoC |
| FPGA | Cyclone V, 5CSEMA5F31C6; JTAG ID 02D120DD |
| HPS | ARM Cortex-A9; runner configured for two CPU threads |
| Operating system | Ubuntu 16.04.1, Linux 4.5.0-00183-g4647b69-dirty |
| Board connection | COM8, 115200 baud; JTAG DE-SoC USB-1 |
| Deployment constraints | No network address, UIO device, or board C++ compiler |
| FPGA tools | Quartus Prime Lite 23.1std.1, Build 993 |
| Software cross-build | Zig 0.13, Clang 18, static ARMv7 hard-float, Cortex-A9 NEON |
| MNN source | Pinned commit 2c3f5b21a223734e5db7da4ef7b492bc02cafcf5 |
| Friend's hardware source | j3rryhu/matmul-soc, commit 4b749cedf88742dc999267999cd234186eb26bf7 |
| Model | tiny-vit.mnn; input [1,3,224,224], output [1,1000], FP32 |
| Final fabric clock | 80 MHz; 12.5 ns clock period |

The friend's actual project is `adder_prototype.qpf`, revision
`adder_prototype`, top module `de1soc_top`, system `adder_prototype.qsys`.
The isolated build checkout on this PC is `%TEMP%/mnn-matmul-soc`.

The accelerator has a physical 16x16 PE array, signed INT8 operands and INT32
internal partial sums. Its externally visible output is rescaled, saturated
INT8. The 256x256 output buffer therefore contains 65,536 bytes (64 KiB),
not 65,536 INT32 values. Each operand buffer is 16 KiB.

## 2. Selected TinyViT operation

Selected runtime operation:

`/layers.3/blocks.1/attn/MatMul_output_0`

MNN exposes this operation as a geometry-lowered `While` command. The dense
matrix execution hook inside that command handles ten attention heads:

| Quantity | Shape or count |
| --- | --- |
| Left input | [1,10,49,32] |
| Right input | [1,10,32,49] |
| Logical result | [1,10,49,49] |
| Per-head multiplication | [49,32] x [32,49] -> [49,49] |
| Head count | 10 |
| Useful output values | 24,010 |
| Useful multiply-accumulates | 768,320 across all heads |

One named model operation is selected, but it contains ten individual matrix
products. The rest of TinyViT remains on the CPU.

## 3. Tiling mechanism and work statistics

The driver accepts general dense `[M,K] x [K,N]` multiplication and submits
independent 16x16x16 hardware jobs. Sizes need not be multiples of 16: missing
rows, columns and reduction elements are padded with zero. Transpose options
and optional bias are supported. Partial results across K tiles are
dequantized and accumulated on the host before the final FP32 output is stored.

For one matrix product, the job count is:

`ceil(M/16) * ceil(N/16) * ceil(K/16)`

For the selected operation:

| Tiling statistic | Per head | All 10 heads |
| --- | ---: | ---: |
| M tiles: ceil(49/16) | 4 | — |
| N tiles: ceil(49/16) | 4 | — |
| K tiles: ceil(32/16) | 2 | — |
| Logical output tile positions | 16 | 160 |
| FPGA jobs | 32 | 320 |
| Useful MACs | 76,832 | 768,320 |
| Submitted padded MACs | 131,072 | 1,310,720 |
| Useful fraction of submitted MACs | 58.62% | 58.62% |

The effective padded matrix dimensions are M=64, N=64, K=32. The fourth
row/column tile contains only one valid row/column. The 58.62% statistic is
derived padding efficiency, not a measured PE occupancy or bandwidth figure.

For each job, software writes 256 INT8 bytes for A and 256 for B, then reads
256 INT8 result bytes: 768 logical payload bytes. This totals 245,760 bytes
(240 KiB) for 320 jobs, excluding register writes, polling, bus framing,
host-memory traffic and software staging. Inputs are resent for each job;
the current implementation does not cache operands across jobs or use DMA.

The driver uses a 16x16 region of the output canvas per job and accumulates
reduction tiles on the CPU. It does not allocate the whole model result in
the physical 256x256 canvas. Consequently larger logical outputs can be
handled through repeated jobs, subject to host memory and runtime limits.

## 4. Full-model timing measurements

Each entry below is one measured inference, with two CPU threads, one run,
and no configured warm-up. Timing excludes model loading, preprocessing,
output copying and serial transfer. FPGA-mode inference timing includes its
software tile preparation, quantization, PIO, polling and result accumulation.

Speedup is `CPU-only time / offload time`. Time reduction is
`(CPU-only time - offload time) / CPU-only time`.

| Input | CPU only (ms) | One operation offloaded (ms) | Time saved (ms) | Speedup | Time reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic zeros | 1265.17 | 1192.56 | 72.61 | 1.0609x | 5.74% |
| Nonuniform synthetic tensor | 1230.07 | 1263.35 | -33.28 | 0.9737x | -2.71% |

The second case took 2.71% longer with offload. These are separate single-run
observations, not a controlled repeated benchmark; no timing variance or
confidence interval was measured. The evidence does not support a reliable
speedup claim. The nonuniform input is deterministic NCHW data generated as
`0.5*sin(0.11*x) + 0.25*cos(0.07*y) + 0.1*c`, not a validation-set image.

An older CPU runner took 2169.86 ms on zero input. The rebuilt CPU runner
returned bit-identical scores in 1265.17 ms. That change came before FPGA
offload and must not be attributed to FPGA acceleration. The rebuilt CPU
runner is the relevant comparison in the table above.

The following were not measured separately: CPU time for this MatMul alone,
FPGA compute-only latency, per-tile latency, quantization time, transfer time,
polling time, host accumulation time, energy or power. Dividing full-model
time by 320 would not give accelerator job latency.

## 5. Numerical behavior and full-model score comparison

Each FP32 input tile is symmetrically quantized with its own scale based on
maximum absolute value divided by 127. A bounded Q0.16 output multiplier
keeps the tile result within INT8 range. Hardware rescaling and INT8 output
introduce additional rounding error. Software reverses the scales and sums
K-tile results using double precision before producing FP32 output.

The selected operator's output is approximate. The rest of the model runs
normally and propagates that difference to the final classifier scores.

| Final 1000-score comparison | Zero input | Nonuniform input |
| --- | ---: | ---: |
| Maximum absolute error | 0.0131550 | 0.0200701 |
| RMSE | 0.00483614 | 0.00516934 |
| Relative L2 error | 0.0130169 (1.3017%) | 0.00320218 (0.3202%) |
| Cosine similarity | 0.99992987 | 0.99999541 |
| CPU and FPGA top-1 index | 674 | 729 |
| CPU and FPGA top-5 indices | 674,593,749,872,836 | 729,789,753,556,811 |
| Top-5 ranking agreement | Yes | Yes |

These are raw model score differences, not probability differences or
classification-accuracy percentages. Agreement on two synthetic inputs
does not establish accuracy on real images or a dataset.

## 6. Physical matrix tests

All six tests use M=17 and N=31, exercising padding in both output dimensions.
The K=33 cases also exercise a padded reduction tile.

| A x B | Input pattern | FPGA jobs | Maximum absolute error | Result |
| --- | --- | ---: | ---: | --- |
| 17x16 x 16x31 | Ones | 4 | 0.000991821 | PASS |
| 17x16 x 16x31 | Signed nonuniform | 4 | 0.296904 | PASS |
| 17x32 x 32x31 | Ones | 8 | 0.00198364 | PASS |
| 17x32 x 32x31 | Signed nonuniform | 8 | 0.505920 | PASS |
| 17x33 x 33x31 | Ones | 12 | 0.00979614 | PASS |
| 17x33 x 33x31 | Signed nonuniform | 12 | 0.506165 | PASS |
| Total | | 48 | | PASS |

Smoke errors were checked against a conservative quantization-error bound,
not exact equality. Software-model tests additionally covered transpose
combinations, 32x48x64 rectangular problems, 17x31x33 tails, 272-row outputs,
zero/nonfinite input, busy hardware and output preservation after failure.
Those broader cases passed on the host and in the static ARM test binary;
they are software-model coverage, not all physical FPGA tests.

Runtime fallback tests also passed: an unavailable UIO node caused one device
diagnostic followed by CPU fallback and bit-identical CPU scores. The
`--require-fpga` guard correctly failed when no hardware jobs completed.

## 7. Quartus build and resource statistics

| Final fitted resource | Used | Available | Reported utilization |
| --- | ---: | ---: | ---: |
| ALMs | 23,908 | 32,070 | 75% |
| Registers | 25,968 | — | — |
| Block memory bits | 2,155,472 | 4,065,280 | 53% |
| RAM blocks | 378 | 397 | 95% |
| DSP blocks | 2 | 87 | 2% |
| Pins | 107 | 457 | 23% |

These are whole-project totals, including interconnect and debug infrastructure,
not isolated PE-array utilization. Low DSP usage does not mean only two PEs
exist; synthesis can implement arithmetic in logic.

The final image runs at 80 MHz. All 88 reported STA summary checks passed;
minimum reported slack was 0.077 ns and slow 85 C core setup slack was 2.049 ns.
The earlier build reported -0.515 ns setup slack and was not programmed.
Investigation found the project referenced an old exported SDC with fixed PLL
dividers that constrained the core at 100 MHz. The final profile replaces it
with a 50 MHz board reference and clocks derived from the regenerated PLL.

The profile also corrects placeholder HPS DDR geometry from 8-bit,
row12/column8 to the board's 32-bit, row15/column10 configuration. Existing
board bootloader settings and live DDR timing/geometry were retained.

Native HPS ROM generation initially failed because Intel's launcher required
unavailable WSL. A temporary installation view and native Makefile-path
conversion solved generation without modifying installed Quartus files.

## 8. Integration changes

| Area | Change |
| --- | --- |
| GEMM driver | General rectangular tiling, tails, transpose/bias, quantization and staged output |
| Physical transport | ARMv7 NEON 128-bit buffer accesses; ordered register accesses; UIO or explicit /dev/mem |
| MNN bridge | Dense execute hook suitable for the selected lowered While operation |
| Runner | --list-ops, exact --fpga-op selection, --require-fpga and submission counts |
| Failure handling | Device locking, validation, timeout and CPU fallback |
| FPGA build | Board profile, native HPS generation, derived clocks and timing gate |
| Validation tools | Physical smoke test, software-model tests, serial transfer with SHA256, score comparator |

The original repository's exact integer-valued FP32/INT32-result demonstration
was preserved separately. Its register ABI is not interchangeable with the
friend's INT8-output design. Buffer transfers for the friend's design require
full 128-bit beats because RAM wrappers ignore byte enables.

## 9. Board bring-up and repeatable boot procedure

Linux initially stopped responding during live bridge isolation. After HPS
reset, a smoke attempt also stalled. Direct JTAG inspection and U-Boot script
inspection established that normal boot loads the stock `soc_system.rbf`,
replacing the accelerator image. The stock image must not be accessed with
this accelerator's driver. The live isolation helper remains experimental.

The successful temporary sequence was:

1. Press KEY5 / HPS_RESET_N and interrupt U-Boot's five-second countdown over
   COM8 at 115200 baud.
2. At the U-Boot prompt, program `matmul-soc-80mhz.sof` via JTAG device 2.
3. Run `run bridge_enable_handoff; run mmcload; run mmcboot` to boot Linux
   without the stock-image script.
4. Log in, enter `/home/ubuntu/tinyvit`, run the matrix smoke test, then TinyViT.

No saved boot environment or SD-card boot files were changed. A normal reboot
will again load the stock image; repeat this sequence before using the driver.
The original board model and `tinyvit_run` were preserved. The board remained
responsive after successful tests, and outputs were flushed with `sync`.

Example validated inference command after that boot sequence:

```sh
sudo env MNN_MATMUL_SOC_DEVMEM=1 ./tinyvit_offload \
  --model tiny-vit.mnn --height 224 --width 224 --zeros --threads 2 --runs 1 \
  --fpga-op /layers.3/blocks.1/attn/MatMul_output_0 --require-fpga \
  --output-dir outputs/fpga-zero
```

## 10. Artifacts and evidence

- Deployment files: `build/matmul-soc-deploy/`, containing the static runner,
  compressed runner, physical smoke executable, FPGA image, README and SHA256SUMS.
- FPGA image SHA256:
  `8e0320ac525a3062fc9f6f13402bb2eaf514f527d34a113849467303c2dff2d7`.
- Quartus reports: `build/matmul-soc-reports/80mhz-derived/`.
- Zero-input CPU/FPGA scores: `build/board-cpu-rebuilt/` and `build/board-fpga-zero/`.
- Nonuniform CPU/FPGA scores: `build/board-synthetic-cpu/` and `build/board-synthetic-fpga/`.
- Both FPGA result directories contain downloaded run logs and `comparison.json`.
- Build artifacts are generated/ignored; source changes and this report are local
  repository changes. The integration work did not push changes to GitHub.

## 11. Remaining performance work

The present implementation proves that the full-model dispatch, tiling,
hardware interface and result reintegration work. To evaluate acceleration:

1. Repeat CPU and offload runs on identical inputs with warm-up, alternating
   modes and reporting median, mean, minimum and variability.
2. Profile the selected CPU MatMul and separately measure quantization, writes,
   FPGA execution/polling, reads and host accumulation.
3. Measure potential improvements from operand reuse, fewer submissions,
   hardware-side K accumulation or a higher-precision output path, and DMA.
4. Evaluate numerical accuracy on real preprocessed images and a labeled dataset.

The selected operation performs only 768,320 useful MACs. With 320 small PIO
jobs and substantial padding, software and transfer overhead are plausible
constraints, but their individual costs have not yet been measured.

Related sources: [integration guide](matmul-soc-integration.md),
[validation record](validation.md), and
[upstream hardware](https://github.com/j3rryhu/matmul-soc/tree/4b749cedf88742dc999267999cd234186eb26bf7).
