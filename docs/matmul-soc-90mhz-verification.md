# Integrated matmul-soc verification at 90 MHz

Verified on 2026-11-11: the integrated MatMul FPGA image meets the corrected
90 MHz timing constraints and works on the connected DE1-SoC board.
The working 80 MHz image was preserved.

## Static timing results

| Check | Result |
| --- | --- |
| Generated PLL request | 90.000000 MHz |
| Fitted core clock period | 11.111 ns (about 90 MHz) |
| Reported STA summary checks | 88, all passed |
| Minimum slack across all reported checks | +0.077 ns |
| Worst core setup slack across reported corners | +1.565 ns |
| Core setup slack, slow 85 C | +1.668 ns |
| Worst core hold slack across reported corners | +0.105 ns |
| STA errors | 0 |

This is a new generation, synthesis, fit, assembly and STA run against the
pinned upstream hardware commit `4b749cedf88742dc999267999cd234186eb26bf7`.
The isolated project is `%TEMP%/mnn-matmul-soc-90mhz/adder_prototype.qpf`.
The profile differs from the validated 80 MHz profile only in these module
parameters: PLL output frequency, F2H AXI clock frequency and F2H SDRAM0 clock
frequency. No accelerator RTL changes or new timing exceptions were added.

The SDC defines the 50 MHz board reference and derives the actual PLL clocks
and uncertainties. It does not load the stale exported SDC that previously
constrained the core at 100 MHz. An additional fitted-netlist check verified
the expected 90 MHz clock and saved detailed setup and hold paths.

The router warned of congestion and possible timing problems during fitting.
The completed STA summary nevertheless has positive slack in every reported
check. Synthesis took 11m45s, fitting 27m56s, assembly 1m17s and STA 3m55s.

| Final fitted resource | Usage |
| --- | --- |
| ALMs | 23,826 / 32,070 (74%) |
| Registers | 25,813 |
| RAM blocks | 378 / 397 (95%) |
| Block memory bits | 2,155,472 / 4,065,280 (53%) |
| DSP blocks | 2 / 87 (2%) |

## Physical board results

Linux was rebooted and stopped at U-Boot. The 90 MHz image was programmed
over JTAG to FPGA device 2. Linux then booted using
`run bridge_enable_handoff; run mmcload; run mmcboot`, avoiding the stock
`soc_system.rbf` loader. No saved boot settings were changed.

All six physical matrix smoke cases passed: M=17, N=31, K=16/32/33, using
ones and signed patterns. They completed 48 FPGA jobs with the same numerical
errors as the earlier 80 MHz smoke tests.

The full TinyViT validation selected
`/layers.3/blocks.1/attn/MatMul_output_0`: ten 49x32 by 32x49 products,
32 tile jobs per head, 320 jobs per inference. Other operations ran on CPU.

Five fresh processes were run per input. All ten returned exit 0 and exactly
320 FPGA submissions each: 3,200 jobs across these full-model checks.
All five zero-input score files have the same SHA256 as the saved 80 MHz
zero-input result; all five nonuniform-input score files likewise match the
80 MHz nonuniform result. Downloaded representative outputs compare with
maximum absolute error **0.0**, RMSE **0.0** and cosine similarity **1.0**
against the corresponding 80 MHz outputs.

| Fresh-process inference time | Zero input | Nonuniform synthetic input |
| --- | ---: | ---: |
| Run 1 (ms) | 1212.54 | 1196.93 |
| Run 2 (ms) | 1194.48 | 1181.36 |
| Run 3 (ms) | 1184.74 | 1186.10 |
| Run 4 (ms) | 1183.70 | 1195.09 |
| Run 5 (ms) | 1181.83 | 1222.98 |
| Mean (ms) | 1191.458 | 1196.492 |
| Median (ms) | 1184.74 | 1195.09 |
| Sample standard deviation (ms) | 12.763 | 16.130 |

These times exclude model loading, preprocessing and output copying, but
include inference-time offload overhead. Each measured inference used a
fresh process/session with two CPU threads and no warm-up. The older 80 MHz
measurements were taken in a different session on a different day, so this
table does not establish an 80-to-90 MHz speedup or a new CPU speedup.

## Session-reuse issue found during validation

An initial test used two warm-ups and five measured inferences in one MNN
session per input. Both FPGA runs completed all measured 1,600 submissions,
but their final scores differed from the fresh-process baseline. CPU-only
session reuse also changed the zero-input output, despite zero FPGA jobs.
Fresh CPU and FPGA processes reproduced their original single-run results.

The root cause of this session-reuse behavior has not been established or
fixed in this clock-verification task. Warmed session-reuse timing runs must
not be treated as validated identical-input benchmarks until that behavior
is resolved. The clock-verification conclusion instead uses physical matrix
tests and ten fresh-process full-model runs with exact 80 MHz result hashes.
Diagnostic CPU and FPGA logs are preserved under `build/board-90mhz/`.

## Reproduction and evidence

From the repository root, against a clean isolated upstream checkout:

```sh
python platform/apply_matmul_soc_profile.py PATH/adder_prototype.qsys --profile platform/de1-soc-matmul-90mhz-profile.json
python platform/build_matmul_soc.py --friend-root PATH --native-hps-launcher
```

After the build, from that Quartus project directory:

```sh
quartus_sta -t PATH_TO_REPO/platform/verify_matmul_clock.tcl adder_prototype 90
```

Artifacts on this PC:

- Image: `build/matmul-soc-deploy/matmul-soc-90mhz.sof`.
- Image SHA256:
  `0af66d4d24302b1ff9bb960ab189d41850603be1e8fd16edae29206fedfea058`.
- STA summary, detailed reports, fitted clocks, core paths, Qsys/SDC snapshots
  and programming log: `build/matmul-soc-reports/90mhz-corrected/`.
- Physical smoke log, ten-run hash/count/timing evidence and computed
  statistics: `build/board-90mhz/`.
- Representative 90 MHz scores: `build/board-90mhz/zero/` and
  `build/board-90mhz/synthetic/`.
- Preserved fallback image: `build/matmul-soc-deploy/matmul-soc-80mhz.sof`.

The board is left running Linux with the 90 MHz image loaded. Normal reboot
still restores the stock FPGA image; repeat the documented U-Boot/JTAG
procedure to load this accelerator again. This verifies the integrated
MatMul image and does not independently validate a separate softmax module.
