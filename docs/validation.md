# Validation record

## Native INT32 reduction and resident MLPs, 2026-11-14

The separate NI82 project fits the DE1-SoC: 22,686 ALMs, 367/397 M10Ks,
35 DSP blocks. All 88 evaluated timing checks pass at 90 MHz, minimum
slack +0.077 ns; remaining unconstrained external HPS/JTAG ports are listed
in the [resident offload report](tinyvit-native-resident-offload.md).
The qualified SOF was programmed through JTAG. Ten physical signed-matrix
and resident-pair cases pass an independent INT64 oracle (221,760 elements),
including K=1024, output-capacity and tail checks. Actual output-stage RTL
passes 126 CXXRTL cases/99,974 elements; host banked-driver tests pass 44
cases and scalar/ARM NEON layout tests pass 72 cases each.

Twenty random images completed 160 fresh-process control/profile inferences.
All 30 supported INT8 linears execute on FPGA; four MLP intermediates remain
in FPGA RAM. Resident/unfused outputs are byte-identical on every image,
as are each configuration's control/profile outputs. Resident FPGA agrees
with mixed CPU top-1 on 20/20 (mean logit cosine 0.997552), with arithmetic
rounding differences described in the report. This repairs the legacy
per-K16-band saturation path without changing the corrected mixed model.

Mean inference is 1155.982 ms FP32 CPU, 1080.605 ms mixed CPU, 1048.245 ms
unfused FPGA and 1006.955 ms resident FPGA. Resident inference is 12.89%
below FP32 and 6.82% below mixed CPU; startup-inclusive wall time is
1522.157 ms versus 1624.037/1508.985 ms. Setup therefore still matters for
a fresh one-image process. Raw evidence, phase details and additional
selection/natural-image checks are linked from the report.

The selected-22 policy and natural-image check add 120 and 160 measured
inferences. On 20 labeled natural images FP32, mixed CPU, resident-30 and
resident-22 make identical top-1 predictions and each scores 16/20; this is
a small reused check, not full ImageNet accuracy. The resident-30 natural
mean is 1011.565 ms versus 1167.530 ms FP32 and 1090.452 ms mixed CPU.

A final session-reuse regression found that the v10 runner did not refresh
input between iterations. v11 fixes this outside the inference timer and
reports repeated-copy time separately. CPU and resident FPGA runs=1/2/3
checks pass against the saved fresh-process outputs (twelve additional
inferences). Primary timings remain explicitly attributed to v10 single-run
processes; the qualified 90 MHz image is unchanged.

## Native W8A8 fc1 experiment, 2026-11-12

MNN 3.6.2 calibrated the board's SHA-verified FP32 TinyViT using 128 real images.
The broad artifact has 68 quantized convolutions and 351 quantized tensor records;
an additional control model quantizes only fc1 and its two activation records.
The original model and 90 MHz FPGA image were preserved.

Twenty saved random images ran in ten balanced control/profile modes:
200 measured full-model inferences, plus ten warmups. All 200 downloaded output
hashes/logs checked; all 20 FP32 outputs match the earlier baseline. Each FPGA
inference replaces only fc1, with 28 jobs and 401408 NEON-rescaled INT8 outputs.
Driver activation quantization and FP32 reconstruction phases are zero. MNN
framework casts elsewhere remain measured separately.

Mean inference: FP32 CPU 1161.753 ms; broad W8A8 CPU 850.314 ms, FPGA 868.014 ms
(+2.08%, 19/20 slower); fc1-only W8A8 CPU 1166.908 ms, FPGA 1184.661 ms (+1.52%,
18/20 slower). The latter mean CI crosses zero due to a retained CPU outlier.
Native INT8 CPU fc1 takes 21.284/21.784 ms; complete FPGA paths 41.538/42.470 ms.

Host/native ARM tests and the NEON-versus-scalar/INT64 rounding oracle pass.
Independent full INT32 fc1 agrees with native MNN within one code on three
values in the broad model and exactly in the fc1-only model; hardware matches
the separate banded arithmetic oracle exactly. Numerical equivalence to full
INT32 does not pass because the RTL floors/saturates each K=16 band. Broad
quantization also fails the 20-real-image host agreement check (3/20 versus
20/20 for fc1-only). These are experiments, not deployment accuracy validation.
See [the full W8A8 report](tinyvit-w8a8-fc1-benchmark.md).

## Capacity-aware fc1 experiment, 2026-11-12

The corrected runner completed 20 random images in six CPU/fixed/capacity
control/profile modes: 120 measured full-model inferences, plus six warmups.
Every offload replaced only stage-1 block-0 fc1, with 16 Fixed64 or 28 capacity
jobs. All 120 downloaded score hashes matched their run records; all 20 CPU
outputs matched the prior baseline. Host tests and ARM warnings-as-errors
builds passed. Physical K=128/K=256 and full fc1 shape smoke checks passed.

Capacity selects 128x128x128 tiles. Mean full inference was 1192.008 ms CPU,
1242.918 ms fixed, 1223.313 ms capacity. A retained 1885.230 ms CPU outlier
makes the +31.305 ms / +2.63% mean capacity penalty uncertain (95% paired CI
-42.833 to +70.178 ms). Capacity was slower in 19/20 pairs; median paired
penalty was 64.400 ms. Complete fc1 was consistently slower: 28.653 ms native
CPU versus 91.405 ms capacity, of which 2.925 ms was execution plus polling.

Explicit MNN layout copies fixed an initial packed-buffer interpretation bug;
the preliminary run is excluded. Independent FP32 fc1 matches native output
within 0.000003. Hardware matches a separate banded arithmetic oracle exactly,
but its repeated INT8 floor rounding produces large bias: the offload fails
numerical equivalence. Capacity's full-model top1 agrees with CPU on 12/20
random inputs, which is agreement on noise, not dataset accuracy. This stays
an opt-in experiment. See [the full report](tinyvit-fc1-capacity-benchmark.md).

## Whole-model CPU operator profiling, 2026-11-11

Twenty random images ran in fresh CPU control/profile pairs, 40 measured
inferences plus two warmups. All 40 output hashes matched the prior CPU
baseline; every run reported zero FPGA jobs. The stable runtime catalog
contained 339 command intervals per inference, covering 99.19% of profiled
inference time. Controls averaged 1167.7915 ms; profiling averaged 1175.514917 ms.

Static-weight 1x1 convolutions accounted for 678.026 ms (57.68%). Stage-1 MLP
fc2 projections took about 35.6 ms each; fc1 took about 28.5 ms, versus 4.70 ms
for the first QK in this profiler. The report includes equivalent GEMM shapes,
execution frequency, current planner job counts, layout requirements and a
shortlist. No new FPGA operator was enabled. ARM warnings-as-errors build,
smoke test, timing accounting, stable catalog and downloaded raw profile/score
hashes passed. See [the full CPU report](tinyvit-cpu-operator-profile.md).

## Repeated zero-input first-QK DMA diagnosis, 2026-11-11

The same 90 MHz image, model and profiling runners completed 20 all-zero FP32
input repetitions with balanced four-mode order, followed by CPU kernel probes:
120 measured inferences. Main control inference averaged 1173.6715 ms CPU and
1230.423 ms DMA, +56.7515 ms (+4.84%); DMA was slower in all 20 pairs. The paired
95% interval for added inference latency was 49.7725 to 63.819 ms.

CPU selected QK averaged 4.81962 ms, complete FPGA QK 58.56095 ms, and FPGA
execution plus polling 0.65852 ms. Input scan/quantization, transfer and result
reconstruction still dominate. All expected jobs and DMA counters passed;
each path's output was identical across repetitions and control/profile modes.
The old 5.74% single-run observation concerned a different attention operation
on the older 80 MHz PIO integration. See the
[full zero-input diagnosis and evidence](tinyvit-qk-zero-input-diagnosis.md).

## First-QK DMA timing diagnosis, 2026-11-11

The validated 90 MHz image ran 20 random RGB inputs with four fresh-process
modes (CPU/DMA control and CPU/DMA profile), followed by 20 paired CPU-only
kernel probes: 120 measured inferences. All expected jobs/bytes/descriptors
completed; control/profile output hashes matched, and all outputs matched the
preceding batched-DMA experiment for the corresponding execution path.

Detailed-timer-disabled inference averaged 1178.2065 ms CPU and 1224.513 ms DMA:
+46.3065 ms (+3.93%). CPU QK wall time was 4.82185 ms versus 59.10804 ms for
complete offloaded QK. FPGA completion including polling averaged 0.65722 ms;
CPU packed-kernel interval union averaged 4.19360 ms. Conversion and transfer
costs dominate. The report includes confidence intervals, process timing,
profiling perturbation, all 20 rows and raw evidence. See
[the full diagnosis](tinyvit-qk-dma-timing-diagnosis.md).

## Prepared INT8, larger hardware jobs and real DMA, 2026-11-11

- Reviewed upstream `b3ea448a36a5db7bd301173d4e5fff027eca2382`; new tests and
  SignalTap change only, no accelerator RTL/ABI changes over the validated image.
- Added prepared signed-INT8 API, once-per-operand FP32 preparation, larger
  256x64 by 64x256 hardware jobs, bulk PIO and explicit DMA transport.
- Host tests passed, including transposes, both output-panel axes, reduction
  tails, full canvas, prepared scale validation and failure-atomic output.
- Physical PIO and DMA suites passed: 14 cases and 18 hardware jobs each,
  including full 64 KiB canvas readback. Reported errors matched in both modes.
- New ARM runner deployed separately as `tinyvit_offload_batched`. First QK
  now completes in 64 submissions versus 2048 for the original runner.
- Twenty paired random-image tests per new transport passed; all 20 PIO/DMA
  output hashes matched, and CPU output hashes matched the earlier baseline.
- New mean inference penalties: bulk PIO +49.810 ms / +4.238%; DMA
  +45.013 ms / +3.817%, each versus its own paired CPU baseline. Both remain
  slower than CPU. Complete-process timings were noisy and inconclusive.

See [prepared INT8 and DMA](matmul-soc-batched-int8-dma.md) for the interface,
memory constraints, commands, numerical checks and benchmark evidence.

## First-QK random-image paired benchmark, 2026-11-11

- Latest validated 90 MHz bitstream explicitly reloaded before measurement.
- First runtime QK: `/layers.1/blocks.0/attn/MatMul_output_0`, shapes
  `[16,4,49,32]` and `[16,4,32,49]`; 2048 FPGA jobs per inference.
- Twenty distinct seeded random RGB images, 20 paired CPU/offload tests,
  alternating order, fresh processes, two CPU threads. All 40 measured
  inferences passed; 40,960 measured FPGA jobs completed.
- Mean full-model inference: CPU 1164.107 ms, offload 1348.315 ms:
  **184.208 ms / 15.824% more time**. Offload slower in all 20 pairs.
- Mean complete process time: CPU 1619.335 ms, offload 1804.964 ms:
  **185.629 ms / 11.463% more time**.
- Paired-bootstrap 95% interval for added inference latency:
  181.70 to 187.09 ms. Top-1 score ranking agrees on 20/20 noise images;
  this does not establish classification accuracy.

See [the full paired benchmark](tinyvit-first-qk-90mhz-benchmark.md) for all
20 rows, timing boundaries, numerical checks, source scripts and evidence.

## Corrected 90 MHz integrated build, 2026-11-11

- Fresh upstream generation, synthesis, fitting, assembly and STA: pass.
- Fitted PLL output verified at 11.111 ns (approximately 90 MHz).
- All 88 reported STA checks passed; minimum slack +0.077 ns, worst core
  setup +1.565 ns and worst core hold +0.105 ns across reported corners.
- JTAG programming, six physical matrix smoke cases (48 jobs), and ten fresh
  TinyViT processes (320 jobs each) passed on the board. All score hashes
  matched the corresponding 80 MHz outputs exactly for both test inputs.
- Reusing a session for warm-up/repeated inference changed scores in CPU-only
  and FPGA modes. Root cause remains unresolved; those runs are excluded
  from numerical validation and identical-input benchmark claims.
- The 80 MHz fallback image and saved boot settings were preserved.

See [90 MHz verification](matmul-soc-90mhz-verification.md) for measurements,
artifact locations and the exact clock profile.

## matmul-soc integration, 2026-11-03

- Upstream inspected at `4b749cedf88742dc999267999cd234186eb26bf7`.
- Located `adder_prototype.qpf`, `de1soc_top`, 16x16 INT8 array, and 64 KB
  INT8 output canvas. This differs from this repository's INT32-result ABI.
- Host tiled-driver tests: pass with GCC 7.4, warnings treated as errors.
- Static ARMv7/Cortex-A9 tiled-driver tests: compiled with Zig 0.13 / Clang 18,
  transferred over COM8 with independently verified SHA256, and passed on the
  connected board. These tests use a software register/datapath model.
- Actual board: Cyclone V JTAG ID `02D120DD`, Linux 4.5.0, Ubuntu 16.04,
  no UIO devices, no network address, and no installed C++ compiler.
- Existing full TinyViT runner: CPU zero-input smoke passed in 2169.86 ms,
  produced 1000 finite FP32 scores and zero FPGA submissions. Baseline scores
  and metadata downloaded with verified SHA256 to `build/board-cpu-before`.
- New hardware smoke-test executable cross-compiled, stripped, and deployed
  with SHA256 verification. Physical execution passed all six cases with 48
  submissions: 17x16, 17x32 and 17x33 inputs times 16/32/33x31, using ones
  and signed patterns. Maximum errors ranged from 0.000992 to 0.506165.
- Updated MNN patch applied successfully to the pinned upstream source.
  Full static ARM MNN library and TinyViT runner build: pass, with Cortex-A9
  NEON assembly and MNN's pthread pool enabled.
- Rebuilt runner deployed as `~/tinyvit/tinyvit_offload`: full zero-input CPU
  inference passed in 1265.17 ms; all 1000 FP32 scores are bit-identical to the
  earlier runner. This CPU timing change is not evidence of FPGA acceleration.
- Runtime trace identifies `/layers.3/blocks.1/attn/MatMul_output_0` as a While
  command multiplying `[1,10,49,32]` by `[1,10,32,49]`. The selected operator's
  ten matrix heads require 320 independent 16x16x16 hardware jobs with this
  driver's padding policy. Hardware-disabled `--require-fpga` correctly fails
  with zero submissions for this valid operation name.
- Actual missing-device fallback: pass. Selecting that operation with
  `/dev/uio999` produced one diagnostic, disabled further device attempts,
  completed CPU inference with zero FPGA jobs, and produced bit-identical
  scores (SHA256 `2ddf6697b353bc9f041bbd0cabff1fe50300b99727ada71c388e18311a505f5b`).
- Standard Quartus regeneration fails because the HPS ROM workflow invokes
  unavailable WSL. A temporary installation-view launcher with Windows path
  conversion now passes generation; installed Quartus files remain unchanged.
  The unmodified 90 MHz design fitted (23,743/32,070 ALMs, 378/397 RAM blocks,
  2/87 DSPs) and assembled, but failed setup timing at -0.515 ns. It was not
  programmed. Reports are in `build/matmul-soc-reports/90mhz`.
- A reproducible DE1-SoC profile now sets the fabric clock to 80 MHz and fixes
  upstream placeholder DDR geometry to the board's 32-bit, 1 GB interface.
  The running board's live DDR interface width was independently read as 32;
  its bootloader and live DDR configuration remain unchanged. The revised
  profile passed generation, fitting and assembly. Replacing the stale exported
  timing snapshot with derived PLL constraints passed all 88 STA summary checks:
  minimum slack 0.077 ns; slow 85 C core setup slack 2.049 ns. Reports are in
  `build/matmul-soc-reports/80mhz-derived`.
- JTAG programming of FPGA device 2 succeeded on 2026-11-03 at 14:01.
  Bitstream: `build/matmul-soc-deploy/matmul-soc-80mhz.sof`, SHA256
  `8e0320ac525a3062fc9f6f13402bb2eaf514f527d34a113849467303c2dff2d7`.
  Linux stopped responding during the preceding bridge-isolation command;
  serial Ctrl+C and FPGA reconfiguration did not restore the shell. An HPS
  reset recovered Linux. The next smoke attempt stalled because normal boot
  had loaded the stock FPGA image: `u-boot.scr` loads `soc_system.rbf`.
- Successful recovery procedure: interrupt U-Boot, program the accelerator
  using JTAG device 2, then `run bridge_enable_handoff; run mmcload; run mmcboot`
  to boot Linux without the stock-image script. No saved boot settings or SD
  files were changed. JTAG register readback and the physical smoke tests passed.
- Full TinyViT selected-operation offload passed on both zero input and a
  deterministic nonuniform synthetic input, exit 0 and **320 FPGA submissions**
  each. All other model operations run on CPU. Scores and run logs downloaded
  with verified SHA256 to `build/board-fpga-zero` and `build/board-synthetic-fpga`.
- Compared to matching CPU baselines, zero input: max absolute error 0.013155,
  RMSE 0.004836, relative L2 0.013017, cosine 0.99992987. Nonuniform input:
  max absolute error 0.020070, RMSE 0.005169, relative L2 0.003202,
  cosine 0.99999541. Top-5 rankings matched CPU for both inputs (top-1 674/729).
  These synthetic tests do not establish image-dataset accuracy.
- Single measured FPGA full-model runs: 1192.56 ms zero input and 1263.35 ms
  nonuniform input, versus earlier CPU 1265.17/1230.07 ms. These timings are
  not a controlled benchmark and do not establish a speedup.

See [the integration guide](matmul-soc-integration.md) for the numeric policy,
separate register ABIs, build instructions and explicit validation limits.

## DSP capacity fix, 2026-10-12

The user's subsequent GUI compilation reached fitting. Its report requested
**174 DSP blocks on a device with 87**, with 15,018 / 32,070 ALMs (47%).
Fitter errors 184036 and 11802 identify DSP capacity/placement as the blocker.
The old report summary is preserved at `build/quartus-jtag/pre-8x8-fit.summary`.

The MMIO controller now uses an 8x8 physical array (64 PEs) for four output
quadrants, each reducing over all 16 k values. The external 16x16 operation,
numeric format and register ABI are unchanged. Job latency is now 384 clocks;
the host smoke tests and timing documentation have been updated. This reduces
PE count by 75%; actual DSP mapping and timing still require a new fitter run.

- Platform Designer regeneration: pass, updated RTL copied into synthesis files.
- Revised RTL/testbench compilation with Questa: pass, zero errors.
- Updated Python/Tcl host tests: pass, 5 tests.
- Full command-line compilation of revised RTL: blocked by the same internal
  QSYN pipe-opening error; see `build/quartus-jtag/compile-8x8.log`. Recompile
  the regenerated QPF through the user's working GUI flow to check fit/timing.
- Historical 304-clock simulation results below apply to the former design.
- Revised RTL execution and final fitting/timing are not yet established.

## Windows Quartus integration, 2026-10-12

The supplied `tinyvit.qpf` targets `5CSEMA5F31C6` and contains DE1-SoC pin
assignments, but has no HPS system. Its top assignment is `tinyvit` while
`tinyvit.v` defines `bram_2_2`. The new build script derives a JTAG project
inside the writable repository, reusing the device and clock/key/LED pins.
The parent project was not modified.

| Check | Actual result |
|---|---|
| Installed Quartus | 23.1std.1 Build 993, `C:/intelFPGA_lite/23.1std/quartus` |
| Component catalog | Fixed incorrect `qsys` component package and global `add_fileset_file` use; clock/reset/Avalon interfaces now load |
| Platform Designer creation and synthesis HDL generation | Pass; 15 modules, 29 files, zero generation warnings/errors |
| Derived project | Created at `build/quartus-jtag/tinyvit_jtag.qpf`, top `tinyvit_jtag`, 50 MHz constraint, supplied board pins |
| Quartus full compilation | Failed in synthesis: internal QSYN error at `qsyn_cmd.cpp:1539`, `FAILED at opening pipe ul_pipertl...`; repeated with one worker |
| Fit, utilization, timing closure, `.sof` | Not obtained; synthesis failure prevents later stages |
| Questa RTL/testbench compilation | Pass, zero errors; two input-port-kind warnings from reused array |
| Questa RTL execution | Not run; `vsim` cannot initialize its licensing environment |
| Tcl host test against independent Python MMIO model | Pass: 5 tests, covering 12 tiles and detection of wrong results, cycle count, device error and invalid input |
| Programmer enumeration | `quartus_pgm -l`: no JTAG hardware available |
| Physical FPGA matrix multiplication | Not run; no connected cable or successfully built bitstream |
| MNN/HPS execution in this session | Not run; existing MNN prototype unchanged |

Generated logs are retained under `build/quartus-jtag/`, including
`gemm_jtag/gemm_jtag_generation.rpt` and `compile-single.log`. The simulator
transcript is under `build/questa/`. Host tests do not establish RTL correctness
or hardware execution. The Quartus pipe error is an internal tool/runtime
failure; its underlying cause has not been established. Compile the generated
QPF in the local Quartus GUI or rerun `platform/build-jtag.ps1` in a working
tool environment, then check fitting and timing before programming.

See [JTAG bring-up](jtag-bringup.md) for reproducible build and board-test steps.

## Original implementation record

The following checks were run during the initial implementation. No physical
DE1-SoC was connected to this environment.

| Check | Result |
|---|---|
| Verilator 5.006 `--lint-only -Wall` on the new wrapper and reused datapath | Pass, no warnings |
| C++ GEMM library against the Verilated RTL | Pass: 12 matrices × 256 outputs |
| Identity, zero, signed random, INT8 extremes, k=0 far-corner impulse | Pass |
| Consecutive jobs and 304-clock job counter | Pass |
| Scratch/input byte enables, masked START | Pass |
| Rejected input writes and START while busy | Pass |
| ACK, rejected C write, abort, reset during RUN and next-job recovery | Pass |
| Exact FP32 integer conversion through simulated RTL | Pass |
| Rejection of fractional/out-of-range/nonfinite inputs | Pass |
| Software timeout and error paths using a stuck/error test transport | Pass, abort requested |
| Standalone C++ build with warnings treated as errors | Pass |
| Cortex-A9 hard-float standalone executable cross-build | Pass |
| MNN patch application and reverse check against pinned upstream files | Pass |
| Patched MNN Linux host library and minimal test build | Pass |
| Actual MNN session, 16×16 MatMul CPU baseline | Pass: all 256 outputs |
| Hardware-required MNN test with nonexistent UIO device | Correctly fails; CPU fallback is not counted as offload |
| Patched MNN ARM library and test executable cross-build | Pass |
| ARM ELF attributes | ARMv7, VFPv3, NEONv1, hard-float calling convention |
| Quartus component validation, synthesis, fit and TimeQuest | Not run: Quartus unavailable |
| UIO driver build against board kernel | Not run: kernel/image/build tree unknown |
| Device-tree/bridge/boot configuration and MMIO on DE1-SoC | Not run: HPS project and board access unavailable |
| MNN → Linux UIO → physical FPGA end-to-end | Not run |
| Full TinyViT FPGA acceleration, accuracy or speedup | Not implemented or claimed |

The cross-build uses GCC 12 from Debian Bookworm. It does not establish
compatibility with the board image's dynamic libraries. MNN's upstream ARM
build emits architecture-option overlap warnings and an assembly GNU-stack
linker warning. The final ELF attributes were checked as above; execution
still requires board validation with a compatible sysroot/toolchain.

Generated build products are ignored under `build/`. The tested local MNN
source snapshot is ignored under
`mnn/vendor/MNN-2c3f5b21a223734e5db7da4ef7b492bc02cafcf5/`.
The reproducible Git-checkout instructions use `mnn/vendor/MNN/` instead;
both refer to the same pinned revision with the included patch.

Useful products from the validation run:

- `build/arm/gemm_smoke_test`
- `build/mnn-arm/libMNN.so`
- `build/mnn-test-arm/mnn_tile_test`
- `mnn/tests/matmul16.mnn` (small generated model, retained in source)

The legacy board demo is preserved, not validated or repaired in place. The
replacement MMIO controller was tested with its extracted PE/array modules.

## Full-model ARM application — 2026-10-20

Added `mnn/app/tinyvit_run.cpp`, a full-model loader and image/raw-tensor runner.
The supplied `tiny-vit.mnn` successfully executes with `[1,3,224,224]` input
and returns `[1,1000]` FP32 scores.

- Host integration suite passed full inference, RGB normalization/channel
  order, constant-image resize/crop, raw-input replay, output files/metadata,
  and invalid-input/error checks.
- Dynamic and static Cortex-A9 application builds passed with application
  compiler warnings treated as errors.
- The packaged static executable is ARM ELF32, ARMv7, VFPv3, NEONv1, hard-float.
  It has no dynamic section/shared-library dependencies and targets Linux 3.2+.
- The actual ARM executable ran the full model in QEMU with `-cpu cortex-a9`,
  using synthetic zero input and a nonzero RGB image. Both returned 1000 finite
  values. This is instruction-set/runtime validation, not physical board testing.
- Full-model runs reported zero FPGA submissions, as expected for this CPU
  baseline. No classification accuracy or physical-board timing is claimed.

Deployment artifacts: `build/de1-tinyvit/tinyvit_run` and
`build/de1-tinyvit.tar.gz` (executable, original full model, instructions).
These are generated/ignored files; copy the archive separately or rebuild it
using `mnn/app/build_arm.sh`. See `mnn/app/DEPLOY.md` for board commands.

## Quantization accuracy repair — 2026-11-12

- Isolated weight-only versus activation quantization and scanned all 68
  convolutions individually against FP32 final logits.
- Created a mixed-precision graph with 37 native W8A8 convolutions and 31
  original FP32 convolutions; normalization, softmax, GELU and residual
  arithmetic execute in FP32. The selected FC1 remains W8A8.
- On 200 disjoint labeled Imagenette images: FP32 149/200 correct, old broad
  quantization 21/200, corrected model 148/200; corrected prediction agreement
  with FP32 is 193/200. This is not full ImageNet validation.
- On physical HPS Linux, 20 real-image CPU pairs agree 20/20. Mean inference
  is 1163.676 ms FP32 versus 1098.456 ms corrected, including all model
  conversion operators. Model loading and preprocessing are outside this timer.
- Full 128-image calibration plus the checked mixed-precision builder
  reproduces the corrected model SHA256 exactly. Holdout preparation
  reproduces all 200 image selections, labels and PNG hashes.
- Host builder compiles with `-Wall -Wextra -Werror`; invalid/missing/duplicate
  precision policies and existing destinations are rejected. The validation
  gate accepts the corrected model and rejects the old broad model.
- Updated and rebuilt the ARM native INT8 hook for explicit INT8 conversion
  boundaries. Actual FC1 native input/output captures are 100352/401408 bytes;
  the CPU output hash matches the prior runner exactly.
- One FPGA compatibility smoke test completes 28 DMA jobs and one replacement,
  but its logits still fail the CPU check. The separate FPGA per-band arithmetic
  limitation remains; this repair does not claim accurate FPGA-enabled inference.

See the [report and evidence](tinyvit-quantization-accuracy-repair.md).

## Corrected mixed W8A8 with FPGA FC1 — 2026-11-12

Completed 200 measured inferences and 10 warmups on the same 20 saved random
RGB inputs, with the existing 90 MHz image, v5 runner and corrected 37/68-W8A8
model. Five configurations have balanced control/profile runs; explicit model
conversion nodes are measured in addition to the existing driver timers.

- Mean inference: FP32 CPU 1201.953 ms, corrected CPU 1092.412 ms, corrected
  plus FPGA FC1 1123.207 ms. FPGA offload adds 30.796 ms / 2.82% versus the
  same-model CPU; 95% paired CI [19.636,45.594] ms, slower in 19/20 pairs.
- Including the surrounding model conversions, FC1 takes 23.897 ms on CPU
  versus 46.183 ms through FPGA. Execution/polling is 2.922 ms; the selected
  offload operator is 43.589 ms before those model conversion nodes.
- All 200 raw score hashes and 10 warmup hashes were independently checked.
  Control/profile pairs are identical and FP32 scores reproduce the previous
  experiment. Each FPGA inference performs one replacement and 28 DMA jobs.
- FPGA/corrected-CPU top1 agreement is 13/20 on these noise images. This is
  output agreement, not labeled accuracy; the hardware numerical issue remains.

See the [timing report, phase tables and raw evidence](tinyvit-mixed-w8a8-fc1-benchmark.md).
