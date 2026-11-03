# Validation record

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
