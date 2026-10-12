# Validation record

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
| Full TinyViT inference, accuracy or speedup | Not implemented or claimed |

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
