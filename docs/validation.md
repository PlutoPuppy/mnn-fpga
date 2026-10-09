# Validation record

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
