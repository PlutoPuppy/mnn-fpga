# MNN on DE1-SoC FPGA

For the friend's **16x16 matmul-soc array / 256x256 INT8 output canvas**, see
[the new integration guide](docs/matmul-soc-integration.md). It adds M/N/K tiling,
an explicit approximate FP32 path, and named-operation selection in TinyViT.
The original INT32 register interface described below remains a separate target.

The [fc1 capacity-aware experiment](docs/tinyvit-fc1-capacity-benchmark.md)
adds an opt-in tile planner and measures one MLP projection on 20 random images
at 90 MHz, including CPU comparison, phase timing and numerical limitations.

The [W8A8 fc1 experiment](docs/tinyvit-w8a8-fc1-benchmark.md) adds calibrated
MNN models and a native INT8 executor hook, with 200 measured runs on the same
20 random inputs. It separates CPU quantization gains, FPGA overhead and
numerical limitations.

HPS Linux submits a signed INT8 **16×16 matrix multiplication** to the FPGA
and reads 256 INT32 results. Target: DE1-SoC **5CSEMA5F31C6N**, Quartus Prime
Lite **23.1std**. The Quartus device picker typically omits ordering suffix N.

```text
MNN CPU MatMul (optional, restricted prototype)
                 ↓
C++ GEMM library → Linux UIO → lightweight HPS-to-FPGA bridge
                                      ↓
                          Avalon-MM registers and A/B buffers
                                      ↓
                            feed/capture controller
                                      ↓
                      8×8 systolic array, four output quadrants
                                      ↓
                                  C buffer
```

| Directory | Contents |
|---|---|
| `rtl/` | Reused PE/array modules and new Avalon-MM wrapper/controller |
| `rtl/legacy/` | Original button/LED demo, preserved unchanged; excluded from builds |
| `sim/` | Verilator testbench driving the C++ GEMM library, Docker toolchain |
| `platform/` | Platform Designer component, example device tree, ARM toolchain |
| `sw/` | UIO access, GEMM library, smoke test, polled UIO kernel driver |
| `mnn/` | Pinned MNN patch, guarded FPGA path, minimal model generator/test |
| `docs/` | Register map, board bring-up, validation status, TinyViT graph |

## Run local validation

With Verilator, a C++ compiler, and Make installed:

```sh
make lint sim
make
```

Or use Docker from this directory:

```sh
docker build -t mnn-fpga-dev -f sim/Dockerfile .
docker run --rm -v "$PWD:/work" mnn-fpga-dev make lint sim
docker run --rm -v "$PWD:/work" mnn-fpga-dev make BUILD_DIR=build/linux
docker run --rm -v "$PWD:/work" mnn-fpga-dev make BUILD_DIR=build/arm CXX=arm-linux-gnueabihf-g++ 'CXXFLAGS=-O2 -std=c++17 -Wall -Wextra -Werror -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard'
```

The 16×16 external operation uses 64 physical PEs, processing four 8×8 output
quadrants sequentially over the full K=16, for 384 fabric clocks per job.
The former 256-PE design exceeded the target's DSP capacity.

`make sim` compares all 256 outputs for 12 deterministic matrix pairs, then
checks byte enables, busy-write rejection, reset/abort recovery, float-input
eligibility, and software timeout/error handling. It simulates the actual RTL;
it does not establish Quartus timing closure or board operation.

## Bring up the board

For the supplied `tinyvit.qpf` project on Windows, start with
[JTAG fabric bring-up](docs/jtag-bringup.md). Run `./platform/build-jtag.ps1`
to generate a derived Quartus project using its device and board pins, then
use `platform/jtag-smoke.tcl` to submit and check 12 matrix multiplications.
This first test requires no HPS Linux setup. See `docs/validation.md` for
build and hardware limitations observed in this environment.

Follow [board-bringup.md](docs/board-bringup.md), using
[register-map.md](docs/register-map.md) as the hardware/software contract.
After programming the FPGA, enabling the bridge, and binding the UIO driver:

```sh
./gemm_smoke_test /dev/uio0
```

Find the correct UIO device by name; it is not necessarily `uio0`. Use a
compiler/sysroot compatible with your Linux image.

## MNN integration

For the **full TinyViT model on HPS Linux**, use
[the ARM application](mnn/app/README.md). It loads your actual `tiny-vit.mnn`
at 224×224, supports images/preprocessed tensors, and exports its 1000 scores.
The deployment package is `build/de1-tinyvit.tar.gz`; see
[board run commands](mnn/app/DEPLOY.md). CPU inference on Cortex-A9 is the default;
the new integration guide describes selection of one FPGA operation.

See [mnn/README.md](mnn/README.md). This is an opt-in patch to CPU MatMul
execution, not a complete new backend. The original 16×16 demo accepts exact
signed INT8-valued FP32 inputs. Named-operation selection adds larger tiled
integer matrices on that hardware, or approximate FP32 tiling on matmul-soc.

The full [TinyViT graph](docs/tiny-vit.mnn.svg) needs further work: ordinary
float activations still require accuracy validation of the new quantization
policy. M/N/K tiling and real DMA are implemented; interrupts and performance
tuning remain future work. The [20-input timing diagnosis](docs/tinyvit-qk-dma-timing-diagnosis.md)
compares CPU matrix kernels, FPGA execution, conversion and transfer costs.
The [whole-model CPU profile](docs/tinyvit-cpu-operator-profile.md) ranks all
runtime operators and identifies larger static-weight projection candidates.
The [quantization accuracy repair](docs/tinyvit-quantization-accuracy-repair.md)
adds a validated mixed-precision model with 37 W8A8 convolutions and preserves
sensitive layers in FP32. Its model accuracy checks are separate from the
legacy FPGA arithmetic mismatch.
The [corrected-model FC1 experiment](docs/tinyvit-mixed-w8a8-fc1-benchmark.md)
reports the new 20-random-input CPU/FPGA comparison and full timing breakdown.
The [native resident offload report](docs/tinyvit-native-resident-offload.md)
documents the newer NI82 datapath, full INT32 reductions, 30 supported INT8
linear operations and four MLP pairs whose hidden activations stay on FPGA.
It includes measured inference and cold-start costs on the qualified 90 MHz
image, plus a measured layer-selection policy.

The [expanded static-linear experiment](docs/tinyvit-expanded-static-offload.md)
adds experimental policies for all 40 transformer linears or all 51 static
linears, NI83 support for K up to 2048, and same-input CPU/FPGA verification.
Both expanded quantization policies fail the CPU accuracy gate; keep the
validated mixed model for accuracy-sensitive use.

## Current limits

The [latency optimization report](docs/tinyvit-latency-optimization.md) measures
CPU attention/GELU kernels and DMA-control changes on the same 90 MHz NI83
image, with separate 20-image timing and 200-image labeled accuracy checks.
It reports the measured speedup and remaining gap to the 1.5x inference target.
The [DMA follow-up](docs/tinyvit-dma-optimization.md) reduces inference to
**905.762 ms versus 1170.188 ms FP32 CPU (1.292x)** in the matched run.
It applies to the same 90 MHz NI83 FPGA image. The local v27 package is
`build/tinyvit-dma-optimized-90mhz-v27.tar.gz`.

The [90 MHz co-design report](docs/tinyvit-codesign-optimization.md) covers
39 FPGA linears, chained DMA and overlapped HPS output staging. Its warmed
forward averages 757.013 ms versus 1207.126 ms on original FP32 CPU (1.595x).
The separate first-forward comparison reaches 1.494x. The corresponding local
deployment bundle is `build/tinyvit-codesign-90mhz.tar.gz`.

- See [validation.md](docs/validation.md) for checks actually performed.
- The original `platform/` target contains integration files. The friend's
  complete HPS project is now supported by a separate build helper and DE1-SoC
  board profile; see the new integration guide. A matching Linux image and
  kernel build tree are still external to this repository.
- Quartus fitting, bridge/boot setup, kernel compatibility, and physical board
  execution must be checked in your board environment.
- The original `systolic_mmio` target prioritizes correctness: each INT8
  element uses one MMIO word. The separate matmul-soc integration now supports
  [prepared INT8, larger hardware jobs and real DMA](docs/matmul-soc-batched-int8-dma.md).
  Lane-parallel buffer reads may synthesize into registers/muxes. Check resource
  use and timing before optimizing memory banking or DMA.
