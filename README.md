# MNN on DE1-SoC FPGA

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

See [mnn/README.md](mnn/README.md). This is an opt-in patch to CPU MatMul
execution, not a complete new backend. It accepts dense 16×16 FP32 tensors
whose values are exact signed INT8 integers. Other inputs keep CPU execution.

The full [TinyViT graph](docs/tiny-vit.mnn.svg) needs further work: ordinary
float activations require validated quantization, and larger/batched operations
need tiling. General GEMM, DMA, interrupts, and TinyViT acceleration come later.

## Current limits

- See [validation.md](docs/validation.md) for checks actually performed.
- No HPS base project, DDR/pin configuration, Linux image, or kernel build tree
  has been supplied. `platform/` contains integration files, not a ready-to-program
  board project or bitstream.
- Quartus fitting, bridge/boot setup, kernel compatibility, and physical board
  execution must be checked in your board environment.
- This version prioritizes correctness: each INT8 element uses one MMIO word.
  Lane-parallel buffer reads may synthesize into registers/muxes. Check resource
  use and timing before optimizing memory banking or DMA.
