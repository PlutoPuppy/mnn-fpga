# First MNN offload experiment

To load and run the **full `tiny-vit.mnn` on ARM Cortex-A9**, use
[app/README.md](app/README.md). The small `mnn_tile_test` below remains the
separate FPGA communication experiment.

This integration patches CPU `CPUMatMul::onExecute` in Alibaba MNN at
`2c3f5b21a223734e5db7da4ef7b492bc02cafcf5`. It is deliberately opt-in:
without `MNN_FPGA_UIO`, the existing CPU path runs.

Supported FPGA operations have exactly two dense row-major FP32 16×16 inputs,
one 16×16 output, no transpose, and no bias. Every input value must be finite,
integral, and between -128 and 127. The integer result is exactly representable
as FP32 for this tile size. This is **not a quantizer** and does not accelerate
ordinary TinyViT floating-point activations. Other shapes/types/values use CPU.
Device errors also use CPU after logging the error. Calls are serialized.

## Prepare pinned source

Run from the repository root. Use a separate checkout; the script verifies the
commit and checks the patch before applying it. It never resets your checkout.

```sh
git clone https://github.com/alibaba/MNN.git mnn/vendor/MNN
git -C mnn/vendor/MNN checkout 2c3f5b21a223734e5db7da4ef7b492bc02cafcf5
python3 mnn/prepare.py mnn/vendor/MNN
```

Apply once. An already-applied or incompatible patch fails the pre-check.
The integration references this repository's `sw/` files via `MNN_FPGA_ROOT`;
keep the repository present during builds.

## Linux host baseline

Use a Linux environment (the Docker image in `sim/` is suitable):

```sh
cmake -S mnn/vendor/MNN -B build/mnn-host \
  -DMNN_FPGA_ROOT="$PWD" -DMNN_BUILD_TOOLS=OFF \
  -DMNN_BUILD_CONVERTER=OFF -DMNN_SEP_BUILD=OFF \
  -DMNN_ARM82=OFF -DMNN_USE_THREAD_POOL=OFF -DCMAKE_CXX_STANDARD=17
cmake --build build/mnn-host -j 4
cmake -S mnn/tests -B build/mnn-test \
  -DMNN_SOURCE="$PWD/mnn/vendor/MNN" -DMNN_BUILD="$PWD/build/mnn-host"
cmake --build build/mnn-test -j 2
env -u MNN_FPGA_UIO build/mnn-test/mnn_tile_test --cpu build/matmul16.mnn
```

The test constructs and saves a two-input MatMul `.mnn` model, runs an actual
MNN session, and compares every result against a CPU reference. A generated
copy is included as `tests/matmul16.mnn`. Both operands are runtime inputs to
avoid constant folding or conversion into a constant-weight convolution.

## Build for the DE1-SoC HPS

Use the board's compatible ARM Linux compiler/sysroot. The provided toolchain
file assumes an `arm-linux-gnueabihf-*` compiler on PATH and a hard-float image.

```sh
cmake -S mnn/vendor/MNN -B build/mnn-arm \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/platform/arm-linux-gnueabihf.cmake" \
  -DMNN_FPGA_ROOT="$PWD" -DMNN_BUILD_HARD=ON \
  -DMNN_BUILD_TOOLS=OFF -DMNN_BUILD_CONVERTER=OFF -DMNN_SEP_BUILD=OFF \
  -DMNN_ARM82=OFF -DMNN_USE_THREAD_POOL=OFF -DCMAKE_CXX_STANDARD=17
cmake --build build/mnn-arm -j 4
cmake -S mnn/tests -B build/mnn-test-arm \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/platform/arm-linux-gnueabihf.cmake" \
  -DMNN_SOURCE="$PWD/mnn/vendor/MNN" -DMNN_BUILD="$PWD/build/mnn-arm"
cmake --build build/mnn-test-arm -j 2
```

Deploy `build/mnn-arm/libMNN.so` and `build/mnn-test-arm/mnn_tile_test` together
to the board. After the standalone hardware smoke test passes, run in that
directory (substitute the correct UIO device):

```sh
LD_LIBRARY_PATH=. ./mnn_tile_test --cpu matmul16.mnn
LD_LIBRARY_PATH=. MNN_FPGA_UIO=/dev/uio0 ./mnn_tile_test --require-fpga matmul16.mnn
```

Unset `MNN_FPGA_UIO` for the CPU baseline. The hardware test requires exactly
one completed FPGA submission and correct output. If the device is missing,
or MNN routes execution elsewhere, it fails even if CPU fallback computes
the right answer. `mnn_fpga_completed_jobs()` provides that explicit evidence.
Keep the environment's UIO path fixed for the process lifetime.

## Beyond this test

Add M/N/K tiling and transpose/batch handling with tests before widening the
eligibility guard. For real FP32 TinyViT inputs, choose and validate a numeric
conversion/quantization policy first. Inspect the pinned runtime's actual
execution graph: not every serialized MatMul reaches this method unchanged.
Eventually a dedicated Backend/Execution implementation can replace this
small CPU integration patch.

See the [pinned CPU MatMul source](https://github.com/alibaba/MNN/blob/2c3f5b21a223734e5db7da4ef7b492bc02cafcf5/source/backend/cpu/CPUMatMul.cpp)
and [MNN backend documentation](https://github.com/alibaba/MNN/wiki/backend).
