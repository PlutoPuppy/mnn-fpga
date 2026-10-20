# Full TinyViT ARM application

`tinyvit_run.cpp` loads the supplied model using `Interpreter::createFromFile`,
resizes its dynamic input to `[1,3,H,W]`, supplies an image/tensor, executes the
full graph, and exports every FP32 output. It does not generate a replacement
test model. Your confirmed input size is **224×224**.

See [DEPLOY.md](DEPLOY.md) for board commands and preprocessing details.

## Build and package for Cortex-A9

Prepare the pinned, patched MNN source as described in [../README.md](../README.md).
From the repository root, on Linux with the ARM hard-float compiler:

```sh
sh mnn/app/build_arm.sh mnn/vendor/MNN /path/to/tiny-vit.mnn
```

Or use Docker on this development computer:

```sh
docker build -t mnn-fpga-dev -f sim/Dockerfile .
docker run --rm -v "$PWD:/work" \
  -v /absolute/path/to/tiny-vit.mnn:/models/tiny-vit.mnn:ro \
  mnn-fpga-dev sh mnn/app/build_arm.sh /work/mnn/vendor/MNN /models/tiny-vit.mnn
```

The existing local pinned snapshot is at
`mnn/vendor/MNN-2c3f5b21a223734e5db7da4ef7b492bc02cafcf5`; use that directory
instead of `mnn/vendor/MNN` when reusing the already-prepared source.

Outputs:

- `build/tinyvit-arm-static/tinyvit_run`: self-contained ARM Linux executable.
- `build/de1-tinyvit/`: executable, full model, and board instructions.
- `build/de1-tinyvit.tar.gz`: deployment archive to copy via SCP or SD card.

The build uses `-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard`. MNN enables its
ARMv7 kernels, with ARMv8.2 support disabled. The application and MNN/C/C++
runtimes are statically linked; board userspace library versions do not need
to match the build container. Board Linux/kernel compatibility still matters.

The deployment files are ignored by Git because they are generated. Copy the
archive separately, or rebuild it on another computer using tracked sources.

## Local host tests

Using the already-built host MNN library in Linux:

```sh
cmake -S mnn/app -B build/tinyvit-host \
  -DMNN_SOURCE="$PWD/mnn/vendor/MNN" -DMNN_BUILD="$PWD/build/mnn-host"
cmake --build build/tinyvit-host -j 2
python3 mnn/app/test_runner.py --exe build/tinyvit-host/tinyvit_run \
  --model /path/to/tiny-vit.mnn
```

The dynamic host build copies `libMNN.so` beside the executable and sets its
runtime search path to that directory. Tests check full-model output shape,
finite values, RGB channel ordering/normalization, resize/crop on a constant
image, raw-tensor replay, and invalid-input/error handling. They do not prove
classification accuracy against the original export framework.

## Cortex-A9 instruction-set execution test

Inside the Docker image, run the ARM executable with QEMU:

```sh
qemu-arm -cpu cortex-a9 build/de1-tinyvit/tinyvit_run \
  --model build/de1-tinyvit/tiny-vit.mnn --height 224 --width 224 \
  --zeros --output-dir build/tinyvit-qemu
```

This exercises the actual 32-bit ARM binary and full model. It is not a
simulation of the DE1-SoC's FPGA, memory controller, boot process, or performance.
