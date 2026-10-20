# Run on DE1-SoC HPS Linux (ARM Cortex-A9)

This directory contains a statically linked **32-bit ARMv7 hard-float Linux**
application, `tinyvit_run`, and the full `tiny-vit.mnn` model. It targets
Cortex-A9 with NEON/VFPv3. It needs neither `libMNN.so` nor a board-installed
libstdc++ runtime. It is a Linux application, not an FPGA programming file.

On the board:

```sh
chmod +x tinyvit_run
./tinyvit_run --model tiny-vit.mnn --height 224 --width 224 --zeros \
  --threads 2 --output-dir outputs
```

This runs the entire model using a synthetic zero tensor, verifies successful
execution and finite results, and saves 1000 output scores. It is a runtime
smoke test, not an image classification accuracy check.

For an image, use preprocessing matching your export. **If your model uses
the usual ImageNet mean/std and resize-short-side-256/center-crop-224**, run:

```sh
./tinyvit_run --model tiny-vit.mnn --height 224 --width 224 \
  --image photo.jpg --resize-short 256 \
  --mean 0.485,0.456,0.406 --std 0.229,0.224,0.225 \
  --threads 2 --output-dir outputs
```

The confirmed size is 224×224. The normalization/crop values above are an
example; they have not been confirmed against the export pipeline. Image
decoding supports JPEG/PNG and other formats accepted by stb_image, converts
to RGB, and uses bilinear resizing without antialiasing. EXIF rotation and
color profiles are not applied. The image path may differ slightly from a
PIL/torchvision export pipeline; use raw input for exact preprocessing parity.

To run a tensor prepared by your original pipeline:

```sh
./tinyvit_run --model tiny-vit.mnn --height 224 --width 224 \
  --input-f32 input.f32 --output-dir outputs
```

`input.f32` must contain exactly 150,528 little-endian float32 values (602,112
bytes), in NCHW order `[1,3,224,224]`, with preprocessing already applied.
For example, save a contiguous NumPy input with
`tensor.astype('<f4').tofile('input.f32')`.

Outputs are stored as `outputs/output_0.f32` with names, shape, and type in
`outputs/outputs.json`. This model outputs `[1,1000]`. Printed top indices are
class indices and raw scores; labels and probability calibration are not
included. Inference timing excludes image/model loading and output transfer.

Inference runs on the HPS CPU. The existing optional `MNN_FPGA_UIO` path only
accepts a restricted 16×16 integer-valued MatMul, so ordinary full TinyViT
inference is expected to report zero FPGA submissions. Do not claim FPGA
acceleration from successful CPU inference. No FPGA image is needed for this
CPU baseline.

The binary is tested with Cortex-A9 user-mode emulation, not a physical board.
It targets Linux 3.2 or later and requires sufficient RAM. The exact board
kernel is still unknown. If execution fails, collect `uname -a`, available RAM
(`free -m`), and the full error message. QEMU timing is not board timing.
