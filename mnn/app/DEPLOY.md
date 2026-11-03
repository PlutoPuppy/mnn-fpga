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

Inference runs on the HPS CPU by default and reports zero FPGA submissions.
No FPGA image is needed for this CPU baseline. The original `MNN_FPGA_UIO`
interface remains an exact integer-valued MatMul experiment.

Physical-board CPU inference now passes on Ubuntu 16.04 / Linux 4.5.0. The
rebuilt static runner, deployed as `tinyvit_offload`, produces the same 1000
scores as the earlier runner for a zero input. See the validation record for
measured results. This is not image-classification accuracy validation.

## One selected operation on matmul-soc

Rebuild with the updated MNN patch. The friend's hardware has an INT8 output
format; this path quantizes FP32 tiles and is approximate. Read
[the integration guide](../../docs/matmul-soc-integration.md) and load the
matching FPGA image before accessing it. The connected board lacks UIO, so
the explicit prototype `/dev/mem` path requires root:

On this board, normal boot loads `soc_system.rbf` and replaces the JTAG image.
Follow the integration guide's **tested boot sequence** before these commands.
The sequence retains the accelerator without changing saved boot settings.

```sh
sudo ./matmul_soc_smoke --devmem
sudo env MNN_MATMUL_SOC_DEVMEM=1 ./tinyvit_offload \
  --model tiny-vit.mnn --height 224 --width 224 --zeros --threads 2 \
  --fpga-op /layers.3/blocks.1/attn/MatMul_output_0 --require-fpga \
  --output-dir outputs/fpga-one-op
```

That runtime operation multiplies ten 49x32 matrices by 32x49 matrices.
The tiler submits 320 independent 16x16x16 jobs, including zero-padded edges.
Other operations stay on the CPU. For another export, run `--list-ops` first
and choose its actual operation name. `--require-fpga` fails on CPU-only
execution; model success alone does not prove an offload.
