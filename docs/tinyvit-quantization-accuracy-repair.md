# TinyViT quantization accuracy repair

Verified on 2026-11-12. The severe accuracy collapse is repaired with an
accuracy-selected **mixed-precision** model: 37 of 68 convolutions use native
W8A8, while 31 sensitive convolutions and the non-convolution arithmetic stay
FP32. The FC1 selected for FPGA offload remains W8A8. This is not a claim that
the original all-68-convolution quantization now works.

| Separate 200-image holdout | Correct labels | Agreement with FP32 | Mean logit cosine |
|---|---:|---:|---:|
| Original FP32 CPU | 149/200 = 74.5% | Reference | 1.000000 |
| Previous broad W8A8 CPU | 21/200 = 10.5% | 25/200 = 12.5% | 0.287578 |
| Corrected mixed-precision CPU | **148/200 = 74.0%** | **193/200 = 96.5%** | **0.997583** |

The corrected model loses one correct classification, or 0.5 percentage points,
relative to FP32 on this sample. Prediction agreement and labeled accuracy are
different metrics. Neither this sample nor the earlier 3/20 agreement result
is full ImageNet accuracy validation.

On the physical HPS, the corrected CPU model also matches FP32's top prediction
on all 20 original real-image checks. Mean inference falls from **1163.676 ms
to 1098.456 ms**, a **5.60% latency reduction / 1.0594x speed ratio**. These
timings have the FPGA disabled and include every conversion operation inside
the model.

## Why broad quantization failed

The failure exists in CPU-only execution, before any FPGA transfer. It is
primarily activation sensitivity, rather than corruption of the encoded INT8
weights. Quantizing all weights while retaining FP32 activations gives 19/20
prediction agreement on the diagnostic images and 195/200 on the holdout.
Adding the blanket activation quantization causes the collapse.

Layer tracing on `n01440764-00.png` shows the problem concretely:

| Tensor | Original FP32 range | Previous broad model range |
|---|---:|---:|
| First GELU output | -0.170 to 26.108 | -0.175 to 11.101 |
| Stage 0, block 0, first convolution | -106.250 to 123.163 | -61.481 to 61.481 |
| Stage 3, block 0, final residual sum | **-1024.073 to 9.040** | **-79.061 to 13.696** |
| Final logits | -3.147 to 8.919 | -6.837 to 4.742 |

The original calibration attaches INT8 quantization information to 351 tensors,
including residual and normalization paths. Its KL-selected ranges clip
important values. The late stage also has extreme channel imbalance: one
tensor reaches about 1,260 in the calibration set. A single INT8 scale must
choose between clipping those outliers and coarsely rounding the smaller
values. Replacing every range with maximum-absolute calibration does not solve
this tradeoff: that all-68-convolution experiment gives 0/20 diagnostic matches.

Single-layer sensitivity confirms that the final stage is especially fragile.
Quantizing only stage 3, block 0's MLP `fc2`, with the rest of the graph FP32,
gives just **1/5** matching predictions and mean final-logit cosine **0.412089**.
A local activation cosine can look high because a large outlier dominates its
norm; the final prediction is a more useful sensitivity check here.

The [layer trace](benchmarks/quantization-repair-2026-11-12/trace-broad.json) and
[all 68 single-layer results](benchmarks/quantization-repair-2026-11-12/sensitivity.json)
preserve the measurements. Intermediate experiments included:

| Diagnostic configuration | FP32 prediction agreement, 20 real images |
|---|---:|
| Original broad KL model | 3/20 |
| INT8 weight storage with FP32 activations | 19/20 |
| All 68 W8A8 convolutions with explicit FP32 boundaries | 4/20 |
| All 68 W8A8 convolutions with maximum-absolute activation ranges | 0/20 |
| Broad KL with the whole final stage restored to FP32 | 17/20 |
| Selected 62 W8A8 convolutions | 18/20 |
| Selected 44 W8A8 convolutions | 19/20 |
| **Selected 37 W8A8 convolutions** | **20/20** |

This is why the repair protects sensitive layers rather than claiming that
changing a global range parameter makes blanket INT8 conversion safe.

## What changed

[`create_mixed_quantized_model.cpp`](../mnn/app/create_mixed_quantized_model.cpp)
starts from the original FP32 graph and imports calibrated weights only for
the selected convolutions. Each selected convolution gets explicit
FloatToInt8 and Int8ToFloat boundaries on new tensor IDs. Original tensor IDs
and output connections remain connected to FP32 arithmetic. The utility
rejects incompatible graphs, invalid calibration, missing or duplicate policy
entries, and existing output paths.

The checked [precision policy](../mnn/app/tinyvit_w8a8_accuracy_allowlist.txt)
selects operators whose single-layer mean final-logit cosine is at least
0.9998 on five diagnostic images. Threshold selection used the 20-image
diagnostic set; the final 37-layer policy was fixed before its separate
200-image validation. The policy is specific to this checkpoint and
calibration; a changed model needs a new sensitivity/validation run.

The 37 selected convolutions comprise nine FC1s, four FC2s, eight QKV
projections, nine attention output projections, and seven local convolutions.
The excluded late FC1 and FC2 layers, the initial spatial stages, and other
sensitive convolutions retain their original FP32 parameters.

An actual HPS runtime profile verifies the precision, rather than inferring it
from a model filename:

| Executed operation | Count | Actual input/output element bits |
|---|---:|---:|
| Selected convolutions | 37 | 8 / 8 |
| Remaining convolutions | 31 | 32 / 32 |
| FloatToInt8 | 37 | 32 / 8 |
| Int8ToFloat | 37 | 8 / 32 |
| LayerNorm | 21 | 32 / 32 |
| Softmax | 10 | 32 / 32 |
| Unary operations, including GELU | 23 | 32 / 32 |
| Runtime binary arithmetic | 40 | 32 / 32 |

The single diagnostic profile measures 20.820 ms in explicit quantization
operators and 14.779 ms in explicit dequantization operators. These are
included in inference and are not 20-run means. The existing framework
CastWrapExecution counters do not count these explicit conversion nodes;
zero events in those counters must not be interpreted as zero conversions.

[`quantize_tinyvit_w8a8.py`](../mnn/app/quantize_tinyvit_w8a8.py) now supports
`--mixed-builder` to produce this graph after calibration. The
[sensitivity scanner](../mnn/app/scan_quantization_sensitivity.py),
[holdout selector](../mnn/app/prepare_quantization_holdout.py), and
[validation gate](../mnn/app/validate_quantized_model.py) make the process
repeatable. The gate checks agreement, logit cosine and, with labels supplied,
accuracy loss independently.

## Validation inputs and results

All accuracy inputs are natural images, with no all-zero or uniform-random
noise cases. Calibration uses 128 Imagenette training images. The diagnostic
set uses two validation images per class. The additional holdout uses 20
different validation images per class, excluding every calibration and
diagnostic source filename.

Preparation converts to RGB and resizes directly to 224x224 with PIL bilinear
resampling. Inference uses NCHW `(pixel / 255 - mean) / std`, with ImageNet
mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
This matches the experiment's preprocessing; it is not a canonical full
ImageNet evaluation with a separate resize/center-crop protocol.

Source images come from the
[Imagenette archive](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz).
Labels use the standard
[PyTorch ImageNet output ordering](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt).
The [calibration manifest](benchmarks/quantization-repair-2026-11-12/calibration-manifest.json)
and [holdout manifest](benchmarks/quantization-repair-2026-11-12/holdout-manifest.json)
record source filenames, selections, labels and PNG hashes. The holdout
selector was rerun independently and reproduced all 200 selections, labels
and image hashes exactly.

The corrected model passes the validation gate: at least 95% agreement,
mean cosine at least 0.995, and labeled accuracy loss at most one percentage
point. The old broad model fails the gate. The
[corrected gate result](benchmarks/quantization-repair-2026-11-12/host-corrected-gate-200.json)
contains every per-image result. The
[comparison file](benchmarks/quantization-repair-2026-11-12/host-comparison-200.json)
also preserves the weight-only and more aggressive mixed-precision controls.

These are small-sample results on ten classes. They establish recovery from
the severe collapse; they do not prove equivalence on all ImageNet classes or
every deployment input.

## HPS timing and execution check

[`benchmark_quantization_accuracy_board.py`](../mnn/app/benchmark_quantization_accuracy_board.py)
ran 40 measured inferences: FP32 and corrected CPU on the same 20 real images,
with alternating pair order, two threads, fresh processes, and two separate
warmup inferences. FPGA variables were removed from each child process.

| HPS CPU inference | FP32 | Corrected mixed precision |
|---|---:|---:|
| Mean | 1163.676 ms | **1098.456 ms** |
| Median | 1161.640 ms | 1082.585 ms |
| Sample standard deviation | 9.896 ms | 43.403 ms |
| Correct labels on these diagnostic images | 16/20 | 16/20 |
| Prediction agreement with FP32 | Reference | 20/20 |

The corrected model is faster in 18/20 pairs. The mean paired change is
**-65.220 ms**, with a 95% paired-bootstrap interval of
**[-81.846, -44.277] ms**, using 20,000 resamples and seed 20261112. All timing
outliers are retained. The corrected model's slower outliers increase its
standard deviation; no run was discarded.

The inference timer encloses `runSession`, including model conversions,
convolutions, attention and other CPU operators. It excludes image decoding,
normalization, model loading, session creation and saving the final output.
Fresh-process wall times average 2261.139 ms for FP32 and 1911.189 ms for the
corrected model, but those include loading and file-system effects and are
not used for the inference speedup claim. The board does not expose the
sampled cpufreq file used by this script, so those metadata fields are null.

The earlier 850 ms broad-quantization timing belongs to the inaccurate model.
It must not be reused as the performance of this repaired model.

## FPGA compatibility and remaining numerical issue

The selected FC1 remains `M=784, K=128, N=512`, with its existing calibrated
INT8 weights and input/output scales. The native hook was updated to accept
both forms of real INT8 tensor used by MNN: logically FP32 tensors whose
backend applies INT8 storage, and tensors explicitly typed as signed INT8.
It still requires one-byte storage, NC4HW4 pack4 layout, compatible scales,
and the selected FC1 shape. The distributable change is in
[`0002-native-int8-fc1.patch`](../mnn/patches/0002-native-int8-fc1.patch).

The rebuilt `tinyvit_fc1_w8a8_v5` captures the selected native input/output as
100352 / 401408 bytes. Its CPU logits match the v4 benchmark output byte for
byte on the diagnostic check. The HPS profile confirms all 37 selected
convolutions actually use INT8 operands and results.

One smoke inference with the existing 90 MHz FPGA image completes exactly
one FC1 replacement and 28 DMA jobs, using capacity-selected 128x128x128
hardware jobs. This proves that the corrected graph connects to the driver.
**It does not pass the numerical check:** on that real image the CPU predicts
class 0 and FPGA-enabled inference predicts class 63; logit cosine is
0.332681 and relative L2 error is 1.452983.

The separate hardware per-band rounding/saturation limitation documented in
the [earlier numerical diagnosis](tinyvit-w8a8-fc1-benchmark.md) remains.
This repair changes model precision selection and tensor compatibility, not
the RTL arithmetic. The corrected model is validated for CPU execution;
accurate FPGA-enabled full inference remains unfinished. No new 20-run
FPGA speedup is claimed from this compatibility test.

## Artifacts and reproduction

The generated model is on this PC at
`build/mnn-w8a8-tools/tiny-vit-mixed-w8a8-accuracy.mnn` and deployed on the board
at `/home/ubuntu/tinyvit/tiny-vit-mixed-w8a8-accuracy.mnn`. The original model
and failed broad model are preserved under their existing names.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| Original FP32 model | 26602144 | `f7aee628e2cef2d726979cc259529b3af0778b429c52ba8b76f3c609a2996de4` |
| Failed broad model | 10686388 | `d4d6bb729e25cd56d9ea5546a2b3a737af0678e497ffd8ae57c12b4377ef0277` |
| Corrected mixed-precision model | 18748848 | `f6a577a805887e486a4bc52179724e9c7c3340ec694c51532073b55beabab1cf` |
| v4 runner used for 20 CPU pairs | 2347920 | `233af95ef5de385b0f694273a69e78b9340ad36b3031924b452b2cf0b561e151` |
| v5 runner with explicit-INT8 hook support | 2348080 | `79f1bf12509594bc65741d78d6b23dfbe4aa9cc987fd45d1615df58ad9419e0b` |

Use MNN 3.6.2 for host calibration/validation. The ARM runtime uses the
project's pinned MNN source and integration patches. The standalone builder
needs the generated schema and FlatBuffers headers, but does not link MNN.
For example, from PowerShell at the repository root:

```powershell
$gxx = 'C:/intelFPGA_lite/23.1std/questa_fse/gcc-7.4.0-mingw64vc16/bin/g++.exe'
& $gxx -std=c++17 -O2 -Wall -Wextra -Werror `
  -isystem build/MNN-build-source/schema/current `
  -isystem build/MNN-build-source/3rd_party/flatbuffers/include `
  mnn/app/create_mixed_quantized_model.cpp -o build/create_mixed_quantized_model.exe

# Use a NEW output filename; the utilities preserve earlier models.
python mnn/app/quantize_tinyvit_w8a8.py `
  --model build/mnn-w8a8-tools/tiny-vit-fp32.mnn `
  --output build/tiny-vit-mixed-w8a8-new.mnn `
  --calibration-dir build/mnn-w8a8-tools/calibration-128 --images 128 `
  --feature-method KL --python-path build/mnn-w8a8-python `
  --mixed-builder build/create_mixed_quantized_model.exe

python mnn/app/prepare_quantization_holdout.py `
  --archive build/mnn-w8a8-tools/imagenette2-160.tgz `
  --exclude-manifest build/mnn-w8a8-tools/calibration-manifest.json `
  --output-dir build/new-accuracy-holdout

python mnn/app/validate_quantized_model.py `
  --reference build/mnn-w8a8-tools/tiny-vit-fp32.mnn `
  --candidate build/tiny-vit-mixed-w8a8-new.mnn `
  --images build/new-accuracy-holdout `
  --manifest build/new-accuracy-holdout/manifest.json `
  --output build/new-accuracy-validation.json --python-path build/mnn-w8a8-python
```

The complete calibration-plus-conversion command was rerun: both its raw
calibration intermediate and final model reproduce the recorded SHA256 hashes
exactly. The host builder compiles with warnings treated as errors; the ARM
runner rebuild and actual native/DMA checks pass. Structural rejection checks
and the accuracy gate's successful and failed cases are retained as evidence.

For a changed checkpoint, regenerate the precision policy with
`scan_quantization_sensitivity.py --reference FP32 --calibrated CALIBRATED
--builder BUILDER --images DIAGNOSTIC_DIR --output-dir NEW_SCAN_DIR
--python-path MNN_INSTALL_DIR`, then validate on a different holdout.

The [evidence directory](benchmarks/quantization-repair-2026-11-12/) contains
per-image results, manifests, raw board/native archives, precision inspection,
model provenance and a [SHA256 manifest](benchmarks/quantization-repair-2026-11-12/SHA256SUMS).
