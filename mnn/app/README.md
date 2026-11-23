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

## Select one FPGA operation

For M/N/K tiling with the friend's 16x16 array, see
[matmul-soc integration](../../docs/matmul-soc-integration.md).
Use `--list-ops` for runtime names, then `--fpga-op NAME --require-fpga` with
`MNN_MATMUL_SOC_UIO=/dev/uioN`. Rebuild the patched MNN library and runner.
The new `tinyvit_offload_batched` deployment supports prepared INT8 operands,
larger jobs and optional DMA; the preserved `tinyvit_offload` is the small-job
baseline. See [prepared INT8 and DMA](../../docs/matmul-soc-batched-int8-dma.md)
for the board's explicit `/dev/mem`/DMA environment settings and validation.
# Timing the first QK attention operation

`--profile --fpga-op /layers.1/blocks.0/attn/MatMul_output_0` emits one
`PROFILE_JSON` record after inference and output saving. With device environment
variables removed, this measures the CPU QK baseline. With
`MNN_MATMUL_SOC_DEVMEM=1 MNN_MATMUL_SOC_DMA=1`, it measures the DMA offload path.
All stage times are milliseconds, aggregated over the selected operation's 64
head/window products. `completion_wait` includes MMIO polling and scheduling;
the current FPGA ABI has no fabric cycle counter. `cpu_execute_ms` in the raw
JSON is summed **CPU thread-time** across 64 executions on two threads, whose
intervals overlap. Use `selected_qk_ms` for the CPU wall-time comparison.

Run `benchmark_qk_profile.py` on the board as root with `--output-dir NEW_DIR`
and `--bitstream-sha256 SHA256_OF_LOADED_SOF`. It uses 20 seeded random RGB
images, four fresh-process modes per image (CPU/DMA control and CPU/DMA
profile), and balanced rotating order. Controls disable detailed timers.
`summarize_qk_profile.py RESULTS_DIR` verifies counts and output hashes and
writes paired bootstrap statistics and a CSV breakdown. See the diagnosis
report for the measured results and timer boundaries. The additional
`benchmark_qk_cpu_kernels.py` and `summarize_qk_cpu_kernels.py` scripts isolate
CPU packed-kernel intervals, merging overlaps across both CPU threads. See the
[complete diagnosis](../../docs/tinyvit-qk-dma-timing-diagnosis.md).

For repeated all-zero NCHW FP32 input, use
`benchmark_qk_profile.py --zeros --order-design williams` with the same runner,
bitstream hash and a new output directory. This bypasses image normalization;
it is different from a normalized black image. The CPU kernel follow-up
automatically follows the source experiment's input mode. The Williams design
balances mode positions, pair precedence and immediate predecessors across 20
repetitions. See the [zero-input diagnosis](../../docs/tinyvit-qk-zero-input-diagnosis.md).

## Whole-model CPU operator profiling

`--profile-ops FILE.json` records every synchronous runtime operation's wall
time, name, type, input/output shapes, layouts and strides. Model metadata
identifies static convolution weights and parameters. Shape collection is
outside each operator timer; its cost remains included in total inference.
JSON serialization happens after inference. Generated schema and FlatBuffers
headers from the pinned MNN source are required by the application build.

`benchmark_operator_profile.py` runs 20 alternating CPU control/profile pairs
using the preceding random experiment's saved images. It removes FPGA device
environment variables and checks output hashes against the prior CPU baseline.
`summarize_operator_profile.py RESULTS_DIR` writes all operator rankings,
equivalent GEMM dimensions, repeated-shape frequencies and candidate notes;
`plot_operator_profile.py RESULTS_DIR` creates figures. See the
[full CPU operator report](../../docs/tinyvit-cpu-operator-profile.md) for the
measured ranking and the next candidate microbenchmark.

## Experimental fc1 replacement

`--fpga-linear --fpga-op /layers.1/blocks.0/mlp/fc1/Add_output_0__matmul_converted
--require-fpga` replaces only this 1x1 convolution. Enable the existing devmem
and DMA environment variables and choose `MNN_MATMUL_SOC_TILE_POLICY=capacity`
or `fixed64`. Static FP32 weights are transposed and quantized once before
inference. Input quantization, tensor layout copies, DMA, hardware execution,
result reconstruction and bias are included in the selected-operation timer.
`--profile` reports disjoint layout allocation/input/output phases outside
the nested bridge/adapter timers. Other TinyViT operators stay on CPU.

Run `benchmark_fc1_capacity.py --runner tinyvit_fc1_capacity_v2 --inputs 20
--output-dir NEW_DIRECTORY --bitstream-sha256 LOADED_SOF_SHA` as root on the
board. It runs paired CPU/fixed/capacity controls and detailed profiles on
the saved random images, then `summarize_fc1_capacity.py DIRECTORY` creates
the comparison and CSV. The arithmetic diagnostic `fc1_numeric_check MODEL
IMAGE OUTPUT.json` separately checks dense tensor interpretation, ordinary
INT8 operand quantization and the actual band's requantize/saturate behavior.
This remains an opt-in experiment; INT8 accelerator scores require accuracy
evaluation before use as a model replacement.
See the [20-image fc1 report](../../docs/tinyvit-fc1-capacity-benchmark.md) for
timing results and the diagnosed per-band arithmetic bias.

## Native W8A8 fc1 experiment

Apply `mnn/patches/0002-native-int8-fc1.patch` to the pinned MNN source in
addition to the existing FPGA patch, then rebuild MNN and this application.
`--fpga-linear-int8 --fpga-op /layers.1/blocks.0/mlp/fc1/Add_output_0__matmul_converted
--require-fpga` intercepts inside the native INT8 convolution executor after
MNN's runtime casts. It requires the calibrated symmetric INT8 input/output
scales and per-channel weights of this exact layer. Unsupported formats fail.
Use the existing devmem/DMA variables and the capacity tile policy.

`prepare_tinyvit_calibration.py` selects 128 real Imagenette images and 20 disjoint
checks; `quantize_tinyvit_w8a8.py` invokes MNN 3.6.2 calibration with an argv list.
The standalone `create_fc1_quantized_model.cpp` creates a control variant by
copying only calibrated fc1 and its two activation quantization records into
the original model. Build that utility with the generated schema and FlatBuffers
include directories. It refuses to overwrite an existing model.

`benchmark_fc1_w8a8.py` compares FP32 CPU, broad W8A8 CPU/FPGA and fc1-only
W8A8 CPU/FPGA in balanced control/profile modes: 20 images, 200 measured inferences.
`summarize_fc1_w8a8.py` writes statistics and per-image phase/workflow CSV;
`plot_fc1_w8a8.py` exports figures. `MNN_INT8_TRACE=PREFIX` enables diagnostic
native input/output/weight dumps for `check_fc1_w8a8_numeric.py`; disable it
during timing. Profiles also report whole-inference MNN quantize/dequantize
cast scopes, separate from the selected-operation driver phases.

The [W8A8 report](../../docs/tinyvit-w8a8-fc1-benchmark.md) records the generated
models, hashes, complete measurements and failed numerical checks. Broad model
quantization and FPGA banded arithmetic both require accuracy work; successful
INT8 execution does not establish a correct model replacement.

## Corrected quantization model

Use `create_mixed_quantized_model.cpp` with
`tinyvit_w8a8_accuracy_allowlist.txt` to construct the accuracy-preserving
candidate from the original FP32 graph and the calibrated graph. It retains
37 W8A8 convolutions (including the offloaded FC1) and keeps 31 sensitive
convolutions in FP32. Explicit FloatToInt8/Int8ToFloat boundaries keep
normalization, softmax, GELU and residual arithmetic in FP32. This is mixed
precision, not an all-INT8 model.

`quantize_tinyvit_w8a8.py --mixed-builder PATH_TO_COMPILED_BUILDER` performs
calibration followed by this conversion. `scan_quantization_sensitivity.py`
can regenerate a policy for a different checkpoint/calibration set; freeze
the policy before testing a separate holdout. `prepare_quantization_holdout.py`
selects labeled images excluding all calibration and earlier diagnostic
images. `validate_quantized_model.py` separates prediction agreement from
labeled accuracy and exits unsuccessfully when acceptance thresholds fail.

The [accuracy repair report](../../docs/tinyvit-quantization-accuracy-repair.md)
contains the 200-image holdout results, the 20-image HPS check, hashes, commands,
and the remaining FPGA arithmetic limitation. The updated native INT8 patch
accepts both explicitly typed INT8 tensors and framework-cast INT8 storage.
Rebuild the runner after applying it; the deployed compatible runner is
`tinyvit_fc1_w8a8_v5`. The corrected model alone does not repair the FPGA's
per-band rounding and saturation.

To repeat the five-configuration, 20-random-input timing experiment with the
corrected model, run `benchmark_fc1_w8a8.py --runner tinyvit_fc1_w8a8_v5
--full-model tiny-vit-mixed-w8a8-accuracy.mnn --operator-details --inputs 20`
with a new output directory and the loaded bitstream SHA256 as before.
`--operator-details` enables runtime operator profiling only in the separate
profile modes. It captures the 37 explicit quantization/dequantization pairs
that are absent from the legacy CastWrapExecution counters, including the
two conversion nodes surrounding FC1. The summary adds
`conversion-breakdown.csv` alongside the driver phase CSV. See the
[corrected-model FC1 timing report](../../docs/tinyvit-mixed-w8a8-fc1-benchmark.md).

## Multiple native INT8 operations and resident MLPs

The separate NI82 image and `tinyvit_native_v11` runner support all 30 eligible
pointwise INT8 operations in the corrected model. Use `--fpga-native-int8
--fpga-native-pairs --require-fpga` with the existing devmem/DMA environment
variables. `--native-allowlist FILE` restricts selection, and `--native-plan`
checks the graph and activation tables without accessing hardware. These flags
cannot be combined with the older single-operation FPGA selectors.

Four verified FC1/activation/FC2 chains keep their hidden INT8 activation in
FPGA RAM. A model-derived activation table and FPGA-local DMA eliminate its
HPS round trip. Static weights are packed and cached during setup; runtime
operand and output transfers use DMA with direct NC4/NEON layout copies.
The NI82 datapath accumulates the full reduction in INT32 before requantizing.
It requires its matching bitstream; the older band-rounded image is rejected.

See the [native resident offload report](../../docs/tinyvit-native-resident-offload.md)
for measured performance, arithmetic checks, hardware qualification, memory
reservation, artifact hashes and reproduction commands. `benchmark_native_int8.py`
records separate unprofiled and profiled runs; `summarize_native_int8.py`,
`select_native_policy.py` and `plot_native_int8.py` analyze the saved evidence.
The v11 runner refreshes session input between invocations; the graph can reuse
its input storage. `repeat_input_copy_ms` reports these copies separately from
inference. `check_native_session_reuse.py` compares runs=1/2/3 CPU/FPGA sessions
with saved single-inference scores. The 20-image performance report records
the v10 single-inference artifact separately from this session-reuse fix.

The [expanded static-linear experiment](../../docs/tinyvit-expanded-static-offload.md)
adds all 40 transformer projections, or all 51 static pointwise/linear
operations, using separate experimental models. NI83 expands the A bank to
32 KiB and supports K up to 2048, including stage-3 FC2. Both expanded models
fail the CPU accuracy gate and must not replace the validated mixed model.
The historical expanded-static experiment uses `tinyvit_native_v12` with
NI83. It also supports the preserved validated
mixed model with 30 offloaded linears and four resident pairs; the older v11
binary requires NI82.
`--native-verify` mirrors unfused CPU operators on FPGA with identical INT8
inputs, preserves CPU graph outputs, and rejects errors above one output code.
`benchmark_expanded_numeric.py` collects these diagnostics; ordinary
`benchmark_native_int8.py --abi NI83 --configurations FILE` runs collect
performance separately. Configurations specify the model, offload/pair flags,
and expected operation/pair counts.

## CPU kernels and transport optimization

The [latency optimization report](../../docs/tinyvit-latency-optimization.md)
records the accuracy gates, paired timings, rejected experiments and complete
pipeline accounting. The original FP32 model and validated mixed model are
preserved; the expanded static-quantization models remain experimental.
The [DMA follow-up](../../docs/tinyvit-dma-optimization.md) measures v27 at
905.762 ms versus 945.195 ms for v25 and 1170.188 ms for FP32 CPU in the
matched 20-image timing run. The v27 package is
`build/tinyvit-dma-optimized-90mhz-v27.tar.gz`.

Additional flags are opt-in:

- `--fused-attention` replaces the ten supported TinyViT QK/softmax/AV regions
  with tiled FP32 CPU attention. Cortex-A9 microkernels process four queries
  and eight columns, reuse MNN's worker pool, and balance odd-head windows
  by splitting independent query rows. Scale and positional bias stay FP32.
- `--gelu-neon` evaluates the original MNN GELU polynomial with vector
  reciprocals and two Newton refinements. The ARM32 implementation replaces
  scalar divisions; it does not change model weights or calibration.
- `--verify-attention --verify-gelu-neon` mirrors these kernels while keeping
  the original CPU graph outputs. Diagnostic latency is not a benchmark.
- `--park-fpga-workers` temporarily suspends the session's idle CPU worker
  pool inside synchronous native FPGA calls. A scope guard restores it before
  control returns to MNN. `PARK_JSON` records actual calls.

`ATTENTION_JSON` and `GELU_JSON` record per-region times, invocation counts,
numerical checks and CPU-pool ownership/capacity. Attention replacement time
is outside the ordinary operator callback intervals; GELU replacement time
is inside its UnaryOp interval. `summarize_tinyvit_optimization.py` accounts
for this difference and checks that all pipeline groups sum to inference.

`MNN_MATMUL_SOC_FAST_POLL=1` reduces timer sampling and redundant native DMA
CSR writes while retaining the one-second deadlines and device barriers.
`MNN_MATMUL_SOC_STREAM_STAGE=1` selects an experimental cached staging copy;
it was slower and remains disabled by default. DMA payload limits remain
2,048 bytes for HPS transfers and 256 bytes for local feedback.

The v27 package enables three transport options explicitly:

- `MNN_MATMUL_SOC_REUSE_A=1` walks tiles in alternating directions to reuse
  FPGA SRAM contents. It chooses the traversal with the lower estimated
  transfer and packing cost and reuses resident FC2 inputs at compatible
  tile boundaries.
- `MNN_MATMUL_SOC_LIVE_ROWS=1` transfers live rows only. The compute shape
  and K padding still meet the accelerator's tile requirements.
- `MNN_MATMUL_SOC_DMA_POLL_BATCH=1` uses one deadline per transfer in the
  existing fast polling mode. It keeps the barriers around DMA start and stop.

The direct A/C experiments use `MNN_MATMUL_SOC_DIRECT_A` and
`MNN_MATMUL_SOC_DIRECT_C`. Both are disabled in the package because they
raised upload or download time compared with DDR DMA. For a labeled run,
pass the chosen options with repeated `--transport` arguments to
`benchmark_optimized_accuracy.py`. `--require-identical-fpga` enforces
exact score parity when reference results are supplied.

The independent kernel tests can be built with
`-DTINYVIT_BUILD_KERNEL_TESTS=ON` when configuring this application.
`tinyvit_attention_test` covers FP32/INT8 input, window/head layouts, query
slices and tails; `tinyvit_gelu_test` covers both kernels and in-place/tail
cases. Run them on the target ISA, or use CTest in a native host build.

The model builder accepts an optional final weight-only allowlist:

```sh
create_mixed_quantized_model FP32.mnn CALIBRATED.mnn STATIC_ALLOWLIST.txt \
  NEW_MODEL.mnn WEIGHT_ONLY_ALLOWLIST.txt
```

Weight-only additions preserve original FP32 biases and activation graph
boundaries. `--memory-low` requests MNN dynamic activation quantization for
those CPU operators; the MNN library must be built with `MNN_LOW_MEMORY=ON`.
These experiments passed several accuracy screens but were slower on this
Cortex-A9. They are not the selected deployment policy.

`benchmark_native_int8.py` accepts optional per-configuration `flags`,
`environment` and `runner` fields, and records all executable hashes.
`benchmark_optimized_accuracy.py` checks labeled accuracy and FP32 agreement
separately. Its optional `--reference-results` validates hashes and
preprocessing before reusing completed FP32 references for a kernel update.

### Calibrated FP32-boundary projections

Use `--native-projections RANGES.tsv` with `--fpga-native-int8 --require-fpga`
to select additional FP32 1×1 convolutions for the NI83 engine. The runner
packs their static weights into INT8 tiles during setup. At inference it
quantizes activations as they become available, runs INT8 GEMM with INT32
accumulation, and returns FP32 values to MNN. `--verify-native-projections`
compares FPGA and CPU projection results on the same input tensor.

`calibrate_native_projection.py` generates the TSV from an operator allowlist
and an independent set of 224×224 RGB calibration images. The generated JSON
records model and input hashes. Each record specifies the exact operator name,
output-channel count, and minimum/maximum for each output channel.
`benchmark_native_int8.py` accepts the TSV filename in a configuration's
`projection_ranges` field. `benchmark_optimized_accuracy.py` accepts the path
through `--projection-ranges`.

### Chained DMA

`MNN_MATMUL_SOC_DMA_CHAIN=1` requires the DMC1 capability in the NI83 Quartus
image. It issues up to 64 KiB in one command per transfer. The FPGA sequences
the existing 2 KiB HPS bursts or 256-byte local bursts. The benchmark counts
HPS-issued commands in the descriptor fields. Boot with the existing reserved
DDR staging window before enabling DMA. Use `--transport dma-chain` to enable
this transport in the labeled accuracy script.

`MNN_MATMUL_SOC_PIPELINE_C=1` overlaps NC4 output unpacking on an HPS worker
with the next FPGA tile's upload and compute. The driver waits before reusing
the DMA destination and drains the worker before returning a completed layer.
The output-copy timer records the worker's service time. It can overlap other
phases, so only the wall-time download phase belongs in the additive forward
breakdown. Pass `--transport pipeline-c` for the labeled accuracy script.
