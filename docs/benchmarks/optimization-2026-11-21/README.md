# TinyViT optimization evidence — 2026-11-21

Read the [full report](../../tinyvit-latency-optimization.md) for interpretation,
timing boundaries and deployment. Selected inference is 937.571 ms versus
1173.468 ms FP32 CPU: 1.2516x, not the requested 1.5x. Labeled accuracy is
153/200 for both on the fresh ten-class Imagenette subset.

## Contents

- `final-random20/`: final v25 controls, profiles, 10,000-resample paired
  intervals, disjoint pipeline CSV, and PNG/SVG plots. Configuration key
  `no_park` is the selected deployment; `optimized` includes optional worker
  parking and is slightly slower for loaded-model inference. The automated
  target record selects the best measured FPGA configuration explicitly.
- `initial-random20/`: six-mode experiment separating mixed CPU, original
  resident FPGA, fast polling, CPU kernels and optimized CPU-only execution.
  `native-layers.csv` supplies the unchanged per-layer shapes/tile plans and
  same-model CPU comparisons. Paired FC1 rows include the whole resident
  chain; paired FC2 callbacks consume saved results.
- `initial-accuracy200/`, `final-accuracy200/`: raw result records, model/input
  metadata and accuracy summaries. The final pass reuses the initial FP32
  references with hash and preprocessing checks; it does not rerun them.
- `board-evidence-v22.tar.gz`, `board-evidence-v25.tar.gz`: complete captured
  board result directories, including logs, raw FP32 output files and operator
  profiles, tuning/rejected experiments and exact board collection scripts.
  The source RGB images themselves are identified by manifest and SHA-256;
  image files and model binaries are not duplicated inside these archives.
- `accuracy/`: disjoint fresh and diagnostic image manifests, host numerical
  screens, dynamic/spatial quantization policies and their rejected results.
- `hardware/`: programming/host/oracle logs and packaged-wrapper smoke result.
- `host-generation/`: exact model/data generation and board-launch helpers,
  configuration JSON and generation logs. Prototype scripts are evidence;
  use maintained application tools for new runs. Board paths refer to the
  experiment's `/home/ubuntu/tinyvit` deployment, not to the PC.
- `model-provenance.json`, `artifact-sha256.json`, `source-sha256.json`,
  `deployment-package.json`: input models, executable/hardware hashes,
  final public source hashes and ready-package identity. Raw and LF-normalized
  source hashes distinguish Windows checkout newlines from source changes.
- `audit_evidence.py`, `offline-audit.json`: independent standard-library
  audit of all 1,040 unique full-model score files, hashes, finite values,
  profile/control parity, FP32 baseline parity and recomputed labeled results.
- `SHA256SUMS`: hashes of all evidence files, excluding itself.

Formal suites execute 240 + 420 + 160 + 220 = 1,040 new inferences. Additional
tuning and mathematical oracles are separate. Every timing configuration
uses two CPU threads, 20 identical saved random inputs and balanced order,
with separate control/profile processes, one run and no warmup. Startup/cache
variation is retained. No outliers were removed. The board RTC is incorrect;
the host supplies the date. Unknown Linux CPU-frequency state is recorded
as unknown rather than inferred from a requested governor.

## Offline reproduction

Extract each archive to its own new directory, preserving member paths.
The commands below assume you are at the repository root and have extracted
them to `build/review-v22` and `build/review-v25` respectively:

```sh
python docs/benchmarks/optimization-2026-11-21/audit_evidence.py \
  build/review-v22 build/review-v25 build/review-audit.json
python mnn/app/summarize_native_int8.py build/review-v25/optimization-v25-random20
python mnn/app/summarize_tinyvit_optimization.py build/review-v25/optimization-v25-random20
python mnn/app/plot_native_int8.py build/review-v25/optimization-v25-random20
```

The audit and summaries use Python's standard library; plotting also requires
Matplotlib. Do not extract over existing evidence. The timing collector is
`mnn/app/benchmark_native_int8.py`; use the saved configuration files, matching
NI83 image, model/runner hashes and board memory reservation for new runs.
`mnn/app/benchmark_optimized_accuracy.py` distinguishes labeled accuracy from
FP32 output agreement and supports validated reference reuse.

The ready deployment package and its model are kept under the ignored local
`build/` directory, identified by the package manifest. Dependency licenses
are included. No commit or push was performed.
