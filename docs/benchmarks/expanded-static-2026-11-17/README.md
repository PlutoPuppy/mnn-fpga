# NI83 expanded-static TinyViT evidence, November 17–18, 2026

See [the report](../../tinyvit-expanded-static-offload.md) for interpretation,
timing boundaries, quantization ablations and qualification limits.

Both expanded quantization policies fail the CPU accuracy gate. A local
one-code INT8 arithmetic tolerance passes; full-model prediction agreement
with the corresponding CPU graph is only 14/20 and 17/20 on the natural images.
These artifacts are experimental and do not replace the validated mixed model.

| Path | Contents |
|---|---|
| `random/` | Twenty random inputs, eight configurations, 320 inferences; control/profile records, means/medians/spread, paired confidence intervals, 51-layer shape/tiling/timing CSV and figures |
| `natural/` | Twenty natural inputs, seven configurations, 280 inferences; full-model CPU/FPGA outputs, labeled accuracy, agreement metrics and timing summaries |
| `numeric/` | Forty diagnostic model runs, comparing 206,180,640 INT8 values on identical CPU inputs; per-layer errors and verification that diagnostic CPU outputs match ordinary controls |
| `accuracy/` | Four 200-image host accuracy/ablation reports, their policies, holdout image hashes and identity checks against prior validation |
| `hardware/` | Ninety-MHz timing/fit summaries and compressed reports, compiled QSF/SDC, programming log, physical oracle results and SOF/application hashes |
| `board-evidence.tar.gz` | All 2235 original board evidence files: 640 score outputs, run logs, detailed operator profiles, metadata, results and exact benchmark scripts; SHA256 `b1a66bc1713e63de9b175941c086823b689c5ea9831f990d039d0e64adb2b61e` |
| `model-provenance.json` | Original, validated and expanded model sizes/hashes; logical precision-policy hashes and calibrated source hash |
| `configurations.json` | Models, CPU/offload flags and required operation/pair counts for all eight configurations |
| `benchmark_*`, `expanded-*` | Exact recorded board harnesses, configuration, launch script, progress logs and successful suite exit status |
| `host-driver-check.log`, `rtl-output-check.log` | Fifty-one host driver cases and 162 output RTL cases |
| `source-sha256*.json` | Integration and analysis source hashes, including a canonical-LF version for Git checkouts |
| `SHA256SUMS` | Every evidence file except this manifest itself |

The archive SHA256 was verified after a framed serial download. Each frame
also had a separate SHA256 with retries. Extraction rejected links, special
files and paths outside the new destination. The summaries verify all saved
score hashes, CPU/FPGA completion counts, control/profile output equality,
fused/unfused equality where tested, and additive operator-time accounting.

Each output contains 1000 little-endian FP32 scores. Diagnostic output files
are deliberately CPU graph outputs; the diagnostic `verification` arrays
record FPGA comparisons against those operators' identical saved inputs.
The ordinary timing experiments contain actual full offload outputs.

The 20 natural board inputs and the 200 host holdout inputs are separate sets.
Some short filenames coincide while their image bytes differ: use hashes,
not filenames, to match them. Image datasets and model/runner/SOF binaries are
not in this archive. They remain in their recorded local/board deployment
paths. Random inputs are the earlier seeded 224x224 uniform-RGB images; natural
inputs are reused diagnostic examples. Neither set establishes full ImageNet
accuracy. The 200-image holdout is disjoint from calibration but has been used
in previous diagnostic validation.

The board wall clock is incorrect (2016). Metadata `date` is the host-supplied
experiment start date, November 17; the suite completed November 18 local time.
CPU frequency sysfs values were unavailable (`cpu_state_before/after` are
empty); U-Boot reported nominal MPU 925 MHz and DDR 400 MHz. All timing modes
used the same boot, runner, two CPU threads and programmed 90-MHz NI83 image.
No samples or profile outliers were removed. File caches were not flushed.

Extract the archive into a new directory and reproduce the summaries:

```sh
python mnn/app/summarize_native_int8.py EXTRACTED/benchmark-native-ni83-v12-random-20
python mnn/app/summarize_native_int8.py EXTRACTED/benchmark-native-ni83-v12-natural-20
python mnn/app/summarize_expanded_numeric.py EXTRACTED/benchmark-native-ni83-v12-numeric-20 \
  --natural-results EXTRACTED/benchmark-native-ni83-v12-natural-20
python mnn/app/plot_native_int8.py EXTRACTED/benchmark-native-ni83-v12-random-20
```

`analyze_expanded_recorded.py` records the additional comparisons, extra-11
operator table and complete additive pipeline grouping. It expects extraction
under `build/expanded-static-20261117/board-evidence/`, with the summaries above
already generated, when invoked from the repository root.

Evidence is marked `-text` in `.gitattributes`, preserving measured bytes
across checkouts. Source-file hashes can differ after Git changes line endings;
normalize CRLF to LF before comparing with `source-sha256-lf.json`.
