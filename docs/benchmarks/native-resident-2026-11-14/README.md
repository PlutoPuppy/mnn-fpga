# NI82 resident TinyViT evidence, 2026-11-14

See [the report](../../tinyvit-native-resident-offload.md) for interpretation,
commands, timing boundaries, arithmetic differences and qualification limits.

| Files | Contents |
|---|---|
| `random-all/` | Primary 20-image, four-configuration summary, metadata, per-image results, layer/policy/timing CSV and figures |
| `random-all-evidence.tar.gz` | All 160 raw run logs, operator profiles and FP32 score outputs, plus physical matrix/MLP smoke log |
| `random-selected/` | Additional 20-image selected-policy experiment, 120 inferences; historical cyclic ordering limitation is documented |
| `natural/` | Twenty labeled natural-image checks, four configurations, 160 inferences; identical predictions, 16/20 correct in each |
| `additional-evidence.tar.gz` | Raw selected/natural experiment directories, allowlist and exact recorded board harness |
| `session-reuse-evidence.tar.gz` | Negative v10 repeated-input check and six passing v11 CPU/FPGA session-reuse cases, twelve correctness inferences |
| `session-reuse-checks.json`, `session-reuse.log` | Reference hashes, job counts and passing v11 reuse checks |
| `hardware-qualified.json` | SOF, v10 benchmark runner, v11 delivered runner, smoke executable and generated hardware source hashes |
| `adder_prototype.*` | Fitter/timing summaries and compressed complete reports; uncompressed report hashes are also retained |
| `native-output-repro-v4.log`, `physical-smoke.log` | Actual RTL CXXRTL and physical signed-matrix/resident-pair oracle results |
| `benchmark_native_int8_recorded.py` | Exact script used for the 440 timing inferences, SHA256 `938e171c98a872590350a8135ad43de0841a9dac9f6091267527dde24983df53` |
| `source-sha256.json` | Current integration and analysis source hashes, including the delivered session-reuse fix |
| `source-sha256-lf.json` | The same source hashes after canonicalizing CRLF to LF, for comparison with Git checkouts |
| `SHA256SUMS` | Hashes of all evidence files except itself |

The primary control timings disable detailed profiling. Separate profile runs
retain operator intervals and adapter phases. Raw `output_0.f32` files contain
1000 little-endian float logits. The summarizer checks every raw output hash and
control/profile agreement before calculating statistics. Model/input hashes,
normalization, execution order and commands are saved in metadata/results.
The input image files remain on the board at those recorded paths; these
archives contain outputs and profiles, not copies of the input image dataset.

The delivered runner is v11; the 20-image timing runs intentionally retain their
v10 provenance. The v11 single-run and repeated-run outputs match the saved v10
fresh-process references. Do not mix these timings with the preliminary CPU
baseline from a different boot or the older band-rounded bitstream.

The reusable harness in `mnn/app/` additionally fixes odd-configuration order
balance and accepts a host-supplied experiment date because the board clock is
incorrect. Its current hash therefore differs from the recorded harness.
Raw source hashes describe the Windows working files; Git may normalize source
line endings. For checkout verification normalize CRLF to LF and compare with
`source-sha256-lf.json`. Benchmark evidence itself is marked `-text` in
`.gitattributes` so its measured bytes and manifests survive checkout unchanged.

Extract archives into a new directory, then summarize the relevant result folder:

```sh
python mnn/app/summarize_native_int8.py EXTRACTED/benchmark-native-v4-v10-random-20
python mnn/app/plot_native_int8.py EXTRACTED/benchmark-native-v4-v10-random-20
```
