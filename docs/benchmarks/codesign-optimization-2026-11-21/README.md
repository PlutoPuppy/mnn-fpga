# TinyViT co-design benchmark evidence

The [report](../../tinyvit-codesign-optimization.md) describes the selected
partition, timing boundaries, and hardware resource use.

| Record | Measurement |
|---|---|
| [Legacy NI83 timing](legacy-random20/optimization-summary.json) | Original image, original FP32 CPU, prior FPGA policy, nine additional linears and vector attention |
| [DMA chaining timing](dma-chain-random20/optimization-summary.json) | Four configurations with staged transfers on the same chained DMA bitstream |
| [Output overlap timing](pipeline-random20/optimization-summary.json) | First-forward comparison and CPU/FPGA pipeline timings |
| [Warmed timing](warm-random20/warm-summary.json) | One untimed forward before each paired latency measurement |
| [Labeled accuracy](accuracy200/summary.json) | 200 held-out Imagenette images compared with original FP32 |
| [Quartus timing](adder_prototype.sta.summary) | 90 MHz timing summary across all reported corners |
| [Quartus resources](adder_prototype.fit.summary) | Placement and resource summary for DMC1 |
| [Board matrix tests](physical-matrix-oracle-pipeline.log) | Chained transport with overlapped staging against an integer oracle |

The timing directories contain runner, model, calibration, and bitstream
hashes. Their pipeline CSV files hold per-input profiles. The warmed timing
directory also records each input and output hash. The labeled accuracy
manifest is the [previous independent holdout split](../optimization-2026-11-21/accuracy/fresh-manifest.json).

The complete raw score and operator archives are available in the local
workspace under `build/codesign-optimization/`.

| Archive | SHA256 |
|---|---|
| `codesign-legacy-timing.tar.gz` | `4f0f4af433cbde6d15b0fb23269b72f746414566bedebb3171882f157526574c` |
| `codesign-chain-timing.tar.gz` | `1098da8699f3c0680a6e26a8fe937f5c6a42ea866715b0fa93474aba3e31ab4e` |
| `codesign-pipeline-timing.tar.gz` | `a895e9ec534f428cbab4dd4dce0b062e541bc53b70af5d8df76e1f0d05c01052` |
| `codesign-final-validation.tar.gz` | `30c964051b9032ae1dd697eff2fdd850795cd36c06b365458369ffd1c550cc1b` |
