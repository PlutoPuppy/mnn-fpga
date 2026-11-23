# TinyViT DMA experiment evidence

The test used the existing 90 MHz NI83 image and validated mixed model on
the DE1-SoC. The experiment and timing scopes are described in the
[DMA optimization report](../../tinyvit-dma-optimization.md).

- `dma-pilot-metadata.tar.gz` holds the board results, metadata, and logs for
  the two eight-image pilots. `pilot-v26-summary.json` and
  `pilot-v27-summary.json` contain the paired timing and transfer statistics.
- `dma-final-timing.tar.gz` contains the final 20-image paired timing run,
  including the raw scores and operator profiles for each control and profile
  run. `final-random20/` contains derived timing, phase and pipeline tables.
- `dma-final-accuracy.tar.gz` contains the labeled validation run with raw
  candidate scores and the summary. It reuses the previously recorded FP32
  references after checking their hashes and preprocessing flags.
- `recovery-boot.log`, `recovery-program.log`, `run-resume.sh` and
  `benchmark_dma_accuracy_resume.py` record the HPS restart and resume.
  The resume checks the settings and hashes of the first 120 completed
  images and processes the rest with the same runner and transport flags.
- `source-and-runners.json` identifies the source, runner and bitstream used.
  `host-native.log` and the two oracle logs record mathematical checks.
- `deployment-package.json` records the selected transport configuration and
  the local package hash. `package-smoke.log` records a physical wrapper run.

To reproduce the offline pipeline tables, extract the timing archive into
its own directory. From the repository root, run the maintained analysis
tools against the extracted `dma-final-random20` directory.

```sh
python mnn/app/summarize_native_int8.py build/dma-optimization/raw-timing/dma-final-random20
python mnn/app/summarize_tinyvit_optimization.py build/dma-optimization/raw-timing/dma-final-random20
python mnn/app/plot_native_int8.py build/dma-optimization/raw-timing/dma-final-random20
```

The `audit_dma.py` script expects the extracted timing folder in `raw-timing`
and the extracted accuracy folder in `raw-accuracy`, both under its first
argument. Pass this evidence directory as the second argument and the
earlier `docs/benchmarks/optimization-2026-11-21` evidence directory as the
third. The audit checks the raw output hashes, finite scores, matched saved
inputs and exact agreement with the earlier qualified FPGA run.
