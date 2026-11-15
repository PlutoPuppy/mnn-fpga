# Linux submission library

## Tiled matrix interfaces

`fpga::exact_float_gemm` tiles dense integer-valued FP32 matrices for the original
INT32-result hardware. `fpga::matmul_soc_gemm` targets the friend's separate
INT8-result ABI, with explicit quantization, transpose support and M/N/K tails.
See [the integration guide](../docs/matmul-soc-integration.md). `make test`
checks both paths without an FPGA. The original single-tile API is described below.

The friend's driver now batches hardware jobs up to 256x64 by 64x256 and
accepts prepared INT8 operands through `matmul_soc_gemm_prepared`. Its optional
DMA transport is separate from the original `uio.cpp` demonstration below.
See [prepared INT8 and DMA](../docs/matmul-soc-batched-int8-dma.md).

`plan_matmul_soc(M,N,K,MatmulSocTilePolicy::Capacity)` searches 16-aligned
tiles satisfying `tile_M*tile_K <= 16384`, `tile_K*tile_N <= 16384`,
`tile_M*tile_N <= 65536`, and the hardware address limits. It estimates the
tradeoff between operand traffic, output traffic, host accumulation and job
count. This is a heuristic; it is not a measured latency prediction or a
guarantee of the optimal tile. Pass the returned plan as the final argument
of `matmul_soc_gemm_prepared`. Omitting it retains the existing Fixed64 policy.
For example, a 128x128 times 128x128 product fits in one job; a 64x256 times
256x64 product can use K=256. Larger logical matrices still span jobs driven
by the host, while the FPGA internally sequences its 16x16 array within each.

`matmul_soc_gemm_int8` accepts native signed INT8 operands with per-channel
output multipliers and output-unit Q16 biases. It returns INT8 and avoids FP32
activation/output arrays. One-K-panel jobs can use the bounded ARMv7 NEON
rescale path; other cases retain safe wider arithmetic. `make test` includes
per-channel/tail/failure checks and the scalar/NEON rounding oracle. This path
still models the hardware's per-band floor/saturation and is approximate
relative to a full INT32 dot product. See the
[native W8A8 experiment](../docs/tinyvit-w8a8-fc1-benchmark.md).

The newer, separate `matmul_soc_native.hpp` interface targets the NI82 image.
`prepare_native_int8` validates its format/capacity registers, packs static
weights into cached DMA buffers and chooses capacity-limited M/N tiles with
the full K reduction (K <= 1024). `native_int8_linear` uses direct NC4 transfers;
`native_int8_pair` adds an activation LUT and FPGA-local feedback for a resident
FC1/FC2 chain. INT32 reduction precedes one per-channel requantization. The
[resident offload report](../docs/tinyvit-native-resident-offload.md) documents
the distinct register contract, DMA limits and physical verification.

`fpga_gemm.hpp` defines the byte-offset ABI and synchronous one-tile API.
`uio.cpp` provides aligned volatile MMIO plus ARM `dmb sy` ordering around each
access. The UIO map must be uncached/device memory, as supplied by the included
driver. This is MMIO, not DMA; no HPS DDR pointers are passed to the FPGA.

`gemm()` probes the ABI, requires idle, writes both complete operands, submits,
polls with a timeout, and reads results. On protocol error or timeout it requests
abort and throws. A CPU bus access that hangs cannot be interrupted by that
polling deadline. The process must have exclusive ownership; UIO opens use an
advisory `flock` and MNN adds an in-process mutex.

Build with `make` at the repository root. Output defaults to
`build/native/gemm_smoke_test`; set `BUILD_DIR` for a separate architecture.
The hardware executable shares its 12-vector suite with the RTL testbench.

The optional kernel module in `driver/` uses the board's matching kernel tree.
See [board bring-up](../docs/board-bringup.md) before accessing hardware.
