# Linux submission library

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
