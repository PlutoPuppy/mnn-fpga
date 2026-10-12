# Board integration files

- `build-jtag.ps1`: derive and compile a JTAG project using the supplied
  `tinyvit.qpf` device and clock/key/LED pins.
- `create_jtag_system.tcl`: JTAG Avalon master + GEMM system source.
- `tinyvit_jtag.sv`, `tinyvit_jtag.sdc`: board top and 50 MHz constraints.
- `jtag-smoke.tcl`: upload and verify 12 matrices through System Console.

- `systolic_mmio_hw.tcl`: Avalon-MM component for Quartus 23.1.
- `systolic-uio.dtsi.example`: illustrative Linux node; adjust address and
  parent cell encoding to match your base design.
- `arm-linux-gnueabihf.cmake`: Cortex-A9 hard-float compiler settings; use a
  compiler/sysroot compatible with your board image.

Start with [JTAG bring-up](../docs/jtag-bringup.md), then follow
[HPS board bring-up](../docs/board-bringup.md). A full HPS project and
verified bitstream are not included. See [validation](../docs/validation.md)
for successful Platform Designer generation and current tool blockers.
