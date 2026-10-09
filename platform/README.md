# Board integration files

- `systolic_mmio_hw.tcl`: Avalon-MM component for Quartus 23.1.
- `systolic-uio.dtsi.example`: illustrative Linux node; adjust address and
  parent cell encoding to match your base design.
- `arm-linux-gnueabihf.cmake`: Cortex-A9 hard-float compiler settings; use a
  compiler/sysroot compatible with your board image.

Follow [board bring-up](../docs/board-bringup.md). A full HPS project and
bitstream are not included. Quartus is unavailable in the development
environment, so validate the Tcl in your 23.1std installation.
