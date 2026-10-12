# DE1-SoC board bring-up

For the supplied FPGA-only `tinyvit.qpf`, first use
[JTAG matrix bring-up](jtag-bringup.md). The steps below apply when a working
HPS/DDR base system and Linux image are available.

Target: Cyclone V `5CSEMA5F31C6N`, Quartus Prime Lite 23.1std. Start from your
working DE1-SoC HPS project or the matching Terasic reference project, keeping
its DDR, pin, clock, and boot configuration. Those settings are not contained
in the supplied systolic module.

## 1. Platform Designer

1. Confirm the board device, typically listed as `5CSEMA5F31C6` in Quartus.
2. Add this repository's `platform/` to the IP search path, refresh the catalog,
   and add **16x16 INT8 systolic GEMM** from `systolic_mmio_hw.tcl`.
3. Enable the HPS **lightweight HPS-to-FPGA master** and connect it to `avs`.
   No FPGA DDR master is needed for this version.
4. Connect `clock` to a stable peripheral clock, initially 50 MHz if available.
   The wrapper/array share one clock. Let Platform Designer handle any crossing
   between that clock and the HPS master clock.
5. Connect active-high `reset` through the base design's reset controller,
   synchronized to the peripheral clock, with appropriate HPS bridge reset
   dependencies. A pushbutton alone is not the subsystem reset network.
6. Assign a free **4 KiB-aligned, 4 KiB span** in the lightweight master address
   space. Record the offset. Offset zero may already be occupied.
7. Validate/generate the system, include its `.qip` in the Quartus project,
   and connect the generated subsystem using the base project's top-level
   wiring. The custom component includes its three RTL files automatically.
8. Exclude `rtl/legacy/systolic_16by16.sv`; it defines duplicate PE/array
   modules and a different top level.
9. Run synthesis, fitting and TimeQuest. Inspect DSP/ALM/register utilization,
   inferred memories, unconstrained paths, and achieved frequency. The current
   64-PE array tiles the 16x16 output into four quadrants to reduce DSP use;
   fitting and timing must still be checked.

No ready-to-program `.qsf`, `.qsys`, pin assignments or bitstream are included.
Save your selected base project into `platform/` and preserve its actual
address map and generated configuration.

## 2. Boot and bridge setup

Program the FPGA through the flow supported by your Linux/boot image. Enable
the lightweight bridge, release its reset, and run the peripheral clock before
CPU access. A wrong or unclocked target can hang an MMIO transaction; software's
GEMM polling deadline cannot recover a CPU transaction that never returns.

The Cyclone V lightweight aperture starts at `0xff200000`; add the component's
offset from the generated map. Confirm this against your configuration. The
device-tree example assumes offset zero. Use the bridge management provided
by your image; do not copy reset-register sequences from another SoC.

## 3. Linux UIO

`sw/driver/systolic_uio.c` binds to `mnn,systolic-mmio-1.0`, providing one
uncached MMIO map with no interrupt. It neither loads the FPGA nor enables
bridges.

1. Run `uname -a`; obtain the exact configured kernel build tree and enable
   `CONFIG_UIO`.
2. Integrate `platform/systolic-uio.dtsi.example` into the board device tree.
   Adjust the physical address and parent address/size cell encoding. Its
   root-level example assumes one cell each. Deploy the rebuilt DTB through
   your image's normal procedure. Avoid overlapping an existing device range.
3. Build against that kernel using a matching compiler:

   ```sh
   make -C sw/driver KDIR=/path/to/board-kernel-build \
     ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf-
   ```

4. On the board, load the UIO core if modular, then the driver:

   ```sh
   modprobe uio
   insmod systolic_uio.ko
   grep . /sys/class/uio/uio*/name
   ```

5. Find the device named `systolic-mmio`. Inspect `maps/map0/addr`, `size` and
   `offset`: expect the assigned physical base, `0x1000`, and zero. Grant your
   test process access to that `/dev/uioN` device.

The driver has not been built against your unknown kernel. Vendor kernel API
differences may require adjustments.

## 4. Run the smoke test

Build on the board or cross-compile with its sysroot. The Docker ARM compiler
is useful for build checks; its glibc/libstdc++ may be newer than a vendor SD
image. Deploy an executable compatible with your image.

```sh
./gemm_smoke_test /dev/uioN
```

The test checks ID/version/format, scratch/operand readback, and 12 matrix
pairs. All outputs must match. Each job reports 384 accelerator clocks. The
legacy LED demo is not involved.

If ID is wrong, inspect bridge, address, FPGA image, clock and reset. For wrong
results, use SignalTap on state, cycle, `a_left`, `b_top`, and store index.
If timing fails, lower the peripheral clock and/or pipeline/bank buffer paths.

## 5. MNN and later work

Follow `mnn/README.md`: CPU baseline first, then the same model with
`--require-fpga`. That mode rejects silent CPU fallback.

After that passes: add larger M/N/K tiling, validate TinyViT quantization,
measure packing/upload/compute/download separately, then introduce burst DDR
transfers with driver-managed DMA buffers. Use the Linux DMA API, not a
`malloc` pointer; do not assume the FPGA shares HPS cache coherence.

References:

- [Cyclone V HPS reference](https://www.intel.com/programmable/technical-pdfs/683126.pdf)
- [Avalon specification](https://cdrdv2-public.intel.com/667068/mnl_avalon_spec-683091-667068.pdf)
- [Linux UIO HOWTO](https://docs.kernel.org/driver-api/uio-howto.html)
- [Linux DMA guide](https://docs.kernel.org/core-api/dma-api-howto.html)
