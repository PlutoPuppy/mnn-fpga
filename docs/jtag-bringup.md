# First matrix-multiplication offload on the supplied TinyViT project

The initial fabric test uses this path:

```text
Windows System Console -> USB-Blaster/JTAG -> Avalon master
                                              |
                                     systolic_mmio (4 KiB)
                                              |
                               16x16 INT8 GEMM -> INT32 C
```

This tests the same accelerator and register ABI used by `sw/` and the MNN
prototype. It does not require an HPS Linux image or DDR configuration. JTAG
is a bring-up transport; the current MNN UIO integration requires the HPS
lightweight bridge and the board's actual Linux setup.

## Build using the supplied project

From `mnn-fpga` in PowerShell:

```powershell
./platform/build-jtag.ps1
```

The script reads `../../../tinyvit.qpf` relative to `platform/`, checks its
Cyclone V device, and copies its clock, key and LED pin assignments into a
derived project at `build/quartus-jtag/tinyvit_jtag.qpf`. It does not modify
the original project. The original project is outside this session's writable
workspace; its configured top `tinyvit` also does not match the `bram_2_2`
module in its `tinyvit.v` source.

The derived project includes the existing 16x16 accelerator, the generated
JTAG master/interconnect/reset controller, a 50 MHz clock constraint and
3.3-V LVTTL assignments. Legacy demo modules and unrelated peripheral pins
are excluded. `KEY[0]` resets the system; `LEDR[0]` indicates the released
button, not matrix-test success. The System Console test reports success.

Override installation or base-project paths if needed:

```powershell
./platform/build-jtag.ps1 -QuartusRoot C:/intelFPGA_lite/23.1std/quartus -BaseProject E:/path/to/tinyvit.qpf
./platform/build-jtag.ps1 -GenerateOnly
```

`create_jtag_system.tcl` is the reproducible Platform Designer system source.
The GEMM byte base is zero in the JTAG master's private address space. This
is distinct from the HPS physical address `0xff200000` used in the Linux
example. No HPS address is accessed by this test.

## Program and execute

After full compilation and timing validation succeed, open the derived QPF
in Quartus, connect and power the DE1-SoC, and program the generated
`build/quartus-jtag/output_files/tinyvit_jtag.sof` using Programmer. Confirm
the Cyclone V device and cable in the chain. Release `KEY[0]`.

Open Tools -> System Debugging Tools -> System Console and run:

```tcl
source E:/Quartus/Quartus-lite-23.1std.1.993-windows/components/tinyvit/git-repo/mnn-fpga/platform/jtag-smoke.tcl
```

Alternatively, from PowerShell in `mnn-fpga`, use an absolute script path:

```powershell
$testScript = (Resolve-Path ./platform/jtag-smoke.tcl).Path
& C:/intelFPGA_lite/23.1std/quartus/sopc_builder/bin/system-console.exe "--script=$testScript"
```

The test claims one JTAG Avalon master, verifies ID/version/format and scratch,
uploads A/B, verifies signed operand readback, starts computation, polls for
completion, then compares all 256 INT32 results with a CPU reference. Its
12 jobs cover a padded 2x2 example (`19,22;43,50`), zeros, left/right identity,
INT8 extremes, the far-corner k=0 impulse and five signed random pairs. Every
completed job must report 384 fabric clocks. Errors fail the test; CPU
reference computation is never counted as a hardware completion.

If multiple masters exist, inspect `get_service_paths master` and select the
intended service explicitly before sourcing the script:

```tcl
set ::env(GEMM_MASTER_INDEX) 0
```

This test owns the whole accelerator and aborts any previous job. Do not run
it concurrently with another client. The five-second polling deadline covers
returned MMIO transactions; System Console's own transport timeout handles
stalled JTAG transactions.

## Local checks

```powershell
python ./sim/test_jtag_smoke.py
./sim/run-questa.ps1
```

The Python check uses an independent software MMIO model to verify the Tcl
script's arithmetic, transfers and error detection. It is not RTL simulation.
The Questa testbench drives actual RTL through its Avalon interface and checks
12 tiles / 3072 results; it requires a configured Questa license. The existing
Verilator/C++ tests remain available via `make lint sim` in a Linux toolchain.

See `validation.md` for actual outcomes in this environment. Do not assume a
bitstream exists just because Platform Designer generation succeeded.

## Transition to MNN

After this fabric test passes on the board, retain `systolic_mmio` and replace
the JTAG master with the existing HPS project's lightweight Avalon master.
Use that project's DDR/boot/pin configuration and assign the accelerator's
4 KiB window. Follow `board-bringup.md` for UIO, then `mnn/README.md` for the
existing hardware-required MNN test. Its current CPU MatMul hook accepts only
16x16 FP32 tensors containing exact INT8 integers. A dedicated MNN backend,
general tiling and TinyViT quantization are subsequent steps.
