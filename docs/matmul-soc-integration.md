# One TinyViT MatMul on matmul-soc

Inspected upstream commit `4b749cedf88742dc999267999cd234186eb26bf7`
from <https://github.com/j3rryhu/matmul-soc>. Local inspection checkout:
`build/matmul-soc` (ignored by Git). No upstream RTL has been copied or changed.

## Quartus project and hardware contract

Open `adder_prototype.qpf` at the upstream repository root. Its revision is
`adder_prototype`, device `5CSEMA5F31C6`, top `de1soc_top`, and Platform Designer
system `adder_prototype.qsys`. Despite its historical name, the system contains
the matrix accelerator. The QSF references
`adder_prototype/synthesis/adder_prototype.qip`, which is not in the checkout:
regenerate Platform Designer synthesis outputs before compiling. The included
flow summary is historical evidence, not validation of a newly built bitstream.
The upstream 90 MHz design generated, synthesized, fitted and assembled on this
Windows installation, using 23,743/32,070 ALMs, 378/397 RAM blocks and 2/87 DSPs.
It **failed setup timing at -0.515 ns and was not programmed**. The revised
80 MHz board profile passed all 88 STA summary checks (minimum slack 0.077 ns,
slow 85 C core setup slack 2.049 ns) and was programmed successfully over JTAG.
Physical matrix smoke tests passed (48 submissions), followed by full TinyViT
inference with 320 submissions for the selected operation on each of two inputs.

The standard HPS ROM workflow requires unavailable WSL. The new build helper
creates a temporary installation view with directory junctions, a local native
launcher and Windows conversion of generated `/mnt/c/...` Makefile paths.
It uses Intel's installed `make.exe` and `uniphy_mcc.exe`; installed Quartus
files remain unchanged. This supports HPS ROM generation, not general Nios
software builds. With a short-path checkout of the pinned friend's source,
apply the explicit DE1-SoC profile before building:

```sh
python platform/apply_matmul_soc_profile.py /path/to/matmul-soc/adder_prototype.qsys
python platform/build_matmul_soc.py --friend-root /path/to/matmul-soc --native-hps-launcher
```

The helper generates the system, synthesizes it, applies HPS memory pin
assignments, fits, assembles and runs timing analysis. A negative or missing
core-clock slack now fails the helper even if Quartus returns exit code zero.
The profile also replaces the upstream exported `adder_prototype.out.sdc`
snapshot, whose fixed dividers incorrectly constrain the core at 100 MHz,
with a board reference clock and PLL clocks derived from the current netlist.
It never programs the
board. Use `--generate-only` or `--skip-generation` to resume a build. On this
machine the build checkout is `%TEMP%/mnn-matmul-soc`; the inspection clone in
`build/matmul-soc` is unchanged except for ignored generation attempts.

The profile fixes upstream placeholder HPS DDR geometry (8-bit, row12/column8)
to the DE1-SoC's 32-bit, 1 GB interface and uses an 80 MHz fabric clock.
The running board's bootloader and live DDR configuration are retained, not
replaced with generated handoff files. Before reconfiguration, isolate the
FPGA bridges. This PIO experiment needs no FPGA-to-SDRAM access; its ports stay
in reset. `socfpga_bridge_control` exposes explicit status/quiesce/enable-mmio
commands and never writes DDR timing, geometry, static configuration or port
width/maps. Do not run its write commands while another FPGA user is active.
The live bridge-isolation sequence is not yet validated on this board image:
the console stopped responding during its first use. Recover the HPS before
further testing; do not treat this helper as a proven deployment procedure.
See [Altera's reconfiguration requirements](https://docs.altera.com/r/docs/683435/17.1/fpga-sdk-for-opencl-cyclone-v-soc-development-kit-reference-platform-porting-guide/fpga-reconfiguration).

### Tested boot sequence on this board

The SD card's `u-boot.scr` loads the stock `soc_system.rbf` on every normal boot.
An HPS reset followed by normal boot therefore replaces a JTAG-loaded image.
Accessing `0xc0000000` with that stock image stalled Linux during testing.
Use this tested temporary sequence; no saved environment or SD files change:

1. Reset the HPS with KEY5 and interrupt U-Boot's five-second countdown over
   COM8 at 115200 baud. Leave it at `SOCFPGA_CYCLONE5 #`.
2. Program `build/matmul-soc-deploy/matmul-soc-80mhz.sof` over JTAG, FPGA
   device 2 (`quartus_pgm -c 1 -m JTAG -o "p;PATH_TO_SOF@2"`).
3. At U-Boot run `run bridge_enable_handoff; run mmcload; run mmcboot`.
   This uses the existing board handoff and boots Linux without `callscript`,
   retaining the accelerator. Do not run the normal `bootcmd` for this test.
4. Log in as ubuntu, enter `/home/ubuntu/tinyvit`, and run
   `sudo ./matmul_soc_smoke --devmem`. All six cases must pass with 48 jobs.
5. Run the selected-operation command below. The tested operation reports
   320 FPGA submissions. After another normal boot, repeat this sequence.

The isolated upstream Quartus project is `%TEMP%/mnn-matmul-soc/adder_prototype.qpf`.
Generated deploy artifacts are under `build/matmul-soc-deploy`; the original
board model and `tinyvit_run` were preserved. See [validation](validation.md)
for score comparisons. This validates execution, not a speedup or dataset accuracy.

The datapath is a physical 16x16 array with signed INT8 operands and INT32 PE
partial sums. **Software receives INT8 results**, after Q0.16 rescaling,
arithmetic right shift (floor for negative values), and saturation. The output
buffer has 16 banks of 4096 bytes = 65536 bytes, enough for a **256x256 INT8
canvas**, not 65536 INT32 results. W/X buffers are each 16 KB.

The full HPS-to-FPGA bridge maps the accelerator at `0xc0000000`, with a
128 KB span in upstream board tests. Verify the actual generated address map
for your bitstream. Registers are 16 bytes apart:

| Offset | Meaning used by board tests |
| --- | --- |
| 0x00 | Start pulse, bit 0 |
| 0x10 | Weight rows = output rows |
| 0x20 | Weight columns = reduction size |
| 0x30 | Input row pitch before rounding to 16 bytes |
| 0x40 | Input columns = output columns |
| 0x50 | Busy bit 0, done bit 1 (read clears done) |
| 0x60 | Unsigned 16-bit Q0.16 output scale |
| 0x70 | Tile row / tile column / use previous output |
| 0x80 | Output canvas columns |
| 0x1000 | Weight buffer |
| 0x5000 | Input buffer |
| 0x9000 | INT8 output buffer |

The RDL dimension comments disagree with the working C tests; this driver
follows `test_program/matmul_sw_tiling.c`. Dimensions must be programmed before
buffer writes because they control the hardware address remapper.

Buffer writes require **full 128-bit beats**: the RAM wrappers do not preserve
unwritten byte lanes. Four ordinary 32-bit stores cannot replace one beat.
The new ARMv7 NEON transport follows upstream's `vst1.8` / `vld1.8` PIO access.
It refuses non-ARM/NEON hosts. Registers use ordered 32-bit accesses in each
16-byte slot. Do not use this repository's original `fpga::Uio` or its ID probe
against the friend's bitstream; those belong to a different ABI.

## Tiling and numeric behavior

`fpga::matmul_soc_gemm` computes `[M,K] * [K,N]` using independent zero-padded
16x16x16 jobs. M/N/K may be arbitrary positive dimensions, including tails and
sizes exceeding 256. Transposed A/B and transposed output layouts are supported;
an optional column bias is added on the host. Each job uses a 16x16 output
canvas at its origin. This first implementation trades throughput for simple
correctness: it does not use DMA or retain multiple output tiles in the canvas.

Each operand tile is symmetrically quantized with scale `max(abs(x))/127`
(zero tile scale = 1), nearest rounding with ties away from zero. A row-L1
bound times B's maximum magnitude chooses a nonzero hardware output scale
that avoids saturation. The INT8 result is dequantized with the **actual
encoded** Q0.16 multiplier. Reduction tiles accumulate in host double precision,
then convert to FP32. Hardware `use_prev_out` is always zero: its saturating
INT8 accumulation would lose precision across reduction blocks. Output is
published only after the whole operation succeeds.

This is an **approximation**, including operand rounding and a downward floor
error in each output block. Quantization is opt-in through a distinct device
environment variable and operation selection. Model accuracy is not yet
established. Compare output scores and accuracy on representative images before
using this path for anything beyond an integration experiment. No speedup is
claimed; per-tile MMIO can be much slower than NEON CPU multiplication.

The original INT32 hardware also now has `fpga::exact_float_gemm`: integral
INT8-valued FP32 operands, arbitrary M/N tails, transpose support, and K<=1024
to preserve FP32 integer exactness. It performs no quantization.

## Build and select one operation

Windows can cross-build a static Cortex-A9 runner with Zig 0.13, CMake and
Ninja using `platform/arm-zig-windows.cmake`. Set `ZIG_EXE` to the compiler's
absolute path, `MNN_BUILD_HARD=ON`, `MNN_BUILD_SHARED_LIBS=OFF`,
`MNN_SEP_BUILD=OFF`, `MNN_ARM82=OFF`, and **`MNN_USE_THREAD_POOL=ON`**.
The thread pool avoids requiring OpenMP headers in this toolchain. The MNN
library and runner have both been built successfully with this setup. Set
`TINYVIT_STATIC=ON` when configuring `mnn/app`, and point `MNN_BUILD` at the
resulting library directory. Keep build/cache paths short on Windows.

The serial helpers `platform/serial-transfer.py` and
`platform/serial-download.py` exchange files with SHA256 verification through
an already logged-in Linux console; close PuTTY first. They require pyserial.
The new runner is deployed under a separate filename, `tinyvit_offload`, so
the earlier working `tinyvit_run` can remain available for a baseline.

Apply the updated `mnn/prepare.py` patch to a clean checkout of the pinned MNN
revision. For an already patched checkout, use a new checkout to avoid combining
old and new patches. Rebuild MNN and the runner using `mnn/app/README.md` and
`platform/arm-linux-gnueabihf.cmake`; the updated `mnn/fpga.cmake` includes both
drivers. Existing prebuilt `tinyvit-arm` binaries do not include these changes.

Run the full model on the CPU with runtime operation logging:

```sh
./tinyvit_run --model tiny-vit.mnn --height 224 --width 224 \
  --input-f32 input.f32 --list-ops --output-dir cpu-output
```

Choose an exact name for a runtime MatMul from that list. Runtime names can
differ from the serialized graph because MNN lowers batched operations. The
patch intercepts `CPUMatMul::execute`, including lowered batched calls, supports
transpose flags and bias, and keeps vector specializations on the CPU. Select
one runtime operation; if it contains several matrix batches, all its supported
batches are tiled. Other operations execute on the CPU.

After programming the matching FPGA bitstream, enabling the full bridge, and
binding a dedicated 128 KB UIO device (example in
`platform/matmul-soc-uio.dtsi.example`), run:

```sh
MNN_MATMUL_SOC_UIO=/dev/uio0 ./tinyvit_run \
  --model tiny-vit.mnn --height 224 --width 224 --input-f32 input.f32 \
  --fpga-op 'EXACT_NAME_FROM_LIST' --require-fpga --output-dir fpga-output
```

Use the correct UIO number from `/sys/class/uio`; no `/dev/mem` or reserved DDR
staging buffer is needed. Keep the device path fixed for the process lifetime.
The updated `sw/driver/systolic_uio.c` supports `mnn,matmul-soc-1.0` with a
strict 128 KB span, separately from its original 4 KB binding. Build it against
the board kernel and load it with the matching device-tree node. Do not map
arbitrary bridge addresses until the matching FPGA image and bridges are ready.

On the connected Ubuntu 16.04 / Linux 4.5 board, no UIO devices or build tools
were available. An explicit prototype fallback maps `/dev/mem` at the fixed
upstream address, requiring root. Use `MNN_MATMUL_SOC_DEVMEM=1` instead of
`MNN_MATMUL_SOC_UIO`; never set both. Only enable it after loading the matching
FPGA image. `matmul_soc_smoke --devmem` checks the hardware communication before
running MNN. Build that tool with `make build/native/matmul_soc_smoke` (substitute
your BUILD_DIR for a cross-build).
Only one synchronous session may use operation selection in a process.
`--require-fpga` checks successful whole-GEMM submissions during measured runs;
it fails on CPU-only execution, including an incompatible runtime route.
A missing operation name also fails. Device access is serialized and locked
against cooperating processes. A device failure disables the friend's path for
the process and falls back to CPU. There is no hardware abort register: after
a timeout reset/reprogram the FPGA before starting a new process.

For the original hardware, use `MNN_FPGA_UIO` instead; selected larger matrices
must satisfy the exact integer eligibility check. Do not set both device
environment variables. Without either variable the model runs on CPU.

## Validation boundary

`make test` checks both tiled software paths against an independent register /
datapath model: signed results, all transpose combinations, rectangular matrices,
M/N/K tails, output exceeding the 256-row canvas, nonfinite rejection, busy
hardware, and output preservation after a mid-operation failure. This does not
simulate the friend's RTL, validate NEON bridge transactions, or prove TinyViT
runtime dispatch. Those require the actual bitstream and ARM board. Save both
CPU and FPGA scores for the same preprocessed input, inspect their difference,
and require nonzero submission counts before calling the integration validated.

```sh
python3 mnn/app/compare_outputs.py cpu-output fpga-output
```

The comparator reports max absolute error, RMSE, relative L2, cosine similarity,
and classifier top indices. An optional `--max-abs-error VALUE` makes a chosen
absolute threshold a failing check. A single image's top-1 agreement does not
establish validation-set accuracy.
