# Programmable matrix shapes and register ABI review

Reviewed 2026-11-12 against `j3rryhu/matmul-accel` revision
`16cd018cc6648aa9571eb891ddc34da1e7e5acfb` and `j3rryhu/matmul-soc` revision
`b3ea448a36a5db7bd301173d4e5fff027eca2382`.

The array supports programmable matrix shapes. **K=64 is the current driver's
conservative reduction-panel policy, not an absolute accelerator limit.** The
driver's current register offsets already match the updated accelerator ABI.
The old SoC repository contains test programs from different ABI versions;
their register maps must not be mixed.

Follow-up: the [fc1 capacity experiment](tinyvit-fc1-capacity-benchmark.md)
implements an opt-in capacity-aware planner and physically verifies K=128 and
K=256 jobs on the same 90 MHz image. The fixed policy described below remains
the default for existing callers; its absolute K limit is not a hardware limit.

## Register map and compatibility

The reference is the pinned accelerator
[register definition](https://github.com/j3rryhu/matmul-accel/blob/16cd018cc6648aa9571eb891ddc34da1e7e5acfb/rdl/ctrl_reg.rdl)
and [top-level RTL](https://github.com/j3rryhu/matmul-accel/blob/16cd018cc6648aa9571eb891ddc34da1e7e5acfb/rtl/accel_top.sv).
Offsets below are bytes relative to the accelerator aperture:

| Register | Offset |
|---|---:|
| CONTROL | 0x00 |
| WEIGHT_ROWS | 0x10 |
| WEIGHT_COLS | 0x20 |
| INPUT_MAX_ADDR | 0x30 |
| INPUT_COLS | 0x40 |
| STATUS | 0x50 |
| OUTPUT_SCALE | 0x60 |
| TILE_CFG | 0x70 |
| TILED_MAT_DIM | 0x80 |

Buffer windows: W `0x1000..0x4fff` (16 KiB), X `0x5000..0x8fff`
(16 KiB), output `0x9000..0x18fff` (64 KiB). Buffer payload accesses are
full 128-bit beats; the control fields occupy the low register bits.
`sw/src/matmul_soc.cpp` and `matmul_soc_uio.cpp` already follow this ABI.

The older
[matmul_accel_test.c](https://github.com/j3rryhu/matmul-soc/blob/b3ea448a36a5db7bd301173d4e5fff027eca2382/test_program/matmul_accel_test.c)
instead defines four-byte control spacing and output at `0x7000`. It must not
be run unchanged against this image. In contrast,
[matmul_partial.c](https://github.com/j3rryhu/matmul-soc/blob/b3ea448a36a5db7bd301173d4e5fff027eca2382/test_program/matmul_partial.c)
uses 16-byte control spacing and output at `0x9000`. It is a useful shape/DMA
example, although some buffer-depth comments and its mapped span are specific
to that smaller test.

The audit compared all 12 accelerator RTL files and the generated register
file against both the local SoC accelerator source and the actual
`%TEMP%/mnn-matmul-soc-90mhz/adder_prototype/synthesis/submodules` inputs:
**13/13 match**, ignoring only final whitespace and text line-ending encoding.
There is no functional core/register-file difference in those files. The
validated deployment SOF SHA-256 remains
`0af66d4d24302b1ff9bb960ab189d41850603be1e8fd16edae29206fedfea058`.
This establishes source compatibility; no FPGA reprogramming, new simulation,
new physical matrix test or timing build was performed for this review.

The file-by-file hashes and capacity checks are recorded in
[shape-and-ABI-audit.json](benchmarks/shape-review-2026-11-12/shape-and-ABI-audit.json).

## What programmable shape means

Use conventional notation **W[M,K] times X[K,N] produces Y[M,N]**. The
testbench's local names differ: its `K` is output rows, `N` is the shared
reduction dimension, and `M` is output columns. Some older RTL/RDL comments
also use older axis descriptions. Follow the executable tests and actual
signal usage rather than inferring dimensions from those comments alone.

The [current testbench](https://github.com/j3rryhu/matmul-accel/blob/16cd018cc6648aa9571eb891ddc34da1e7e5acfb/tb/test_accel_top.py)
contains a 128x128 times 128x128 test, tests for partial output-row blocks, an
unaligned output-column case, and a tiled-canvas/use-previous-output case.
Its capacity assertions use per-bank depths and row pitches.

More precisely than requiring every weight dimension to be a multiple of 16:

- The shared reduction dimension K must be a multiple of 16 for direct
  execution on this datapath. A logical nonmultiple can be zero-padded by the
  host in both operands; that is how the current general driver handles tails.
- Output-row count M can be partial; the tests include 25 output rows.
- Output-column count N can be unaligned; the tests include 18 columns.
  Each stored operand/output row still has a 16-byte-aligned pitch.
- W and X must agree on their shared dimension. Both operand buffers and the
  output buffer must fit their bank layout, not just a nominal matrix edge.
- Dimension field widths also apply; INPUT_MAX_ADDR is 10 bits. Current
  fully padded submissions must not program 1024 into that field.

## Capacity is not a fixed 128x128 shape

For an untiled output origin and canvas width N, define
`pitch(q) = ceil(q/16)*16` bytes. The banked layout gives:

```text
weight: ceil(M/16) * pitch(K) <= 1024 bytes per bank
input:  ceil(K/16) * pitch(N) <= 1024 bytes per bank
output: ceil(M/16) * pitch(N) <= 4096 bytes per bank
```

For a larger output canvas or nonzero origin, the canvas bounds and offsets
must also be included. These formulas come from the testbench and address
remapper, not from an assumption that the buffers have fixed square dimensions.
Per-bank fit is stronger than checking only the sum of matrix bytes.

| W times X | Weight bytes/bank | Input bytes/bank | Output bytes/bank | Capacity/alignment result |
|---|---:|---:|---:|---|
| 128x128 times 128x128 | 1024 | 1024 | 1024 | Fits; explicit upstream test exists |
| 64x256 times 256x64 | 1024 | 1024 | 256 | Fits capacity; not newly board-tested here |
| 256x64 times 64x256 | 1024 | 1024 | 4096 | Fits; current full-canvas policy |
| 25x16 times 16x32 | 32 | 32 | 64 | Fits partial output rows |
| 20x32 times 32x18 | 64 | 64 | 64 | Fits unaligned output columns |
| 128x128 times 128x192 | 1024 | 1536 | 1536 | Input bank overflows; split the columns |
| 16x17 times 17x16 | 32 | 32 | 16 | Reduction alignment fails; pad K to 32 |
| 1x1280 times 1280x1 | 1280 | 1280 | 16 | Banks overflow despite small total live byte count |

Thus “128x128 bytes” describes each operand buffer's total 16 KiB capacity.
It is neither a mandatory shape nor permission to pair a fitting W with an
arbitrarily large X. Input values can be arbitrary signed INT8 values, subject
to the arithmetic's saturation behavior; floating-point inputs still require
conversion or a compatible quantized producer.

## Implications for this integration

The existing planner always chooses output panels up to 256x256 and reduction
panels up to 64. Those dimensions are safe together because both operand
buffers then fit. Smaller output tiles can permit a larger K panel, such as
128x128 times 128x128 or 64x256 times 256x64. The software should eventually
choose between such shapes using capacity, submissions, operand retransfers,
readback/accumulation cost and numerical error rather than a fixed K cap.
Fewer reduction panels can require more row/column panels, so they do not
automatically mean fewer jobs or lower latency.

The [CPU profiling report](tinyvit-cpu-operator-profile.md) already labels its
job counts as current-planner counts, not absolute hardware limits. Its stage-1
MLP fc1 and fc2 candidates remain useful. Their MNN runtime type is Convolution,
so a different integration hook is still required; programmable hardware
shape registers do not make the existing CPUMatMul hook intercept them.

The first QK currently uses 64 independent products of 49x32 times 32x49,
padded to 64x32 times 32x64. Each head already takes one hardware submission.
Increasing the allowed K panel cannot reduce that submission count. Partial-row
and unaligned-column support could reduce padding/data movement, but input
quantization and FP32 result reconstruction would remain. No performance
improvement from such a change was measured in this review.

The output arithmetic also remains important: the testbench's `golden_blocked`
models rescaling/flooring/saturation of each 16-wide reduction band followed
by saturated INT8 accumulation. `use_prev_out` accumulates those quantized
outputs; it is not a full-precision INT32 accumulation buffer. Supporting a
larger shape does not by itself eliminate numerical error or the host conversion
boundary. The older standalone DMA program also times DDR-to-DDR transfer/run
separately from CPU staging/readback and generates quantized operands before
that interval. That timing scope differs from the complete FP32 TinyViT offload.

This review updates the hardware/driver distinction and ABI reference. It does
not change the default tile policy or invalidate the previously captured
timing/output evidence.
