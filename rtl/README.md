# RTL

`systolic_pe.sv` and `systolic_array.sv` were extracted from your supplied
source; the datapath is unchanged. `systolic_mmio.sv` replaces the button/LED
controller with a host interface, buffers, skewed feeder and result capture.
Only these three files belong in the new Platform Designer component.

The MMIO wrapper now instantiates an **8x8 physical array (64 PEs)** and
computes four output quadrants sequentially, each over the full K=16. The
external operation remains 16x16 INT8 -> INT32. This replaces the 256-PE
implementation that requested 174 DSP blocks on an 87-DSP device. Each full
job now takes 384 accelerator clocks; the register ABI is unchanged.

`legacy/systolic_16by16.sv` preserves the original file unchanged. It contains
duplicate PE/array module declarations, so never compile it alongside the
new files. Its original top-level `tinyvit` is a matrix demo, not a complete
TinyViT model execution system.

The initial buffer implementation uses parallel combinational reads for all
8 lanes. Do not assume these arrays infer ordinary dual-port block RAM.
Use Quartus reports to assess mux/register cost, then bank memory or pipeline
feeding as a later optimization. The ABI does not depend on the storage type.
