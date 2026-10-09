# mnn-fpga
Explore communication for MNN on HPS with FPGA fabric on DE1-SOC

[View full TinyViT graph](docs/tinyvit-graph.svg)

build two connections:
1. HPS ↔ FPGA: a hardware interface and Linux software that can submit matrix multiplications.
2. MNN ↔ that interface: an MNN execution implementation that offloads selected operations.
The .mnn model stays on the HPS. You send operands and commands to the FPGA.

# rough flow


# 1. Connect accelerator with HPS
```text
HPS: Linux
  MNN → FPGA execution code → Linux driver
                                  │
                      Lightweight HPS-to-FPGA bridge
                                  │ commands/status
                                  ▼
FPGA:                      Control registers
                                  │
                            Tile controller
                                  │
HPS DDR ↔ FPGA memory master ↔ A/B buffers
                                  │
                           4×4 systolic array
                                  │
                             Output buffer
```

In Platform Designer/Qsys, connect:
- HPS lightweight bridge → your accelerator’s memory-mapped control interface.
- FPGA memory master → configured HPS SDRAM interface.
- Clocks, resets, and optionally a completion interrupt.

# 2. Define small confined interface

For a GEMM operation \(C=AB\), your accelerator could expose these registers:
```text
Register	                               Purpose
CONTROL	                               Start a job
STATUS	                          Busy, complete, error
A_ADDR, B_ADDR, C_ADDR	      Device-visible buffer addresses
M, N, K	                             Matrix dimensions
A_STRIDE, B_STRIDE, C_STRIDE	         Row strides
```

Still a proposed interface, not an existing DE1-SoC standard. Fix the numeric format initially—for example, signed INT8 inputs and INT32 outputs—to keep the first implementation manageable.
The HPS submits a complete GEMM. Your FPGA controller handles the 4×4 output tiles and accumulation across K.



# 3. Connect the working GEMM interface to MNN
In the plutopuppy/matmul-soc repo, this call executes the model:
```cpp
net->runSession(session);
```

The offload belongs inside the execution of a selected operation during that call.
MNN has Backend and Execution abstractions: a backend creates an execution for a supported operation, and that execution provides preparation and compute methods such as onResize and onExecute. MNN backend documentation. Documentation: https://github.com/alibaba/MNN/wiki/backend

For the first prototype, I will:
modifying pinned MNN source to add an FPGA execution path for one supported operation, while retaining its existing CPU implementation for other cases. A dedicated FPGA backend is a larger, more general integration.

```cpp

// Pseudocode: these are your proposed helper functions.
onExecute(inputs, outputs) {
    pack_inputs_for_fpga(inputs);

    fpga_gemm_submit(a_buffer, b_buffer, c_buffer, M, N, K);
    fpga_wait_with_timeout();

    unpack_result_for_mnn(c_buffer, outputs);
}

```
which is essentially:

```text
MNN runs preceding operations on ARM
                 ↓
Selected linear operation calls your FPGA interface
                 ↓
FPGA computes the matrix multiplication
                 ↓
MNN receives the output and continues on ARM
```



