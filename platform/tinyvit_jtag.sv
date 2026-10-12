// First fabric offload: PC -> JTAG Avalon master -> existing MMIO GEMM.
// KEY[0] resets the subsystem. No HPS/DDR configuration is required.
module tinyvit_jtag (
    input wire CLOCK_50,
    input wire [3:0] KEY,
    output wire [9:0] LEDR
);
    gemm_jtag system_inst (
        .clock_clk(CLOCK_50),
        .reset_reset_n(KEY[0])
    );
    assign LEDR = {9'b0, KEY[0]};
endmodule
