// Fixed 16x16 signed INT8 GEMM, using an 8x8 array for four output quadrants.
// INT32 output. Avalon address is a WORD index; the host ABI stays 16x16.
// See docs/register-map.md. All logic uses clk; reset is synchronous active-high.
module systolic_mmio (
    input  logic        clk,
    input  logic        reset,
    input  logic [9:0]  avs_address,
    input  logic        avs_read,
    input  logic        avs_write,
    input  logic [31:0] avs_writedata,
    input  logic [3:0]  avs_byteenable,
    output logic [31:0] avs_readdata,
    output wire         avs_waitrequest
);
    localparam [31:0] ID = 32'h53413136; // "SA16"
    typedef enum logic [2:0] {IDLE, CLEAR, RUN, DRAIN, STORE} state_t;
    state_t state;
    logic signed [7:0] a_mem [0:255];
    logic signed [7:0] b_mem [0:255];
    logic signed [31:0] c_mem [0:255];
    logic signed [7:0] a_left [0:7];
    logic signed [7:0] b_top [0:7];
    wire signed [31:0] result [0:7][0:7];
    logic [4:0] cycle;
    logic [5:0] store_index;
    logic [1:0] quadrant; // bit 1: output row half; bit 0: output column half
    wire [7:0] output_index = {quadrant[1], store_index[5:3],
                               quadrant[0], store_index[2:0]};
    logic done, error;
    logic [31:0] cycles, scratch;
    wire busy = state != IDLE;
    wire control_write = avs_write && avs_address == 10'h002 && avs_byteenable[0];
    wire abort_job = control_write && avs_writedata[2];
    assign avs_waitrequest = 1'b0;

    // Each quadrant reduces over all 16 k values, with local lane delays.
    // 30 RUN edges cover t=0..29; the last product is at 15+7+7.
    always_comb begin
        for (int lane = 0; lane < 8; lane++) begin
            a_left[lane] = '0;
            b_top[lane] = '0;
            if (state == RUN && int'(cycle) >= lane && int'(cycle) < lane + 16) begin
                a_left[lane] = a_mem[(int'(quadrant[1])*8+lane)*16 + int'(cycle)-lane];
                b_top[lane] = b_mem[(int'(cycle)-lane)*16 + int'(quadrant[0])*8+lane];
            end
        end
    end

    systolic_array #(.N(8), .DATA_W(8), .ACC_W(32)) array_inst (
        .clk(clk), .reset(reset), .clear(state == CLEAR || abort_job),
        .a_left(a_left), .b_top(b_top), .result(result)
    );

    // Fixed read latency: one clock, with no wait states. No combinational
    // read of the Avalon address is exposed to the master.
    always_ff @(posedge clk) begin
        if (reset) avs_readdata <= 0;
        else if (avs_read) begin
            case (avs_address)
                10'h000: avs_readdata <= ID;
                10'h001: avs_readdata <= 32'h00010000;
                10'h002: avs_readdata <= 0;
                10'h003: avs_readdata <= {29'b0, error, done, busy};
                10'h004: avs_readdata <= cycles;
                10'h005: avs_readdata <= scratch;
                10'h006: avs_readdata <= 32'h20101008; // acc,N,K,input widths
                default: begin
                    case (avs_address[9:8])
                        2'b01: avs_readdata <= {{24{a_mem[avs_address[7:0]][7]}}, a_mem[avs_address[7:0]]};
                        2'b10: avs_readdata <= {{24{b_mem[avs_address[7:0]][7]}}, b_mem[avs_address[7:0]]};
                        2'b11: avs_readdata <= c_mem[avs_address[7:0]];
                        default: avs_readdata <= 0;
                    endcase
                end
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (reset || abort_job) begin
            state <= IDLE;
            cycle <= 0;
            store_index <= 0;
            quadrant <= 0;
            done <= 0;
            error <= 0;
            cycles <= 0;
            scratch <= 0;
        end else begin
            if (busy) cycles <= cycles + 1'b1;
            case (state)
                IDLE: ;
                CLEAR: begin cycle <= 0; state <= RUN; end
                RUN: begin
                    if (cycle == 29) state <= DRAIN;
                    else cycle <= cycle + 1'b1;
                end
                DRAIN: begin store_index <= 0; state <= STORE; end
                STORE: begin
                    c_mem[output_index] <= result[store_index[5:3]][store_index[2:0]];
                    if (store_index == 63) begin
                        if (quadrant == 3) begin state <= IDLE; done <= 1; end
                        else begin quadrant <= quadrant + 1'b1; state <= CLEAR; end
                    end else store_index <= store_index + 1'b1;
                end
                default: begin state <= IDLE; error <= 1; end
            endcase

            if (avs_write) begin
                if (control_write) begin
                    if (avs_writedata[1]) begin done <= 0; error <= 0; end
                    if (avs_writedata[0]) begin
                        if (busy) error <= 1;
                        else begin
                            state <= CLEAR;
                            quadrant <= 0;
                            done <= 0;
                            error <= 0;
                            cycles <= 0;
                        end
                    end
                end else if (avs_address == 10'h005) begin
                    for (int b = 0; b < 4; b++)
                        if (avs_byteenable[b]) scratch[8*b +: 8] <= avs_writedata[8*b +: 8];
                end else if (avs_address[9:8] == 2'b01 || avs_address[9:8] == 2'b10) begin
                    if (avs_byteenable[0]) begin
                        if (busy) error <= 1;
                        else if (avs_address[9:8] == 2'b01) a_mem[avs_address[7:0]] <= avs_writedata[7:0];
                        else b_mem[avs_address[7:0]] <= avs_writedata[7:0];
                    end
                end else if (avs_address != 10'h002 && |avs_byteenable) error <= 1;
            end
        end
    end
endmodule
