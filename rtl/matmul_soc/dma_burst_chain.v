// Issue the existing DMA's legal single-burst commands from one host command.
// The child holds the Avalon data ports and its original FIFO/backpressure.
module dma_burst_chain #(parameter BURST_BYTES = 2048) (
    input wire clk,
    input wire reset_n,
    input wire [2:0] addr,
    input wire cs,
    input wire wr_n,
    input wire [31:0] wdata,
    output wire [31:0] rdata,
    output reg [2:0] child_addr,
    output reg child_cs,
    output reg child_wr_n,
    output reg [31:0] child_wdata,
    input wire [31:0] child_rdata
);
    localparam [31:0] BURST_CONTROL = 32'h00000888;
    localparam [16:0] MAX_BURST = BURST_BYTES;
    localparam [3:0] IDLE=0, STOP=1, CLEAR=2, SRC=3, DST=4, LEN=5,
                     START=6, WAIT_DONE=7, FINISH=8, DONE=9, ERROR=10;
    reg [3:0] state;
    reg [31:0] source, destination;
    reg [16:0] remaining;
    reg length_valid, cancelled;
    reg override_result;
    reg [31:0] result;
    wire [16:0] chunk = remaining > MAX_BURST ? MAX_BURST : remaining;
    wire host_write = cs && !wr_n;
    wire host_control = host_write && addr == 6;
    wire chain_start = host_control && wdata[13] && wdata[3];
    wire stop_request = host_control && !wdata[3];
    wire active = state >= STOP && state <= FINISH;
    wire idle = state == IDLE;

    // Match the generated CSR port's registered read timing.
    assign rdata = override_result ? result : child_rdata;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            override_result <= 0;
            result <= 0;
        end else begin
            override_result <= addr == 4 || (!idle && (addr == 0 || addr == 3));
            case (addr)
                4: result <= 32'h444d4331;
                3: result <= {15'b0, remaining};
                default: result <= {29'b0, state == ERROR, active, state == DONE};
            endcase
        end
    end

    always @* begin
        child_addr = addr;
        child_cs = cs;
        child_wr_n = wr_n;
        child_wdata = wdata;
        if (!idle || chain_start) begin
            child_addr = 0;
            child_cs = 1;
            child_wr_n = 0;
            child_wdata = 0;
            case (state)
                STOP, FINISH: child_addr = 6;
                CLEAR: child_addr = 0;
                SRC: begin child_addr = 1; child_wdata = source; end
                DST: begin child_addr = 2; child_wdata = destination; end
                LEN: begin child_addr = 3; child_wdata = {15'b0, chunk}; end
                START: begin child_addr = 6; child_wdata = BURST_CONTROL; end
                WAIT_DONE: child_wr_n = 1;
                default: begin child_cs = 0; child_wr_n = 1; end
            endcase
        end
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state <= IDLE;
            source <= 0;
            destination <= 0;
            remaining <= 0;
            length_valid <= 0;
            cancelled <= 0;
        end else begin
            if (idle && host_write) begin
                case (addr)
                    1: source <= wdata;
                    2: destination <= wdata;
                    3: begin
                        remaining <= wdata[16:0];
                        length_valid <= wdata != 0 && wdata <= 65536 && wdata[3:0] == 0;
                    end
                    default: ;
                endcase
            end
            if (active && stop_request) cancelled <= 1;
            case (state)
                IDLE: if (chain_start) begin
                    cancelled <= 0;
                    if (length_valid && source[3:0] == 0 && destination[3:0] == 0)
                        state <= STOP;
                    else state <= ERROR;
                end
                STOP: state <= CLEAR;
                CLEAR: state <= SRC;
                SRC: state <= DST;
                DST: state <= LEN;
                LEN: state <= START;
                START: state <= WAIT_DONE;
                WAIT_DONE: if (child_rdata[0]) begin
                    remaining <= remaining - chunk;
                    if (remaining == chunk || cancelled || stop_request) state <= FINISH;
                    else begin
                        source <= source + {15'b0, chunk};
                        destination <= destination + {15'b0, chunk};
                        state <= STOP;
                    end
                end
                FINISH: state <= cancelled || stop_request ? IDLE : DONE;
                DONE, ERROR: if (stop_request) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
