// Native W8A8 output: accumulate every K=16 band in INT32, then apply the
// per-output-column Q24 multiplier and Q16 bias once. Requires weight_ctrl
// traversal with reduction bands inner and output-row blocks outer.
// Only one 16-row block of wide partials is live: 16*128*4 = 8192 bytes.
module output_loader_native #(
    parameter DATA_WIDTH=8, ACC_WIDTH=32, ARRAY_ROWS=16, ARRAY_COLS=16,
    parameter ROW_ADDR_WIDTH=12, RD_DATA_WIDTH=128, DIM_WIDTH=16,
    parameter ROW_RADDR_WIDTH=ROW_ADDR_WIDTH-$clog2(RD_DATA_WIDTH/DATA_WIDTH),
    parameter COL_SEL_W=$clog2(ARRAY_COLS+1), ROW_SEL_W=$clog2(ARRAY_ROWS+1)
)(
    input clock, rst_n,
    input [ARRAY_ROWS*ACC_WIDTH-1:0] p_in,
    output [ARRAY_COLS-1:0] b_en,
    input a_en_last,
    input [ROW_ADDR_WIDTH-1:0] total_rows,
    input [DIM_WIDTH-1:0] committed_k_blk_idx, committed_n_blk_idx,
    input committed_first_k_blk,
    input [DIM_WIDTH-1:0] tile_row_blk,
    input [ROW_ADDR_WIDTH-1:0] tile_col_off,
    input use_prev_out,
    input [ROW_ADDR_WIDTH:0] row_pitch,
    input [15:0] reduction_cols,
    input [ROW_SEL_W-1:0] weight_rows,
    input [COL_SEL_W-1:0] weight_cols,
    input cfg_write,
    input [6:0] cfg_index,
    input [127:0] cfg_data,
    input lut_write,
    input [3:0] lut_index,
    input [127:0] lut_data,
    input lut_enable,
    output [ARRAY_COLS-1:0] obuf_rden,
    output [ARRAY_COLS*ROW_RADDR_WIDTH-1:0] obuf_rdaddress,
    input [ARRAY_COLS*RD_DATA_WIDTH-1:0] obuf_q,
    output [ARRAY_COLS-1:0] obuf_wren,
    output [ARRAY_COLS*ROW_ADDR_WIDTH-1:0] obuf_waddr,
    output [ARRAY_COLS*DATA_WIDTH-1:0] obuf_wdata,
    output busy, done
);
    reg running;
    reg [4:0] elapsed;
    reg [ROW_ADDR_WIDTH-1:0] row_count[0:ARRAY_ROWS-1];
    wire [ARRAY_ROWS-1:0] row_done;
    reg [ROW_ADDR_WIDTH-1:0] segment_base;
    reg first_band, final_band;
    wire [ROW_ADDR_WIDTH-1:0] next_segment =
        (tile_row_blk + committed_k_blk_idx)*row_pitch + tile_col_off;
    wire pass_start = !running && a_en_last;
    wire [ARRAY_ROWS-1:0] pipe_busy;
    assign b_en={ARRAY_COLS{running}};
    assign obuf_rden=0;
    assign obuf_rdaddress=0;
    assign done=running && &(row_done | ~((32'd1 << weight_rows)-1));
    assign busy=running || |pipe_busy;

    always @(posedge clock) begin
        if (!rst_n) begin
            running<=0;elapsed<=0;segment_base<=0;first_band<=0;final_band<=0;
        end else if (running) begin
            if (elapsed<ARRAY_ROWS-1) elapsed<=elapsed+1'b1;
            if (done) begin running<=0;elapsed<=0;end
        end else if (a_en_last) begin
            running<=1;elapsed<=0;segment_base<=next_segment;
            first_band<=committed_n_blk_idx==0;
            final_band<=((committed_n_blk_idx+1)*ARRAY_COLS)>=reduction_cols;
        end
    end

    genvar r;
    generate for (r=0;r<ARRAY_ROWS;r=r+1) begin:ROW
        wire active=running && elapsed>=r && r<weight_rows && row_count[r]<total_rows;
        assign row_done[r]=row_count[r]==total_rows;
        always @(posedge clock) begin
            if (!rst_n || pass_start) row_count[r]<=0;
            else if (active) row_count[r]<=row_count[r]+1'b1;
        end

        // No reset on RAM contents. A first-band write initializes each used
        // location before a subsequent band reads it. Coefficients are loaded
        // before launching a job. Replicated MLABs provide a read per PE lane.
        (* ramstyle="M10K" *) reg signed [31:0] partials[0:127];
        // One 128-bit host write contains two (Q24 scale,Q16 bias) pairs.
        // Store pairs together so each replica has one write port, not two.
        (* ramstyle="MLAB" *) reg [127:0] parameters[0:63];
        (* ramstyle="MLAB" *) reg [127:0] activation_lut[0:15];
        reg [127:0] lookup_word;
        reg [3:0] lookup_lane;
        reg signed [7:0] bypass_value;
        reg lookup_valid;
        reg [ROW_ADDR_WIDTH-1:0] lookup_address;
        reg signed [31:0] old_partial;
        reg [127:0] parameters_read;
        reg parameter_lane;
        reg [4:0] valid_pipe,final_pipe;
        reg [ROW_ADDR_WIDTH-1:0] address_pipe[0:4];
        reg [6:0] index0;
        reg first0;
        reg signed [31:0] psum0,accum1;
        reg [23:0] coeff1;
        reg signed [31:0] bias1,bias2,bias3;
        (* multstyle="dsp" *) reg signed [40:0] product_hi2;
        (* multstyle="dsp" *) reg [39:0] product_lo2;
        reg signed [55:0] product3,biased4;
        wire signed [31:0] accumulated=first0 ? psum0 : old_partial+psum0;
        wire signed [55:0] hi_extended=product_hi2;
        wire signed [55:0] bias_extended=bias3;
        wire signed [55:0] rounded=biased4+(biased4[55] ? 56'sd8388607 : 56'sd8388608);
        wire signed [31:0] integer_result=rounded>>>24;
        wire signed [7:0] quantized=integer_result>127 ? 8'sd127 :
            (integer_result< -127 ? -8'sd127 : integer_result[7:0]);

        always @(posedge clock) begin
            if (cfg_write) begin
                parameters[cfg_index[6:1]]<=cfg_data;
            end
            if (lut_write) activation_lut[lut_index]<=lut_data;
            lookup_word<=activation_lut[quantized[7:4]];
            lookup_lane<=quantized[3:0];
            bypass_value<=quantized;
            if (active) begin
                old_partial<=partials[row_count[r][6:0]];
                parameters_read<=parameters[row_count[r][6:1]];
                parameter_lane<=row_count[r][0];
            end
            if (valid_pipe[0]) partials[index0]<=accumulated;
        end

        integer i;
        always @(posedge clock) begin
            if (!rst_n) begin
                valid_pipe<=0;final_pipe<=0;index0<=0;first0<=0;
                psum0<=0;accum1<=0;coeff1<=0;bias1<=0;bias2<=0;bias3<=0;
                product_hi2<=0;product_lo2<=0;product3<=0;biased4<=0;
                lookup_valid<=0;lookup_address<=0;
                for(i=0;i<5;i=i+1)address_pipe[i]<=0;
            end else begin
                valid_pipe<={valid_pipe[3:0],active};
                final_pipe<={final_pipe[3:0],final_band};
                address_pipe[0]<=segment_base+row_count[r];
                for(i=1;i<5;i=i+1)address_pipe[i]<=address_pipe[i-1];
                index0<=row_count[r][6:0];first0<=first_band;
                psum0<=p_in[r*32+:32];
                accum1<=accumulated;
                coeff1<=parameter_lane ? parameters_read[87:64] : parameters_read[23:0];
                bias1<=parameter_lane ? parameters_read[127:96] : parameters_read[63:32];
                product_hi2<=$signed(accum1[31:16])*$signed({1'b0,coeff1});
                product_lo2<=accum1[15:0]*coeff1;
                bias2<=bias1;
                product3<=(hi_extended<<<16)+$signed({1'b0,product_lo2});
                bias3<=bias2;
                biased4<=product3+(bias_extended<<<8);
                lookup_valid<=valid_pipe[4] && final_pipe[4];
                lookup_address<=address_pipe[4];
            end
        end
        assign pipe_busy[r]=|valid_pipe || lookup_valid;
        assign obuf_wren[r]=lookup_valid;
        assign obuf_waddr[r*ROW_ADDR_WIDTH+:ROW_ADDR_WIDTH]=lookup_address;
        assign obuf_wdata[r*8+:8]=lut_enable ? lookup_word[lookup_lane*8+:8] : bypass_value;
    end endgenerate
endmodule
