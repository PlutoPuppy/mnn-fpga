`timescale 1ns/1ps
module native_output_test;
    reg clk=0;always #5 clk=~clk;
    reg reset_n=0,a_last=0,cfg_write=0;
    reg [511:0] p=0;
    reg [11:0] columns=16;
    reg [15:0] block_id=0,band_id=0,reduction=16;
    reg [4:0] valid_rows=16;
    reg [12:0] pitch=16;
    reg [6:0] cfg_index=0;
    reg [127:0] cfg_data=0;
    wire [15:0] we,b_en;
    wire [191:0] addresses;
    wire [127:0] values;
    wire busy,done;
    integer bands=1,current_block=0,writes=0,checks=0,test_cases=0;
    integer coeff[0:127],bias[0:127];
    integer r,c,t,b,m,n,expected_value;
    integer expected_writes;
    output_loader_native dut(
        .clock(clk),.rst_n(reset_n),.p_in(p),.b_en(b_en),.a_en_last(a_last),
        .total_rows(columns),.committed_k_blk_idx(block_id),.committed_n_blk_idx(band_id),
        .committed_first_k_blk(band_id==0),.tile_row_blk(16'd0),.tile_col_off(12'd0),
        .use_prev_out(1'b0),.row_pitch(pitch),.reduction_cols(reduction),
        .weight_rows(valid_rows),.weight_cols(5'd16),.cfg_write(cfg_write),.cfg_index(cfg_index),.cfg_data(cfg_data),
        .lut_write(1'b0),.lut_index(4'd0),.lut_data(128'd0),.lut_enable(1'b0),
        .obuf_rden(),.obuf_rdaddress(),.obuf_q(2048'd0),.obuf_wren(we),.obuf_waddr(addresses),
        .obuf_wdata(values),.busy(busy),.done(done));

    function automatic integer partial(input integer band,blk,lane,col);
        if(col==0) partial=(band%2==0 ? 50000 : -50000);
        else partial=((band*119+blk*37+lane*53+col*71)%2001)-1000;
    endfunction
    function automatic integer reference(input integer blk,lane,col);
        longint signed sum,x,y;
        integer j;
        begin
            sum=0;for(j=0;j<bands;j=j+1)sum=sum+partial(j,blk,lane,col);
            x=sum*coeff[col]+($signed(bias[col])*256);
            y=(x+(x<0 ? 8388607 : 8388608))>>>24;
            reference=y>127 ? 127 : (y< -127 ? -127 : y);
        end
    endfunction
    always @(negedge clk) begin
        for(integer j=0;j<16;j=j+1)
            p[j*32+:32]=partial(band_id,current_block,j,dut.row_count[j]);
    end
    always @(posedge clk) if(reset_n) begin
        for(integer j=0;j<16;j=j+1)if(we[j])begin
            c=addresses[j*12+:12]-current_block*pitch;
            if(band_id!=bands-1 || j>=valid_rows || c<0 || c>=columns)
                $fatal(1,"Invalid/early write band=%d lane=%d column=%d",band_id,j,c);
            expected_value=reference(current_block,j,c);
            if($signed(values[j*8+:8])!=expected_value)
                $fatal(1,"Mismatch bands=%d block=%d lane=%d col=%d got=%d expected=%d",
                    bands,current_block,j,c,$signed(values[j*8+:8]),expected_value);
            writes=writes+1;checks=checks+1;
        end
    end
    initial begin
        repeat(4)@(negedge clk);reset_n=1;
        for(t=0;t<128;t=t+1)begin
            coeff[t]=(t==0 ? 4194304 : 10000+t*37);
            bias[t]=(t%9-4)*32768;
        end
        for(t=0;t<128;t=t+2)begin
            @(negedge clk);cfg_write=1;cfg_index=t;
            cfg_data={bias[t+1],coeff[t+1],bias[t],coeff[t]};
        end
        @(negedge clk);cfg_write=0;
        for(n=0;n<3;n=n+1)for(b=0;b<4;b=b+1)begin
            columns=(n==0 ? 16 : (n==1 ? 49 : 128));pitch=((columns+15)/16)*16;
            bands=(b==0 ? 1 : (b==1 ? 2 : (b==2 ? 3 : 8)));reduction=bands*16;
            for(m=0;m<3;m=m+1)begin
                current_block=m;block_id=m;valid_rows=(m==2 ? 5 : 16);writes=0;
                expected_writes=valid_rows*columns;
                for(integer j=0;j<bands;j=j+1)begin
                    @(negedge clk);band_id=j;a_last=1;
                    @(negedge clk);a_last=0;
                    wait(busy);wait(!busy);repeat(2)@(negedge clk);
                end
                if(writes!=expected_writes)$fatal(1,"Write count got=%d expected=%d",writes,expected_writes);
                test_cases=test_cases+1;
            end
        end
        $display("PASS native output: %0d cases, %0d checked outputs, cancellation, tails, biases, saturation",test_cases,checks);
        $finish;
    end
    initial begin #1000000;$fatal(1,"Timeout");end
endmodule
