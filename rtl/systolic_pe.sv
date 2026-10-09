module systolic_pe #(

    parameter DATA_W = 8,

    parameter ACC_W  = 32

)(

    input logic clk,

    input logic reset,

    input logic clear,


    input logic signed [DATA_W-1:0] a_in,

    input logic signed [DATA_W-1:0] b_in,


    output logic signed [DATA_W-1:0] a_out,

    output logic signed [DATA_W-1:0] b_out,


    output logic signed [ACC_W-1:0] acc

);


    always_ff @(posedge clk) begin


        // ========================================================
        // Reset / clear
        // ========================================================

        if (reset || clear) begin

            a_out <= '0;

            b_out <= '0;

            acc <= '0;

        end


        // ========================================================
        // Normal systolic operation
        // ========================================================

        else begin


            // ----------------------------------------------------
            // Send A to PE on the RIGHT
            // ----------------------------------------------------

            a_out <= a_in;


            // ----------------------------------------------------
            // Send B to PE BELOW
            // ----------------------------------------------------

            b_out <= b_in;


            // ----------------------------------------------------
            // Multiply-accumulate
            //
            // C[r][c] += A * B
            // ----------------------------------------------------

            acc <= acc + (a_in * b_in);

        end

    end


endmodule
