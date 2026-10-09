module systolic_array #(

    parameter N      = 16,

    parameter DATA_W = 8,

    parameter ACC_W  = 32

)(

    input logic clk,

    input logic reset,

    input logic clear,


    // ------------------------------------------------------------
    // 16 values entering from LEFT
    // ------------------------------------------------------------

    input logic signed [DATA_W-1:0]
        a_left [0:N-1],


    // ------------------------------------------------------------
    // 16 values entering from TOP
    // ------------------------------------------------------------

    input logic signed [DATA_W-1:0]
        b_top [0:N-1],


    // ------------------------------------------------------------
    // 16 x 16 result matrix
    // ------------------------------------------------------------

    output logic signed [ACC_W-1:0]
        result [0:N-1][0:N-1]

);


    // ============================================================
    // Internal systolic connections
    //
    // A moves horizontally →
    //
    // B moves vertically ↓
    //
    // ============================================================


    // A connections
    //
    // N rows
    //
    // N+1 horizontal points

    logic signed [DATA_W-1:0]
        a_wire [0:N-1][0:N];


    // B connections
    //
    // N+1 vertical points
    //
    // N columns

    logic signed [DATA_W-1:0]
        b_wire [0:N][0:N-1];


    genvar r;
    genvar c;


    // ============================================================
    // Connect external A inputs to LEFT of array
    // ============================================================

    generate

        for (r = 0; r < N; r = r + 1) begin : A_INPUT

            assign a_wire[r][0] = a_left[r];

        end

    endgenerate


    // ============================================================
    // Connect external B inputs to TOP of array
    // ============================================================

    generate

        for (c = 0; c < N; c = c + 1) begin : B_INPUT

            assign b_wire[0][c] = b_top[c];

        end

    endgenerate


    // ============================================================
    // Generate 16 x 16 PE array
    //
    // Total:
    //
    // 16 * 16 = 256 Processing Elements
    //
    // ============================================================

    generate

        for (r = 0; r < N; r = r + 1) begin : ROW

            for (c = 0; c < N; c = c + 1) begin : COL

                systolic_pe #(

                    .DATA_W(DATA_W),

                    .ACC_W(ACC_W)

                ) pe_inst (

                    .clk(clk),

                    .reset(reset),

                    .clear(clear),


                    // A enters from left

                    .a_in(
                        a_wire[r][c]
                    ),


                    // B enters from top

                    .b_in(
                        b_wire[r][c]
                    ),


                    // A leaves to right

                    .a_out(
                        a_wire[r][c+1]
                    ),


                    // B leaves downward

                    .b_out(
                        b_wire[r+1][c]
                    ),


                    // PE accumulator is C[r][c]

                    .acc(
                        result[r][c]
                    )

                );

            end

        end

    endgenerate


endmodule
