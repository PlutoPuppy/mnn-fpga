module tinyvit (
    input  logic        CLOCK_50,
    input  logic [3:0]  KEY,
    input  logic [9:0]  SW,
    output logic [9:0]  LEDR
);

    // ============================================================
    // Parameters
    // ============================================================

    parameter N      = 16;
    parameter DATA_W = 8;
    parameter ACC_W  = 32;


    // ============================================================
    // Matrix storage
    //
    // A, B: 16 x 16 = 256 elements
    //
    // Row-major:
    //
    // A[i][k] = A_mem[i*16 + k]
    // B[k][j] = B_mem[k*16 + j]
    // C[i][j] = C_mem[i*16 + j]
    // ============================================================

    logic signed [DATA_W-1:0] A_mem [0:N*N-1];
    logic signed [DATA_W-1:0] B_mem [0:N*N-1];

    logic signed [ACC_W-1:0] C_mem [0:N*N-1];


    // ============================================================
    // Systolic array inputs / outputs
    // ============================================================

    logic signed [DATA_W-1:0] a_left [0:N-1];
    logic signed [DATA_W-1:0] b_top  [0:N-1];

    logic signed [ACC_W-1:0] result [0:N-1][0:N-1];

    logic clear_array;


    // ============================================================
    // FSM
    // ============================================================

    localparam IDLE  = 3'd0;
    localparam CLEAR = 3'd1;
    localparam RUN   = 3'd2;
    localparam STORE = 3'd3;
    localparam DONE  = 3'd4;

    logic [2:0] state;

    // Need enough bits for about 46 cycles
    logic [6:0] cycle;

    // 0-255
    logic [7:0] store_index;

    logic key1_prev;

    integer row;
    integer col;


    // ============================================================
    // Test matrix initialization
    //
    // A = matrix containing:
    //
    // A[i][k] = k + 1
    //
    // Therefore every row is:
    //
    // [1 2 3 ... 16]
    //
    //
    // B = identity matrix
    //
    // Therefore:
    //
    // C = A * I = A
    //
    // This is easy to verify on the LEDs.
    //
    // Expected:
    //
    // Every C row =
    //
    // [1 2 3 4 ... 16]
    //
    // ============================================================

    initial begin

        for (row = 0; row < N; row = row + 1) begin

            for (col = 0; col < N; col = col + 1) begin

                // A[row][col] = col + 1
                A_mem[row*N + col] = col + 1;

                // B = identity matrix
                if (row == col)
                    B_mem[row*N + col] = 8'sd1;
                else
                    B_mem[row*N + col] = 8'sd0;

                C_mem[row*N + col] = '0;

            end

        end

    end


    // ============================================================
    // SYSTOLIC INPUT SKEWING
    //
    // A travels LEFT -> RIGHT
    //
    // B travels TOP -> BOTTOM
    //
    //
    // For A row:
    //
    // k = cycle - row
    //
    //
    // For B column:
    //
    // k = cycle - column
    //
    // ============================================================

    always_comb begin

        // --------------------------------------------------------
        // Default everything to zero
        // --------------------------------------------------------

        for (row = 0; row < N; row = row + 1) begin
            a_left[row] = '0;
        end

        for (col = 0; col < N; col = col + 1) begin
            b_top[col] = '0;
        end


        // --------------------------------------------------------
        // Feed A from LEFT
        //
        // A[row][k]
        //
        // k = cycle - row
        // --------------------------------------------------------

        for (row = 0; row < N; row = row + 1) begin

            if ((cycle >= row) &&
                ((cycle - row) < N)) begin

                a_left[row] =
                    A_mem[row*N + (cycle - row)];

            end

        end


        // --------------------------------------------------------
        // Feed B from TOP
        //
        // B[k][col]
        //
        // k = cycle - col
        // --------------------------------------------------------

        for (col = 0; col < N; col = col + 1) begin

            if ((cycle >= col) &&
                ((cycle - col) < N)) begin

                b_top[col] =
                    B_mem[(cycle - col)*N + col];

            end

        end

    end


    // ============================================================
    // 16 x 16 SYSTOLIC ARRAY
    // ============================================================

    systolic_array #(
        .N(N),
        .DATA_W(DATA_W),
        .ACC_W(ACC_W)
    ) systolic_inst (

        .clk(CLOCK_50),

        .reset(!KEY[0]),

        .clear(clear_array),

        .a_left(a_left),

        .b_top(b_top),

        .result(result)

    );


    // ============================================================
    // Main FSM controller
    // ============================================================

    always_ff @(posedge CLOCK_50) begin

        key1_prev <= KEY[1];

        // --------------------------------------------------------
        // KEY[0] = reset
        // active LOW
        // --------------------------------------------------------

        if (!KEY[0]) begin

            state       <= IDLE;

            cycle       <= 0;

            store_index <= 0;

            clear_array <= 0;

            for (row = 0; row < N*N; row = row + 1) begin
                C_mem[row] <= '0;
            end

        end

        else begin

            case (state)

                // =================================================
                // IDLE
                //
                // Press KEY[1] to start
                // =================================================

                IDLE: begin

                    clear_array <= 0;

                    cycle <= 0;

                    if (key1_prev && !KEY[1]) begin
                        state <= CLEAR;
                    end

                end


                // =================================================
                // Clear all 256 PE accumulators
                // =================================================

                CLEAR: begin

                    clear_array <= 1;

                    cycle <= 0;

                    state <= RUN;

                end


                // =================================================
                // Run systolic computation
                //
                // For N x N systolic:
                //
                // approximately 3N - 2 cycles
                //
                // For N = 16:
                //
                // 3*16 - 2 = 46 cycles
                // =================================================

                RUN: begin

                    clear_array <= 0;

                    if (cycle == (3*N - 2)) begin

                        store_index <= 0;

                        state <= STORE;

                    end

                    else begin

                        cycle <= cycle + 1'b1;

                    end

                end


                // =================================================
                // Store PE accumulators into C_mem
                //
                // 256 elements
                //
                // One result written each clock
                // =================================================

                STORE: begin

                    C_mem[store_index] <=
                        result[store_index / N]
                              [store_index % N];

                    if (store_index == N*N - 1) begin

                        state <= DONE;

                    end

                    else begin

                        store_index <= store_index + 1'b1;

                    end

                end


                // =================================================
                // DONE
                // =================================================

                DONE: begin

                    state <= DONE;

                end


                default: begin

                    state <= IDLE;

                end

            endcase

        end

    end


    // ============================================================
    // LED display
    //
    // SW[7:0] selects one of the 256 C matrix elements.
    //
    //
    // Address:
    //
    //      index = row * 16 + column
    //
    //
    // Examples:
    //
    // SW = 0
    //      C[0][0]
    //
    // SW = 1
    //      C[0][1]
    //
    // SW = 15
    //      C[0][15]
    //
    // SW = 16
    //      C[1][0]
    //
    // SW = 17
    //      C[1][1]
    //
    // ...
    //
    // SW = 255
    //      C[15][15]
    //
    // ============================================================

    always_comb begin

        LEDR = C_mem[SW[7:0]][9:0];

    end


endmodule



// ============================================================================
// ============================================================================
//
//                      SYSTOLIC ARRAY
//
// ============================================================================
// ============================================================================

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



// ============================================================================
// ============================================================================
//
//                      PROCESSING ELEMENT
//
// ============================================================================
// ============================================================================
//
// Each PE does:
//
//             B in
//              |
//              v
//
// A in ---> [ PE ] ---> A out
//            |
//            |
//            v
//          B out
//
//
//
// Internally:
//
// acc = acc + A * B
//
//
//
// A propagates horizontally.
//
// B propagates vertically.
//
// Result stays locally in the PE accumulator.
//
// ============================================================================

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