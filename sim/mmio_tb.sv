`timescale 1ns/1ps
module mmio_tb;
    logic clk = 0;
    always #10 clk = ~clk;
    logic reset = 1;
    logic [9:0] address = 0;
    logic rd = 0, wr = 0;
    logic [31:0] wdata = 0;
    logic [3:0] be = 4'hf;
    wire [31:0] rdata;
    wire waitrequest;
    integer a[256], b[256];
    integer expected, got;
    logic [31:0] status, cycles;
    logic [31:0] random_state;
    systolic_mmio dut(clk, reset, address, rd, wr, wdata, be, rdata, waitrequest);
    task automatic write_word(input integer byte_address, input logic [31:0] value);
        @(negedge clk);
        address = 10'(byte_address >> 2); wdata = value; wr = 1; rd = 0;
        @(negedge clk); wr = 0;
    endtask
    task automatic read_word(input integer byte_address, output logic [31:0] value);
        @(negedge clk);
        address = 10'(byte_address >> 2); rd = 1; wr = 0;
        @(negedge clk); value = rdata; rd = 0;
    endtask
    initial begin
        repeat (3) @(negedge clk);
        reset = 0;
        read_word(0, status);
        if (status !== 32'h53413136) $fatal(1, "Wrong ID");
        for (int test = 0; test < 12; test++) begin
            random_state = test + 1;
            for (int p = 0; p < 256; p++) begin
                random_state = 1664525*random_state + 1013904223;
                a[p] = int'((random_state >> 16) & 255) - 128;
                random_state = 1664525*random_state + 1013904223;
                b[p] = int'((random_state >> 16) & 255) - 128;
                case (test)
                    0: begin a[p] = 0; b[p] = 0; end
                    1: a[p] = p/16 == p%16 ? 1 : 0;
                    2: b[p] = p/16 == p%16 ? 1 : 0;
                    3: begin a[p] = -128; b[p] = -128; end
                    4: begin a[p] = -128; b[p] = 127; end
                    5: begin a[p] = p == 240 ? -128 : 0; b[p] = p == 15 ? 127 : 0; end
                    6: begin
                        a[p] = p == 0 ? 1 : p == 1 ? 2 : p == 16 ? 3 : p == 17 ? 4 : 0;
                        b[p] = p == 0 ? 5 : p == 1 ? 6 : p == 16 ? 7 : p == 17 ? 8 : 0;
                    end
                    default: ;
                endcase
                write_word('h400 + 4*p, 32'(a[p]));
                write_word('h800 + 4*p, 32'(b[p]));
            end
            write_word(8, 1);
            status = 0;
            for (int poll = 0; poll < 400; poll++) begin
                read_word(12, status);
                if (status[2]) $fatal(1, "Accelerator error");
                if (status[1:0] == 2) break;
            end
            if (status[1:0] != 2) $fatal(1, "Timeout");
            read_word(16, cycles);
            if (cycles != 384) $fatal(1, "Cycles = %d", cycles);
            for (int i = 0; i < 16; i++) begin
                for (int j = 0; j < 16; j++) begin
                    expected = 0;
                    for (int k = 0; k < 16; k++) expected += a[16*i+k]*b[16*k+j];
                    read_word('hc00 + 4*(16*i+j), status);
                    got = $signed(status);
                    if (got !== expected) $fatal(1, "test %d C[%d,%d] got %d expected %d", test, i, j, got, expected);
                end
            end
            write_word(8, 2);
            $display("PASS RTL tile %0d, 256 outputs, 384 clocks", test);
        end
        $display("PASS RTL: 12 matrices, 3072 outputs");
        $finish;
    end
    initial begin #2000000; $fatal(1, "Testbench watchdog"); end
endmodule
