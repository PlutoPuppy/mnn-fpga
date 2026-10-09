module tinyvit (
    input  logic [9:0] SW,
    output logic [9:0] LEDR
);

    assign LEDR = SW;

endmodule