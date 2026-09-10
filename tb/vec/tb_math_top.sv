// tb_math_top.sv — Verilator wrapper for the isqrt / udiv unit tests.
module tb_math_top (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic [47:0] sq_a,
  input  logic [31:0] dv_a,
  input  logic [23:0] dv_b24,
  input  logic [31:0] dv_b32,
  output logic        sq_busy,
  output logic        sq_done,
  output logic [23:0] sq_q,
  output logic        d24_busy,
  output logic        d24_done,
  output logic [31:0] d24_q,
  output logic        d32_busy,
  output logic        d32_done,
  output logic [31:0] d32_q
);
  isqrt u_isqrt (.clk(clk), .rst_n(rst_n), .start(start), .a(sq_a), .busy(sq_busy), .done(sq_done), .q(sq_q));
  udiv #(.AW(32), .BW(24)) u_d24 (.clk(clk), .rst_n(rst_n), .start(start), .a(dv_a), .b(dv_b24), .busy(d24_busy), .done(d24_done), .q(d24_q));
  udiv #(.AW(32), .BW(32)) u_d32 (.clk(clk), .rst_n(rst_n), .start(start), .a(dv_a), .b(dv_b32), .busy(d32_busy), .done(d32_done), .q(d32_q));
endmodule
