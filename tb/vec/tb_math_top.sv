// tb_math_top.sv — Verilator wrapper for the isqrt / udiv unit tests.
module tb_math_top (
  input logic signed [63:0] round_v,
  input logic [5:0] round_s,
  output logic signed [63:0] round_q,
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
  output logic [31:0] d32_q,
  output logic        d1_done,
  output logic        d1_q,
  output logic        d2_done,
  output logic [1:0]  d2_q
);
  assign round_q = llaccel_pkg::rshr64(round_v, round_s);
  isqrt u_isqrt (.clk(clk), .rst_n(rst_n), .start(start), .a(sq_a), .busy(sq_busy), .done(sq_done), .q(sq_q));
  udiv #(.AW(32), .BW(24)) u_d24 (.clk(clk), .rst_n(rst_n), .start(start), .a(dv_a), .b(dv_b24), .busy(d24_busy), .done(d24_done), .q(d24_q));
  udiv #(.AW(32), .BW(32)) u_d32 (.clk(clk), .rst_n(rst_n), .start(start), .a(dv_a), .b(dv_b32), .busy(d32_busy), .done(d32_done), .q(d32_q));
  logic unused_d1_busy, unused_d2_busy;
  udiv #(.AW(1), .BW(1)) u_d1 (.clk(clk), .rst_n(rst_n), .start(start), .a(dv_a[0]), .b(dv_b32[0]), .busy(unused_d1_busy), .done(d1_done), .q(d1_q));
  udiv #(.AW(2), .BW(2)) u_d2 (.clk(clk), .rst_n(rst_n), .start(start), .a(dv_a[1:0]), .b(dv_b32[1:0]), .busy(unused_d2_busy), .done(d2_done), .q(d2_q));
endmodule
