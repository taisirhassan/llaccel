// tb_vec_top.sv — Verilator wrapper: vec_engine + behavioral SRAM stub.
module tb_vec_top (
  input  logic         clk,
  input  logic         rst_n,
  input  logic [511:0] instr_flat,
  input  logic         instr_valid,
  output logic         instr_ready,
  input  logic [31:0]  pos,
  input  logic [7:0]   deny_pct,
  output logic         busy,
  output logic         done_pulse,
  output logic [7:0]   done_sig_sem,
  output logic         perf_busy,
  output logic         perf_sram_stall,
  output logic         rd0_active,     // request/grant activity, for cycle accounting
  output logic         rd1_active,
  output logic         wr_active
);
  import llaccel_pkg::*;

  instr_words_t instr;
  assign instr = instr_flat;

  logic rd0_valid, rd0_grant, rd0_rvalid;
  logic rd1_valid, rd1_grant, rd1_rvalid;
  logic wr_valid, wr_grant;
  sram_req_t rd0_req, rd1_req, wr_req;
  logic [511:0] rd0_rdata, rd1_rdata, wr_wdata;
  logic [63:0] wr_wstrb;

  vec_engine u_dut (
    .clk(clk), .rst_n(rst_n),
    .instr_valid(instr_valid), .instr(instr), .instr_ready(instr_ready), .pos(pos),
    .busy(busy), .done_pulse(done_pulse), .done_sig_sem(done_sig_sem),
    .vec_rd0_valid(rd0_valid), .vec_rd0_req(rd0_req), .vec_rd0_grant(rd0_grant), .vec_rd0_rvalid(rd0_rvalid), .vec_rd0_rdata(rd0_rdata),
    .vec_rd1_valid(rd1_valid), .vec_rd1_req(rd1_req), .vec_rd1_grant(rd1_grant), .vec_rd1_rvalid(rd1_rvalid), .vec_rd1_rdata(rd1_rdata),
    .vec_wr_valid(wr_valid), .vec_wr_req(wr_req), .vec_wr_wdata(wr_wdata), .vec_wr_wstrb(wr_wstrb), .vec_wr_grant(wr_grant),
    .perf_busy(perf_busy), .perf_sram_stall(perf_sram_stall));

  sram_stub u_sram (
    .clk(clk), .rst_n(rst_n), .deny_pct(deny_pct),
    .rd0_valid(rd0_valid), .rd0_req(rd0_req), .rd0_grant(rd0_grant), .rd0_rvalid(rd0_rvalid), .rd0_rdata(rd0_rdata),
    .rd1_valid(rd1_valid), .rd1_req(rd1_req), .rd1_grant(rd1_grant), .rd1_rvalid(rd1_rvalid), .rd1_rdata(rd1_rdata),
    .wr_valid(wr_valid), .wr_req(wr_req), .wr_wdata(wr_wdata), .wr_wstrb(wr_wstrb), .wr_grant(wr_grant));

  assign rd0_active = rd0_valid && rd0_grant;
  assign rd1_active = rd1_valid && rd1_grant;
  assign wr_active  = wr_valid && wr_grant;

  // Port-holding rule: an ungranted request must be held unchanged next cycle
  // (no reset needed: the engine presents no request while in reset).
  sram_req_t rd0_req_q, rd1_req_q, wr_req_q;
  logic rd0_held, rd1_held, wr_held;
  always_ff @(posedge clk) begin
    rd0_held <= rd0_valid && !rd0_grant; rd0_req_q <= rd0_req;
    rd1_held <= rd1_valid && !rd1_grant; rd1_req_q <= rd1_req;
    wr_held  <= wr_valid  && !wr_grant;  wr_req_q  <= wr_req;
    if (rd0_held && !(rd0_valid && rd0_req == rd0_req_q)) $fatal(1, "vec rd0 request dropped/changed while ungranted");
    if (rd1_held && !(rd1_valid && rd1_req == rd1_req_q)) $fatal(1, "vec rd1 request dropped/changed while ungranted");
    if (wr_held  && !(wr_valid  && wr_req  == wr_req_q))  $fatal(1, "vec wr request dropped/changed while ungranted");
  end
endmodule
