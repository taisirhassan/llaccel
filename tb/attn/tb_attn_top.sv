// tb_attn_top.sv — Verilator wrapper: attn_engine + behavioral SRAM stub.
// The stub's rd0 port carries attn_rd and its wr port attn_wr (rd1 is idle),
// so the stub's rd0 > wr bank-conflict priority models the crossbar's
// attn_rd > attn_wr ordering.
module tb_attn_top (
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
  output logic dram_req_valid,
  output logic dram_req_we,
  output logic [31:0] dram_req_addr,
  output logic [511:0] dram_req_wdata,
  output logic [63:0] dram_req_wstrb,
  input logic dram_req_ready,
  input logic dram_rsp_valid,
  input logic [511:0] dram_rsp_rdata,
  output logic         perf_busy,
  output logic         perf_sram_stall,
  output logic         perf_mac_cycles,
  output logic         rd_active,      // request/grant activity, for cycle accounting
  output logic         wr_active
);
  import llaccel_pkg::*;

  instr_words_t instr;
  assign instr = instr_flat;

  logic rd_valid, rd_grant, rd_rvalid;
  logic wr_valid, wr_grant;
  logic rd1_grant, rd1_rvalid;
  sram_req_t rd_req, wr_req;
  logic [511:0] rd_rdata, rd1_rdata, wr_wdata;
  logic [63:0] wr_wstrb;

  attn_engine u_dut (
    .clk(clk), .rst_n(rst_n),
    .instr_valid(instr_valid), .instr(instr), .instr_ready(instr_ready), .pos(pos),
    .busy(busy), .done_pulse(done_pulse), .done_sig_sem(done_sig_sem),
    .attn_rd_valid(rd_valid), .attn_rd_req(rd_req), .attn_rd_grant(rd_grant), .attn_rd_rvalid(rd_rvalid), .attn_rd_rdata(rd_rdata),
    .attn_wr_valid(wr_valid), .attn_wr_req(wr_req), .attn_wr_wdata(wr_wdata), .attn_wr_wstrb(wr_wstrb), .attn_wr_grant(wr_grant),
    .dram_req_valid,.dram_req_we,.dram_req_addr,.dram_req_wdata,.dram_req_wstrb,
    .dram_req_ready,.dram_rsp_valid,.dram_rsp_rdata,
    .perf_dram_wait(), .perf_busy(perf_busy), .perf_sram_stall(perf_sram_stall), .perf_mac_cycles(perf_mac_cycles));

  sram_stub u_sram (
    .clk(clk), .rst_n(rst_n), .deny_pct(deny_pct),
    .rd0_valid(rd_valid), .rd0_req(rd_req), .rd0_grant(rd_grant), .rd0_rvalid(rd_rvalid), .rd0_rdata(rd_rdata),
    .rd1_valid(1'b0), .rd1_req('{addr: '0, size: SZ_16}), .rd1_grant(rd1_grant), .rd1_rvalid(rd1_rvalid), .rd1_rdata(rd1_rdata),
    .wr_valid(wr_valid), .wr_req(wr_req), .wr_wdata(wr_wdata), .wr_wstrb(wr_wstrb), .wr_grant(wr_grant));

  assign rd_active = rd_valid && rd_grant;
  assign wr_active = wr_valid && wr_grant;

  // Port-holding rule: an ungranted request must be held unchanged next cycle
  // (no reset needed: the engine presents no request while in reset).
  sram_req_t rd_req_q, wr_req_q;
  logic [511:0] wr_wdata_q;
  logic [63:0] wr_wstrb_q;
  logic rd_held, wr_held;
  always_ff @(posedge clk) begin
    rd_held <= rd_valid && !rd_grant; rd_req_q <= rd_req;
    wr_held <= wr_valid && !wr_grant; wr_req_q <= wr_req; wr_wdata_q <= wr_wdata; wr_wstrb_q <= wr_wstrb;
    if (rd_held && !(rd_valid && rd_req == rd_req_q)) $fatal(1, "attn rd request dropped/changed while ungranted");
    if (wr_held && !(wr_valid && wr_req == wr_req_q && wr_wdata == wr_wdata_q && wr_wstrb == wr_wstrb_q))
      $fatal(1, "attn wr request dropped/changed while ungranted");
  end

  logic unused_ok;
  assign unused_ok = rd1_grant | rd1_rvalid | (|rd1_rdata);
endmodule
