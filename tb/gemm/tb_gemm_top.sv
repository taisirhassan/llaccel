// tb_gemm_top.sv — gemm_engine + the real sram_xbar + 16 sram_bank (other xbar
// ports tied off). The C++ testbench loads / inspects the banks directly.
module tb_gemm_top
  import llaccel_pkg::*;
#(
  parameter bit EPILOGUE_FUSION = 1'b0
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic [511:0] instr_flat,
  input  logic         instr_valid,
  output logic         instr_ready,
  output logic         busy,
  output logic         done_pulse,
  output logic [7:0]   done_sig_sem,
  output logic         perf_mac_cycle,
  output logic         perf_epilogue_cycle,
  output logic [1:0]   perf_sram_stall,
  output logic         w_active,       // gemm_w request granted this cycle
  output logic         a_active,
  output logic         wr_active,
  output logic         fusion,         // elaborated EPILOGUE_FUSION, for the C++ side
  input  logic [6:0]   deny_pct,
  input  logic         dbg             // per-cycle trace of the engine's FSM and ports
);
  instr_words_t instr;
  assign instr = instr_flat;
  assign fusion = EPILOGUE_FUSION;

  always_ff @(posedge clk) begin
    if (dbg && rst_n)
      $display("st=%0d nt=%0d kt=%0d m=%0d | w v%0d g%0d rv%0d ld_act%0d infl%0d par%0d wbv%0d%0d | a v%0d g%0d rv%0d addr=%h sz=%0d tag=%0d fq=%0d/%0d | dr m=%0d d=%0d%0d%0d%0d fire%0d adv%0d | wr v%0d g%0d addr=%h",
               u_dut.state, u_dut.st_nt, u_dut.st_kt, u_dut.st_m,
               u_dut.gemm_w_valid, u_dut.gemm_w_grant, u_dut.gemm_w_rvalid, u_dut.ld_active, u_dut.ld_inflight, u_dut.ld_par,
               u_dut.wbuf_valid[0], u_dut.wbuf_valid[1],
               u_dut.gemm_a_valid, u_dut.gemm_a_grant, u_dut.gemm_a_rvalid, u_dut.gemm_a_req.addr, u_dut.gemm_a_req.size, u_dut.a_tag,
               u_dut.fq_kind, u_dut.fq_off,
               u_dut.dr_m, u_dut.d1_v, u_dut.d2_v, u_dut.d3_v, u_dut.d4_v, u_dut.d0_fire, u_dut.adv,
               u_dut.gemm_wr_valid, u_dut.gemm_wr_grant, u_dut.gemm_wr_req.addr);
  end

  logic          gemm_w_valid, gemm_w_grant, gemm_w_rvalid;
  sram_req_t     gemm_w_req;
  logic [2047:0] gemm_w_rdata;
  logic          gemm_a_valid, gemm_a_grant, gemm_a_rvalid;
  sram_req_t     gemm_a_req;
  logic [511:0]  gemm_a_rdata;
  logic          gemm_wr_valid, gemm_wr_grant;
  sram_req_t     gemm_wr_req;
  logic [511:0]  gemm_wr_wdata;
  logic [63:0]   gemm_wr_wstrb;
  logic          perf_busy;

  gemm_engine #(.EPILOGUE_FUSION(EPILOGUE_FUSION)) u_dut (
    .clk, .rst_n,
    .instr_valid, .instr, .instr_ready, .busy, .done_pulse, .done_sig_sem,
    .gemm_w_valid, .gemm_w_req, .gemm_w_grant, .gemm_w_rvalid, .gemm_w_rdata,
    .gemm_a_valid, .gemm_a_req, .gemm_a_grant, .gemm_a_rvalid, .gemm_a_rdata,
    .gemm_wr_valid, .gemm_wr_req, .gemm_wr_wdata, .gemm_wr_wstrb, .gemm_wr_grant,
    .perf_busy, .perf_mac_cycle, .perf_epilogue_cycle, .perf_sram_stall);

  // ---- crossbar + banks ------------------------------------------------------------
  logic                  bank_en    [NBANKS];
  logic [BANK_BYTES-1:0] bank_we    [NBANKS];
  logic [BANK_AW-1:0]    bank_addr  [NBANKS];
  logic [BANK_DW-1:0]    bank_wdata [NBANKS];
  logic [BANK_DW-1:0]    bank_rdata [NBANKS];
  logic [15:0] rd_bytes, wr_bytes;
  logic s0, s1, s2, s3, s4, s5, s6, s7, s8, s9;
  logic unused_rv [5];
  logic [511:0] unused_rd [5];

  logic [31:0] random_state;
  logic allow_w, allow_a, allow_wr;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) random_state <= 32'h12345678;
    else random_state <= {random_state[30:0], random_state[31] ^ random_state[21] ^ random_state[1] ^ random_state[0]};
  end
  assign allow_w = (32'(random_state[7:0]) * 32'd100) >= (32'(deny_pct) * 32'd256);
  assign allow_a = (32'(random_state[15:8]) * 32'd100) >= (32'(deny_pct) * 32'd256);
  assign allow_wr = (32'(random_state[23:16]) * 32'd100) >= (32'(deny_pct) * 32'd256);

  sram_xbar u_xbar (
    .clk, .rst_n,
    .gemm_w_valid(gemm_w_valid && allow_w), .gemm_w_req, .gemm_w_grant, .gemm_w_rvalid, .gemm_w_rdata,
    .gemm_a_valid(gemm_a_valid && allow_a), .gemm_a_req, .gemm_a_grant, .gemm_a_rvalid, .gemm_a_rdata,
    .gemm_wr_valid(gemm_wr_valid && allow_wr), .gemm_wr_req, .gemm_wr_wdata, .gemm_wr_wstrb, .gemm_wr_grant,
    .attn_rd_valid(1'b0), .attn_rd_req('0), .attn_rd_grant(), .attn_rd_rvalid(unused_rv[0]), .attn_rd_rdata(unused_rd[0]),
    .attn_wr_valid(1'b0), .attn_wr_req('0), .attn_wr_wdata('0), .attn_wr_wstrb('0), .attn_wr_grant(),
    .vec_rd0_valid(1'b0), .vec_rd0_req('0), .vec_rd0_grant(), .vec_rd0_rvalid(unused_rv[1]), .vec_rd0_rdata(unused_rd[1]),
    .vec_rd1_valid(1'b0), .vec_rd1_req('0), .vec_rd1_grant(), .vec_rd1_rvalid(unused_rv[2]), .vec_rd1_rdata(unused_rd[2]),
    .vec_wr_valid(1'b0), .vec_wr_req('0), .vec_wr_wdata('0), .vec_wr_wstrb('0), .vec_wr_grant(),
    .dma_wr_valid(1'b0), .dma_wr_req('0), .dma_wr_wdata('0), .dma_wr_wstrb('0), .dma_wr_grant(),
    .dma_rd_valid(1'b0), .dma_rd_req('0), .dma_rd_grant(), .dma_rd_rvalid(unused_rv[3]), .dma_rd_rdata(unused_rd[3]),
    .bank_en, .bank_we, .bank_addr, .bank_wdata, .bank_rdata,
    .gemm_w_stall(s0), .gemm_a_stall(s1), .gemm_wr_stall(s2), .attn_rd_stall(s3), .attn_wr_stall(s4),
    .vec_rd0_stall(s5), .vec_rd1_stall(s6), .vec_wr_stall(s7), .dma_wr_stall(s8), .dma_rd_stall(s9),
    .rd_bytes, .wr_bytes);

  for (genvar b = 0; b < NBANKS; b++) begin : g_bank
    sram_bank u_bank (.clk, .en(bank_en[b]), .we(bank_we[b]), .addr(bank_addr[b]), .wdata(bank_wdata[b]), .rdata(bank_rdata[b]));
  end

  assign w_active  = gemm_w_valid && gemm_w_grant;
  assign a_active  = gemm_a_valid && gemm_a_grant;
  assign wr_active = gemm_wr_valid && gemm_wr_grant;

  // Port-holding rule: an ungranted request must be held unchanged next cycle.
  sram_req_t w_q, a_q, wr_q;
  logic w_held, a_held, wr_held;
  always_ff @(posedge clk) begin
    w_held  <= rst_n && gemm_w_valid  && !gemm_w_grant;  w_q  <= gemm_w_req;
    a_held  <= rst_n && gemm_a_valid  && !gemm_a_grant;  a_q  <= gemm_a_req;
    wr_held <= rst_n && gemm_wr_valid && !gemm_wr_grant; wr_q <= gemm_wr_req;
    if (w_held  && !(gemm_w_valid  && gemm_w_req  == w_q))  $fatal(1, "gemm_w request dropped/changed while ungranted");
    if (a_held  && !(gemm_a_valid  && gemm_a_req  == a_q))  $fatal(1, "gemm_a request dropped/changed while ungranted");
    if (wr_held && !(gemm_wr_valid && gemm_wr_req == wr_q)) $fatal(1, "gemm_wr request dropped/changed while ungranted");
  end

  logic unused_ok;
  always_comb begin
    unused_ok = perf_busy | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | (|rd_bytes) | (|wr_bytes);
    for (int i = 0; i < 4; i++) unused_ok = unused_ok | unused_rv[i] | (|unused_rd[i]);
  end
endmodule
