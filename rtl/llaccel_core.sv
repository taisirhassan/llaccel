// llaccel_core.sv — the synthesized boundary: command processor, four engines,
// SRAM crossbar, DRAM arbiter and performance counters. SRAM bank storage is
// outside (bank ports); llaccel_top adds the 16 behavioral banks.
module llaccel_core
  import llaccel_pkg::*;
#(
  parameter bit EPILOGUE_FUSION = 1'b0,
  parameter int unsigned DMA_MAX_OUTSTANDING = 16
) (
  input  logic               clk,
  input  logic               rst_n,
  input  logic               start,
  input  logic [31:0]        pc_start,
  input  logic [31:0]        pos,
  output logic               done,
  // DRAM
  output logic               dram_req_valid,
  input  logic               dram_req_ready,
  output logic               dram_req_we,
  output logic [31:0]        dram_req_addr,
  output logic [DRAM_DW-1:0] dram_req_wdata,
  output logic [DRAM_BEAT-1:0] dram_req_wstrb,
  input  logic               dram_rsp_valid,
  input  logic [DRAM_DW-1:0] dram_rsp_rdata,
  // perf
  output logic [63:0]        perf [NUM_PERF],
  // SRAM banks
  output logic               bank_en    [NBANKS],
  output logic [BANK_BYTES-1:0] bank_we  [NBANKS],
  output logic [BANK_AW-1:0] bank_addr  [NBANKS],
  output logic [BANK_DW-1:0] bank_wdata [NBANKS],
  input  logic [BANK_DW-1:0] bank_rdata [NBANKS]
);
  // ---- cmd_proc <-> engines ---------------------------------------------------------
  logic         cp_running;
  logic         fetch_req_valid, fetch_req_ready, fetch_rsp_valid;
  logic [31:0]  fetch_req_addr;
  logic [DRAM_DW-1:0] arb_rsp_rdata;

  logic         dma_iv, dma_ir, dma_busy, dma_done;
  logic [7:0]   dma_sig;
  instr_words_t dma_instr;
  logic         gemm_iv, gemm_ir, gemm_busy, gemm_done;
  logic [7:0]   gemm_sig;
  instr_words_t gemm_instr;
  logic         vec_iv, vec_ir, vec_busy, vec_done;
  logic [7:0]   vec_sig;
  instr_words_t vec_instr;
  logic         attn_iv, attn_ir, attn_busy, attn_done;
  logic [7:0]   attn_sig;
  instr_words_t attn_instr;

  logic cp_instr_issued, cp_stall_wait, cp_stall_qfull, cp_stall_fetch;
  logic dma_q_empty, gemm_q_empty, vec_q_empty, attn_q_empty;

  // ---- DMA <-> dram_arb ---------------------------------------------------------------
  logic         dma_dram_req_valid, dma_dram_req_we, dma_dram_req_ready, dma_dram_rsp_valid;
  logic [31:0]  dma_dram_req_addr;
  logic [DRAM_DW-1:0] dma_dram_req_wdata;
  logic [DRAM_BEAT-1:0] dma_dram_req_wstrb;

  logic attn_dram_req_valid, attn_dram_req_we, attn_dram_req_ready, attn_dram_rsp_valid;
  logic [31:0] attn_dram_req_addr;
  logic [511:0] attn_dram_req_wdata;
  logic [63:0] attn_dram_req_wstrb;

  // ---- SRAM ports -------------------------------------------------------------------------
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
  logic          attn_rd_valid, attn_rd_grant, attn_rd_rvalid;
  sram_req_t     attn_rd_req;
  logic [511:0]  attn_rd_rdata;
  logic          attn_wr_valid, attn_wr_grant;
  sram_req_t     attn_wr_req;
  logic [511:0]  attn_wr_wdata;
  logic [63:0]   attn_wr_wstrb;
  logic          vec_rd0_valid, vec_rd0_grant, vec_rd0_rvalid;
  sram_req_t     vec_rd0_req;
  logic [511:0]  vec_rd0_rdata;
  logic          vec_rd1_valid, vec_rd1_grant, vec_rd1_rvalid;
  sram_req_t     vec_rd1_req;
  logic [511:0]  vec_rd1_rdata;
  logic          vec_wr_valid, vec_wr_grant;
  sram_req_t     vec_wr_req;
  logic [511:0]  vec_wr_wdata;
  logic [63:0]   vec_wr_wstrb;
  logic          dma_wr_valid, dma_wr_grant;
  sram_req_t     dma_wr_req;
  logic [511:0]  dma_wr_wdata;
  logic [63:0]   dma_wr_wstrb;
  logic          dma_rd_valid, dma_rd_grant, dma_rd_rvalid;
  sram_req_t     dma_rd_req;
  logic [511:0]  dma_rd_rdata;

  logic gemm_w_stall, gemm_a_stall, gemm_wr_stall, attn_rd_stall, attn_wr_stall;
  logic vec_rd0_stall, vec_rd1_stall, vec_wr_stall, dma_wr_stall, dma_rd_stall;
  logic [15:0] sram_rd_bytes, sram_wr_bytes;

  // ---- engine perf pulses ---------------------------------------------------------------------
  logic gemm_p_busy, gemm_p_mac, gemm_p_ep;
  logic [1:0] gemm_p_stall;
  logic dma_p_busy, dma_p_stall, dma_p_wait;
  logic vec_p_busy, vec_p_stall;
  logic attn_p_busy, attn_p_stall, attn_p_mac, attn_p_dram_wait;

  // =============================================================================
  cmd_proc u_cp (
    .clk, .rst_n, .start, .pc_start, .done, .running(cp_running),
    .fetch_req_valid, .fetch_req_addr, .fetch_req_ready, .fetch_rsp_valid, .fetch_rsp_rdata(arb_rsp_rdata),
    .dma_instr_valid(dma_iv), .dma_instr, .dma_instr_ready(dma_ir), .dma_busy, .dma_done_pulse(dma_done), .dma_done_sig_sem(dma_sig),
    .gemm_instr_valid(gemm_iv), .gemm_instr, .gemm_instr_ready(gemm_ir), .gemm_busy, .gemm_done_pulse(gemm_done), .gemm_done_sig_sem(gemm_sig),
    .vec_instr_valid(vec_iv), .vec_instr, .vec_instr_ready(vec_ir), .vec_busy, .vec_done_pulse(vec_done), .vec_done_sig_sem(vec_sig),
    .attn_instr_valid(attn_iv), .attn_instr, .attn_instr_ready(attn_ir), .attn_busy, .attn_done_pulse(attn_done), .attn_done_sig_sem(attn_sig),
    .perf_instr_issued(cp_instr_issued), .perf_stall_wait(cp_stall_wait), .perf_stall_qfull(cp_stall_qfull), .perf_stall_fetch(cp_stall_fetch),
    .dma_q_empty, .gemm_q_empty, .vec_q_empty, .attn_q_empty);

  dram_arb u_arb (
    .clk, .rst_n,
    .dma_req_valid(dma_dram_req_valid), .dma_req_we(dma_dram_req_we), .dma_req_addr(dma_dram_req_addr),
    .dma_req_wdata(dma_dram_req_wdata), .dma_req_wstrb(dma_dram_req_wstrb), .dma_req_ready(dma_dram_req_ready),
    .dma_rsp_valid(dma_dram_rsp_valid),
    .attn_req_valid(attn_dram_req_valid),.attn_req_we(attn_dram_req_we),.attn_req_addr(attn_dram_req_addr),
    .attn_req_wdata(attn_dram_req_wdata),.attn_req_wstrb(attn_dram_req_wstrb),
    .attn_req_ready(attn_dram_req_ready),.attn_rsp_valid(attn_dram_rsp_valid),
    .cp_req_valid(fetch_req_valid), .cp_req_addr(fetch_req_addr), .cp_req_ready(fetch_req_ready), .cp_rsp_valid(fetch_rsp_valid),
    .rsp_rdata(arb_rsp_rdata),
    .dram_req_valid, .dram_req_ready, .dram_req_we, .dram_req_addr, .dram_req_wdata, .dram_req_wstrb,
    .dram_rsp_valid, .dram_rsp_rdata);

  dma_engine #(.MAX_OUTSTANDING(DMA_MAX_OUTSTANDING)) u_dma (
    .clk, .rst_n,
    .instr_valid(dma_iv), .instr(dma_instr), .instr_ready(dma_ir), .busy(dma_busy), .done_pulse(dma_done), .done_sig_sem(dma_sig),
    .dram_req_valid(dma_dram_req_valid), .dram_req_we(dma_dram_req_we), .dram_req_addr(dma_dram_req_addr),
    .dram_req_wdata(dma_dram_req_wdata), .dram_req_wstrb(dma_dram_req_wstrb), .dram_req_ready(dma_dram_req_ready),
    .dram_rsp_valid(dma_dram_rsp_valid), .dram_rsp_rdata(arb_rsp_rdata),
    .dma_wr_valid, .dma_wr_req, .dma_wr_wdata, .dma_wr_wstrb, .dma_wr_grant,
    .dma_rd_valid, .dma_rd_req, .dma_rd_grant, .dma_rd_rvalid, .dma_rd_rdata,
    .perf_busy(dma_p_busy), .perf_sram_stall(dma_p_stall), .perf_dram_wait(dma_p_wait));

  gemm_engine #(.EPILOGUE_FUSION(EPILOGUE_FUSION)) u_gemm (
    .clk, .rst_n,
    .instr_valid(gemm_iv), .instr(gemm_instr), .instr_ready(gemm_ir), .busy(gemm_busy), .done_pulse(gemm_done), .done_sig_sem(gemm_sig),
    .gemm_w_valid, .gemm_w_req, .gemm_w_grant, .gemm_w_rvalid, .gemm_w_rdata,
    .gemm_a_valid, .gemm_a_req, .gemm_a_grant, .gemm_a_rvalid, .gemm_a_rdata,
    .gemm_wr_valid, .gemm_wr_req, .gemm_wr_wdata, .gemm_wr_wstrb, .gemm_wr_grant,
    .perf_busy(gemm_p_busy), .perf_mac_cycle(gemm_p_mac), .perf_epilogue_cycle(gemm_p_ep), .perf_sram_stall(gemm_p_stall));

  vec_engine u_vec (
    .clk, .rst_n,
    .instr_valid(vec_iv), .instr(vec_instr), .instr_ready(vec_ir), .pos, .busy(vec_busy), .done_pulse(vec_done), .done_sig_sem(vec_sig),
    .vec_rd0_valid, .vec_rd0_req, .vec_rd0_grant, .vec_rd0_rvalid, .vec_rd0_rdata,
    .vec_rd1_valid, .vec_rd1_req, .vec_rd1_grant, .vec_rd1_rvalid, .vec_rd1_rdata,
    .vec_wr_valid, .vec_wr_req, .vec_wr_wdata, .vec_wr_wstrb, .vec_wr_grant,
    .perf_busy(vec_p_busy), .perf_sram_stall(vec_p_stall));

  attn_engine u_attn (
    .clk, .rst_n,
    .instr_valid(attn_iv), .instr(attn_instr), .instr_ready(attn_ir), .pos, .busy(attn_busy), .done_pulse(attn_done), .done_sig_sem(attn_sig),
    .attn_rd_valid, .attn_rd_req, .attn_rd_grant, .attn_rd_rvalid, .attn_rd_rdata,
    .attn_wr_valid, .attn_wr_req, .attn_wr_wdata, .attn_wr_wstrb, .attn_wr_grant,
    .dram_req_valid(attn_dram_req_valid),.dram_req_we(attn_dram_req_we),.dram_req_addr(attn_dram_req_addr),
    .dram_req_wdata(attn_dram_req_wdata),.dram_req_wstrb(attn_dram_req_wstrb),.dram_req_ready(attn_dram_req_ready),
    .dram_rsp_valid(attn_dram_rsp_valid),.dram_rsp_rdata(arb_rsp_rdata),
    .perf_dram_wait(attn_p_dram_wait), .perf_busy(attn_p_busy), .perf_sram_stall(attn_p_stall), .perf_mac_cycles(attn_p_mac));

  sram_xbar u_xbar (
    .clk, .rst_n,
    .gemm_w_valid, .gemm_w_req, .gemm_w_grant, .gemm_w_rvalid, .gemm_w_rdata,
    .gemm_a_valid, .gemm_a_req, .gemm_a_grant, .gemm_a_rvalid, .gemm_a_rdata,
    .gemm_wr_valid, .gemm_wr_req, .gemm_wr_wdata, .gemm_wr_wstrb, .gemm_wr_grant,
    .attn_rd_valid, .attn_rd_req, .attn_rd_grant, .attn_rd_rvalid, .attn_rd_rdata,
    .attn_wr_valid, .attn_wr_req, .attn_wr_wdata, .attn_wr_wstrb, .attn_wr_grant,
    .vec_rd0_valid, .vec_rd0_req, .vec_rd0_grant, .vec_rd0_rvalid, .vec_rd0_rdata,
    .vec_rd1_valid, .vec_rd1_req, .vec_rd1_grant, .vec_rd1_rvalid, .vec_rd1_rdata,
    .vec_wr_valid, .vec_wr_req, .vec_wr_wdata, .vec_wr_wstrb, .vec_wr_grant,
    .dma_wr_valid, .dma_wr_req, .dma_wr_wdata, .dma_wr_wstrb, .dma_wr_grant,
    .dma_rd_valid, .dma_rd_req, .dma_rd_grant, .dma_rd_rvalid, .dma_rd_rdata,
    .bank_en, .bank_we, .bank_addr, .bank_wdata, .bank_rdata,
    .gemm_w_stall, .gemm_a_stall, .gemm_wr_stall, .attn_rd_stall, .attn_wr_stall,
    .vec_rd0_stall, .vec_rd1_stall, .vec_wr_stall, .dma_wr_stall, .dma_rd_stall,
    .rd_bytes(sram_rd_bytes), .wr_bytes(sram_wr_bytes));

  // =============================================================================
  // perf counters: per-cycle increments per index (docs/ARCH.md)
  // =============================================================================
  logic [15:0] inc [NUM_PERF];
  logic dram_rd_accept, dram_wr_accept;
  assign dram_rd_accept = dram_req_valid && dram_req_ready && !dram_req_we;
  assign dram_wr_accept = dram_req_valid && dram_req_ready &&  dram_req_we;

  always_comb begin
    for (int unsigned i = 0; i < NUM_PERF; i++) inc[i] = '0;
    inc[PERF_CYCLES]               = {15'd0, cp_running};
    inc[PERF_INSTR_ISSUED]         = {15'd0, cp_instr_issued};
    inc[PERF_CP_STALL_WAIT]        = {15'd0, cp_stall_wait};
    inc[PERF_CP_STALL_QFULL]       = {15'd0, cp_stall_qfull};
    inc[PERF_CP_STALL_FETCH]       = {15'd0, cp_stall_fetch};
    inc[PERF_GEMM_BUSY]            = {15'd0, gemm_p_busy};
    inc[PERF_GEMM_MAC_CYCLES]      = {15'd0, gemm_p_mac};
    inc[PERF_GEMM_SRAM_STALL]      = {14'd0, gemm_p_stall};
    inc[PERF_GEMM_EPILOGUE_CYCLES] = {15'd0, gemm_p_ep};
    inc[PERF_VEC_BUSY]             = {15'd0, vec_p_busy};
    inc[PERF_VEC_SRAM_STALL]       = {15'd0, vec_p_stall};
    inc[PERF_ATTN_BUSY]            = {15'd0, attn_p_busy};
    inc[PERF_ATTN_SRAM_STALL]      = {15'd0, attn_p_stall};
    inc[PERF_ATTN_MAC_CYCLES]      = {15'd0, attn_p_mac};
    inc[PERF_DMA_BUSY]             = {15'd0, dma_p_busy};
    inc[PERF_DMA_SRAM_STALL]       = {15'd0, dma_p_stall};
    inc[PERF_DMA_DRAM_WAIT]        = {15'd0, dma_p_wait};
    inc[PERF_SRAM_RD_BYTES]        = sram_rd_bytes;
    inc[PERF_SRAM_WR_BYTES]        = sram_wr_bytes;
    inc[PERF_DRAM_RD_BYTES]        = dram_rd_accept ? 16'(DRAM_BEAT) : 16'd0;
    inc[PERF_DRAM_WR_BYTES]        = dram_wr_accept ? 16'(DRAM_BEAT) : 16'd0;
    inc[PERF_GEMM_IDLE_QEMPTY]     = {15'd0, cp_running && !gemm_busy && gemm_q_empty};
    inc[PERF_VEC_IDLE_QEMPTY]      = {15'd0, cp_running && !vec_busy  && vec_q_empty};
    inc[PERF_ATTN_DRAM_WAIT] = {15'd0, attn_p_dram_wait};
    inc[PERF_ATTN_DRAM_RD_BYTES] = (attn_dram_req_valid && attn_dram_req_ready && !attn_dram_req_we) ? 16'(DRAM_BEAT) : 16'd0;
    inc[PERF_ATTN_DRAM_WR_BYTES] = (attn_dram_req_valid && attn_dram_req_ready && attn_dram_req_we) ? 16'(DRAM_BEAT) : 16'd0;
    inc[PERF_ATTN_IDLE_QEMPTY]     = {15'd0, cp_running && !attn_busy && attn_q_empty};
  end

  perf_counters u_perf (.clk, .rst_n, .clear(start), .inc, .perf);

  logic unused_ok;
  assign unused_ok = &{1'b0, gemm_w_stall, gemm_a_stall, gemm_wr_stall, attn_rd_stall, attn_wr_stall,
                       vec_rd0_stall, vec_rd1_stall, vec_wr_stall, dma_wr_stall, dma_rd_stall, dma_q_empty};
endmodule
