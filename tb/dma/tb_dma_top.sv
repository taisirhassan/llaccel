// tb_dma_top.sv — dma_engine + the real sram_xbar + 16 sram_bank; the DRAM
// ports come out to the C++ testbench (llaccel::DramModel). A random "hog"
// requester on the higher-priority attn_rd port (reads only, never writes)
// steals banks with probability hog_pct/100 per cycle so the DMA's SRAM
// requests are denied and must be held.
module tb_dma_top
  import llaccel_pkg::*;
#(parameter int unsigned MAX_OUTSTANDING = 16)
(
  input  logic               clk,
  input  logic               rst_n,
  input  logic [511:0]       instr_flat,
  input  logic               instr_valid,
  output logic               instr_ready,
  output logic               busy,
  output logic               done_pulse,
  output logic [7:0]         done_sig_sem,
  input  logic [7:0]         hog_pct,
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
  output logic               perf_busy,
  output logic               perf_sram_stall,
  output logic               perf_dram_wait,
  output logic               wr_active,      // dma_wr granted this cycle
  output logic               rd_active       // dma_rd granted this cycle
);
  instr_words_t instr;
  assign instr = instr_flat;

  logic          dma_wr_valid, dma_wr_grant;
  sram_req_t     dma_wr_req;
  logic [511:0]  dma_wr_wdata;
  logic [63:0]   dma_wr_wstrb;
  logic          dma_rd_valid, dma_rd_grant, dma_rd_rvalid;
  sram_req_t     dma_rd_req;
  logic [511:0]  dma_rd_rdata;

  dma_engine #(.MAX_OUTSTANDING(MAX_OUTSTANDING)) u_dut (
    .clk, .rst_n,
    .instr_valid, .instr, .instr_ready, .busy, .done_pulse, .done_sig_sem,
    .dram_req_valid, .dram_req_we, .dram_req_addr, .dram_req_wdata, .dram_req_wstrb, .dram_req_ready,
    .dram_rsp_valid, .dram_rsp_rdata,
    .dma_wr_valid, .dma_wr_req, .dma_wr_wdata, .dma_wr_wstrb, .dma_wr_grant,
    .dma_rd_valid, .dma_rd_req, .dma_rd_grant, .dma_rd_rvalid, .dma_rd_rdata,
    .perf_busy, .perf_sram_stall, .perf_dram_wait);

  // ---- bank hog on attn_rd (higher priority than both dma ports) ----
  logic      hog_valid;
  sram_req_t hog_req;
  logic      hog_rvalid;
  logic [511:0] hog_rdata;
  always_ff @(negedge clk) begin
    hog_valid <= rst_n && (32'($urandom % 100) < 32'(hog_pct));
    hog_req   <= '{addr: {$urandom % 4096, 2'($urandom), 6'd0}, size: SZ_64};   // random 64-B aligned block
  end

  // ---- crossbar + banks ------------------------------------------------------------
  logic                  bank_en    [NBANKS];
  logic [BANK_BYTES-1:0] bank_we    [NBANKS];
  logic [BANK_AW-1:0]    bank_addr  [NBANKS];
  logic [BANK_DW-1:0]    bank_wdata [NBANKS];
  logic [BANK_DW-1:0]    bank_rdata [NBANKS];
  logic [15:0] rd_bytes, wr_bytes;
  logic s0, s1, s2, s3, s4, s5, s6, s7, s8, s9;
  logic unused_rv [4];
  logic [511:0] unused_rd [4];
  logic [2047:0] unused_wrd;

  sram_xbar u_xbar (
    .clk, .rst_n,
    .gemm_w_valid(1'b0), .gemm_w_req('0), .gemm_w_grant(), .gemm_w_rvalid(unused_rv[0]), .gemm_w_rdata(unused_wrd),
    .gemm_a_valid(1'b0), .gemm_a_req('0), .gemm_a_grant(), .gemm_a_rvalid(unused_rv[1]), .gemm_a_rdata(unused_rd[0]),
    .gemm_wr_valid(1'b0), .gemm_wr_req('0), .gemm_wr_wdata('0), .gemm_wr_wstrb('0), .gemm_wr_grant(),
    .attn_rd_valid(hog_valid), .attn_rd_req(hog_req), .attn_rd_grant(), .attn_rd_rvalid(hog_rvalid), .attn_rd_rdata(hog_rdata),
    .attn_wr_valid(1'b0), .attn_wr_req('0), .attn_wr_wdata('0), .attn_wr_wstrb('0), .attn_wr_grant(),
    .vec_rd0_valid(1'b0), .vec_rd0_req('0), .vec_rd0_grant(), .vec_rd0_rvalid(unused_rv[2]), .vec_rd0_rdata(unused_rd[1]),
    .vec_rd1_valid(1'b0), .vec_rd1_req('0), .vec_rd1_grant(), .vec_rd1_rvalid(unused_rv[3]), .vec_rd1_rdata(unused_rd[2]),
    .vec_wr_valid(1'b0), .vec_wr_req('0), .vec_wr_wdata('0), .vec_wr_wstrb('0), .vec_wr_grant(),
    .dma_wr_valid, .dma_wr_req, .dma_wr_wdata, .dma_wr_wstrb, .dma_wr_grant,
    .dma_rd_valid, .dma_rd_req, .dma_rd_grant, .dma_rd_rvalid, .dma_rd_rdata,
    .bank_en, .bank_we, .bank_addr, .bank_wdata, .bank_rdata,
    .gemm_w_stall(s0), .gemm_a_stall(s1), .gemm_wr_stall(s2), .attn_rd_stall(s3), .attn_wr_stall(s4),
    .vec_rd0_stall(s5), .vec_rd1_stall(s6), .vec_wr_stall(s7), .dma_wr_stall(s8), .dma_rd_stall(s9),
    .rd_bytes, .wr_bytes);

  for (genvar b = 0; b < NBANKS; b++) begin : g_bank
    sram_bank u_bank (.clk, .en(bank_en[b]), .we(bank_we[b]), .addr(bank_addr[b]), .wdata(bank_wdata[b]), .rdata(bank_rdata[b]));
  end

  assign wr_active = dma_wr_valid && dma_wr_grant;
  assign rd_active = dma_rd_valid && dma_rd_grant;

  // Port-holding rule for the two dma ports.
  sram_req_t wr_q, rd_q;
  logic wr_held, rd_held;
  always_ff @(posedge clk) begin
    wr_held <= rst_n && dma_wr_valid && !dma_wr_grant; wr_q <= dma_wr_req;
    rd_held <= rst_n && dma_rd_valid && !dma_rd_grant; rd_q <= dma_rd_req;
    if (wr_held && !(dma_wr_valid && dma_wr_req == wr_q)) $fatal(1, "dma_wr request dropped/changed while ungranted");
    if (rd_held && !(dma_rd_valid && dma_rd_req == rd_q)) $fatal(1, "dma_rd request dropped/changed while ungranted");
  end

  logic unused_ok;
  always_comb begin
    unused_ok = hog_rvalid | (|hog_rdata) | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | (|rd_bytes) | (|wr_bytes) |
                (|unused_wrd) | (|unused_rd[3]);
    for (int i = 0; i < 4; i++) unused_ok = unused_ok | unused_rv[i];
    for (int i = 0; i < 3; i++) unused_ok = unused_ok | (|unused_rd[i]);
  end
endmodule
