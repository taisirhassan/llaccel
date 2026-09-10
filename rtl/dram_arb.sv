// dram_arb.sv — two requesters onto the single DRAM request port.
// dma (high priority, reads and writes) vs cmd_proc instruction fetch (low
// priority, reads only). Read responses return in request order, so a FIFO of
// owner tags routes each response back to the requester that issued it.
module dram_arb
  import llaccel_pkg::*;
#(
  parameter int unsigned TAG_DEPTH = 64   // >= DRAM max outstanding reads (32)
) (
  input  logic               clk,
  input  logic               rst_n,
  // dma requester
  input  logic               dma_req_valid,
  input  logic               dma_req_we,
  input  logic [31:0]        dma_req_addr,
  input  logic [DRAM_DW-1:0] dma_req_wdata,
  input  logic [DRAM_BEAT-1:0] dma_req_wstrb,
  output logic               dma_req_ready,
  output logic               dma_rsp_valid,
  // cmd_proc fetch requester (read only)
  input  logic               cp_req_valid,
  input  logic [31:0]        cp_req_addr,
  output logic               cp_req_ready,
  output logic               cp_rsp_valid,
  // shared response data
  output logic [DRAM_DW-1:0] rsp_rdata,
  // DRAM port
  output logic               dram_req_valid,
  input  logic               dram_req_ready,
  output logic               dram_req_we,
  output logic [31:0]        dram_req_addr,
  output logic [DRAM_DW-1:0] dram_req_wdata,
  output logic [DRAM_BEAT-1:0] dram_req_wstrb,
  input  logic               dram_rsp_valid,
  input  logic [DRAM_DW-1:0] dram_rsp_rdata
);
  logic tag_full, tag_empty, tag_head;
  logic tag_push, tag_pop;
  logic sel_dma;
  logic [$clog2(TAG_DEPTH+1)-1:0] tag_count;

  assign sel_dma        = dma_req_valid;
  assign dram_req_valid = (dma_req_valid || cp_req_valid) && !tag_full;
  assign dram_req_we    = sel_dma ? dma_req_we : 1'b0;
  assign dram_req_addr  = sel_dma ? dma_req_addr : cp_req_addr;
  assign dram_req_wdata = dma_req_wdata;
  assign dram_req_wstrb = sel_dma ? dma_req_wstrb : '0;
  assign dma_req_ready  = dram_req_ready && !tag_full;
  assign cp_req_ready   = dram_req_ready && !tag_full && !dma_req_valid;

  // A read is accepted this cycle -> remember its owner (1 = dma, 0 = cp).
  assign tag_push = dram_req_valid && dram_req_ready && !dram_req_we;
  assign tag_pop  = dram_rsp_valid;

  sync_fifo #(.WIDTH(1), .DEPTH(TAG_DEPTH)) u_tags (
    .clk, .rst_n, .clr(1'b0),
    .push(tag_push), .wdata(sel_dma),
    .pop(tag_pop), .rdata(tag_head),
    .full(tag_full), .empty(tag_empty), .count(tag_count)
  );

  assign rsp_rdata     = dram_rsp_rdata;
  assign dma_rsp_valid = dram_rsp_valid && !tag_empty && tag_head;
  assign cp_rsp_valid  = dram_rsp_valid && !tag_empty && !tag_head;

  logic unused_ok;
  assign unused_ok = &{1'b0, tag_count};
endmodule
