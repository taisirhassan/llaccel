// dram_arb.sv — DMA, command fetch and attention share one DRAM request port
// using round-robin arbitration. Read responses return in request order, so a FIFO of
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
  input logic attn_req_valid,
  input logic attn_req_we,
  input logic [31:0] attn_req_addr,
  input logic [DRAM_DW-1:0] attn_req_wdata,
  input logic [DRAM_BEAT-1:0] attn_req_wstrb,
  output logic attn_req_ready,
  output logic attn_rsp_valid,
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
  logic tag_full, tag_empty;
  logic [1:0] tag_head, selected, last_owner, held_owner;
  logic tag_push, tag_pop, holding;
  logic [$clog2(TAG_DEPTH+1)-1:0] tag_count;
  // Round robin among fetch, DMA and attention. Retain the selected owner
  // under external backpressure so address/data stay stable until acceptance.
  always_comb begin
    selected = 0;
    case (last_owner)
      0: if (dma_req_valid) selected=1; else if(attn_req_valid) selected=2;
      1: if (attn_req_valid) selected=2; else if(cp_req_valid) selected=0; else selected=1;
      default: if(cp_req_valid) selected=0; else if(dma_req_valid) selected=1; else selected=2;
    endcase
    if (holding) selected=held_owner;
  end
  assign dram_req_valid = (dma_req_valid || cp_req_valid || attn_req_valid) && !tag_full;
  assign dram_req_we = selected==1 ? dma_req_we : selected==2 ? attn_req_we : 1'b0;
  assign dram_req_addr = selected==1 ? dma_req_addr : selected==2 ? attn_req_addr : cp_req_addr;
  assign dram_req_wdata = selected==2 ? attn_req_wdata : dma_req_wdata;
  assign dram_req_wstrb = selected==1 ? dma_req_wstrb : selected==2 ? attn_req_wstrb : '0;
  assign dma_req_ready = dram_req_ready && !tag_full && selected==1;
  assign attn_req_ready = dram_req_ready && !tag_full && selected==2;
  assign cp_req_ready = dram_req_ready && !tag_full && selected==0;
  assign tag_push = dram_req_valid && dram_req_ready && !dram_req_we;
  assign tag_pop = dram_rsp_valid;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin last_owner<=2; holding<=0; held_owner<=0; end
    else begin
      if (dram_req_valid && !dram_req_ready) begin holding<=1; held_owner<=selected; end
      if (dram_req_valid && dram_req_ready) begin holding<=0; last_owner<=selected; end
    end
  end
  sync_fifo #(.WIDTH(2), .DEPTH(TAG_DEPTH)) u_tags (
    .clk,.rst_n,.clr(1'b0),.push(tag_push),.wdata(selected),.pop(tag_pop),.rdata(tag_head),
    .full(tag_full),.empty(tag_empty),.count(tag_count));
  assign rsp_rdata = dram_rsp_rdata;
  assign cp_rsp_valid = dram_rsp_valid && !tag_empty && tag_head==0;
  assign dma_rsp_valid = dram_rsp_valid && !tag_empty && tag_head==1;
  assign attn_rsp_valid = dram_rsp_valid && !tag_empty && tag_head==2;
  logic unused_ok;
  assign unused_ok = &{1'b0,tag_count};
endmodule
