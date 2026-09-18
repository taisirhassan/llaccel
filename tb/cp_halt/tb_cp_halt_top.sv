module tb_cp_halt_top
 import llaccel_pkg::*;
(
 input logic clk,rst_n,start,engine_busy,dma_req_valid,dram_ready,dram_rsp,
 input logic [511:0] response_data,
 output logic req_valid,req_we,cp_valid,cp_ready,cp_rsp,done,dma_ready,
 output logic [31:0] req_addr,
 output logic [2:0] outstanding,discard,
 output logic fetch_held,dma_instr_valid,instr_issued
);
 logic [31:0] cp_addr;
 logic [511:0] rdata;
 cmd_proc u_cp(
  .clk,.rst_n,.start,.pc_start(32'd0),.done,
  .fetch_req_valid(cp_valid),.fetch_req_addr(cp_addr),.fetch_req_ready(cp_ready),
  .fetch_rsp_valid(cp_rsp),.fetch_rsp_rdata(rdata),.perf_instr_issued(instr_issued),
  .dma_instr_valid,.dma_instr_ready(1'b1),.dma_busy(engine_busy),
  .dma_done_pulse(1'b0),.dma_done_sig_sem(8'd0),
  .gemm_instr_ready(1'b1),.gemm_busy(1'b0),.gemm_done_pulse(1'b0),.gemm_done_sig_sem(8'd0),
  .vec_instr_ready(1'b1),.vec_busy(1'b0),.vec_done_pulse(1'b0),.vec_done_sig_sem(8'd0),
  .attn_instr_ready(1'b1),.attn_busy(1'b0),.attn_done_pulse(1'b0),.attn_done_sig_sem(8'd0)
 );
 dram_arb u_arb(
  .clk,.rst_n,
  .dma_req_valid,.dma_req_we(1'b1),.dma_req_addr(32'h1000),
  .dma_req_wdata(512'd0),.dma_req_wstrb(64'hFFFF_FFFF_FFFF_FFFF),.dma_req_ready(dma_ready),
  .attn_req_valid(1'b0),.attn_req_we(1'b0),.attn_req_addr(32'd0),
  .attn_req_wdata(512'd0),.attn_req_wstrb(64'd0),
  .cp_req_valid(cp_valid),.cp_req_addr(cp_addr),.cp_req_ready(cp_ready),.cp_rsp_valid(cp_rsp),
  .rsp_rdata(rdata),
  .dram_req_valid(req_valid),.dram_req_ready(dram_ready),.dram_req_we(req_we),
  .dram_req_addr(req_addr),.dram_rsp_valid(dram_rsp),.dram_rsp_rdata(response_data)
 );
 assign outstanding=u_cp.outstanding;
 assign discard=u_cp.discard;
 assign fetch_held=u_cp.fetch_held;
endmodule
