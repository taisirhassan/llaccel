// Exercise every real crossbar port with independent requests and byte strobes.
module tb_xbar_top import llaccel_pkg::*; (
 input logic clk, rst_n,
 input logic [9:0] valid,
 input logic [23:0] addr [10],
 input logic [1:0] size [10],
 input logic [511:0] wdata [10],
 input logic [63:0] wstrb [10],
 output logic [9:0] grant, stall, rvalid,
 output logic [2047:0] rdata [10],
 output logic [15:0] rd_bytes, wr_bytes
);
 sram_req_t req [10];
 for (genvar p=0; p<10; p++) begin
   assign req[p].addr=addr[p];
   assign req[p].size=sram_size_e'(size[p]);
 end
 logic bank_en [NBANKS];
 logic [BANK_BYTES-1:0] bank_we [NBANKS];
 logic [BANK_AW-1:0] bank_addr [NBANKS];
 logic [BANK_DW-1:0] bank_wdata [NBANKS], bank_rdata [NBANKS];
 assign rdata[1][2047:512]='0;
 assign rvalid[2]=1'b0; assign rdata[2]='0;
 assign rdata[3][2047:512]='0;
 assign rvalid[4]=1'b0; assign rdata[4]='0;
 assign rdata[5][2047:512]='0;
 assign rdata[6][2047:512]='0;
 assign rvalid[7]=1'b0; assign rdata[7]='0;
 assign rvalid[8]=1'b0; assign rdata[8]='0;
 assign rdata[9][2047:512]='0;
 sram_xbar dut ( .clk, .rst_n,
 .gemm_w_valid(valid[0]), .gemm_w_req(req[0]), .gemm_w_grant(grant[0]), .gemm_w_stall(stall[0]),
 .gemm_w_rvalid(rvalid[0]), .gemm_w_rdata(rdata[0]),
 .gemm_a_valid(valid[1]), .gemm_a_req(req[1]), .gemm_a_grant(grant[1]), .gemm_a_stall(stall[1]),
 .gemm_a_rvalid(rvalid[1]), .gemm_a_rdata(rdata[1][511:0]),
 .gemm_wr_valid(valid[2]), .gemm_wr_req(req[2]), .gemm_wr_grant(grant[2]), .gemm_wr_stall(stall[2]),
 .gemm_wr_wdata(wdata[2]), .gemm_wr_wstrb(wstrb[2]),
 .attn_rd_valid(valid[3]), .attn_rd_req(req[3]), .attn_rd_grant(grant[3]), .attn_rd_stall(stall[3]),
 .attn_rd_rvalid(rvalid[3]), .attn_rd_rdata(rdata[3][511:0]),
 .attn_wr_valid(valid[4]), .attn_wr_req(req[4]), .attn_wr_grant(grant[4]), .attn_wr_stall(stall[4]),
 .attn_wr_wdata(wdata[4]), .attn_wr_wstrb(wstrb[4]),
 .vec_rd0_valid(valid[5]), .vec_rd0_req(req[5]), .vec_rd0_grant(grant[5]), .vec_rd0_stall(stall[5]),
 .vec_rd0_rvalid(rvalid[5]), .vec_rd0_rdata(rdata[5][511:0]),
 .vec_rd1_valid(valid[6]), .vec_rd1_req(req[6]), .vec_rd1_grant(grant[6]), .vec_rd1_stall(stall[6]),
 .vec_rd1_rvalid(rvalid[6]), .vec_rd1_rdata(rdata[6][511:0]),
 .vec_wr_valid(valid[7]), .vec_wr_req(req[7]), .vec_wr_grant(grant[7]), .vec_wr_stall(stall[7]),
 .vec_wr_wdata(wdata[7]), .vec_wr_wstrb(wstrb[7]),
 .dma_wr_valid(valid[8]), .dma_wr_req(req[8]), .dma_wr_grant(grant[8]), .dma_wr_stall(stall[8]),
 .dma_wr_wdata(wdata[8]), .dma_wr_wstrb(wstrb[8]),
 .dma_rd_valid(valid[9]), .dma_rd_req(req[9]), .dma_rd_grant(grant[9]), .dma_rd_stall(stall[9]),
 .dma_rd_rvalid(rvalid[9]), .dma_rd_rdata(rdata[9][511:0]),
 .bank_en, .bank_we, .bank_addr, .bank_wdata, .bank_rdata, .rd_bytes, .wr_bytes);
 for (genvar b=0; b<NBANKS; b++) begin : g_bank
 sram_bank u_bank(.clk, .en(bank_en[b]), .we(bank_we[b]), .addr(bank_addr[b]), .wdata(bank_wdata[b]), .rdata(bank_rdata[b]));
 end
endmodule
