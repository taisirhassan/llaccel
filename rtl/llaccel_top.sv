// llaccel_top.sv — llaccel_core + 16 behavioral SRAM banks. This is what the
// simulation runtime backend and the system testbench build.
module llaccel_top
  import llaccel_pkg::*;
#(
  parameter bit EPILOGUE_FUSION = 1'b0
) (
  input  logic               clk,
  input  logic               rst_n,
  input  logic               start,
  input  logic [31:0]        pc_start,
  input  logic [31:0]        pos,
  output logic               done,
  output logic               dram_req_valid,
  input  logic               dram_req_ready,
  output logic               dram_req_we,
  output logic [31:0]        dram_req_addr,
  output logic [DRAM_DW-1:0] dram_req_wdata,
  output logic [DRAM_BEAT-1:0] dram_req_wstrb,
  input  logic               dram_rsp_valid,
  input  logic [DRAM_DW-1:0] dram_rsp_rdata,
  output logic [63:0]        perf [NUM_PERF]
);
  logic                  bank_en    [NBANKS];
  logic [BANK_BYTES-1:0] bank_we    [NBANKS];
  logic [BANK_AW-1:0]    bank_addr  [NBANKS];
  logic [BANK_DW-1:0]    bank_wdata [NBANKS];
  logic [BANK_DW-1:0]    bank_rdata [NBANKS];

  llaccel_core #(.EPILOGUE_FUSION(EPILOGUE_FUSION)) u_core (
    .clk, .rst_n, .start, .pc_start, .pos, .done,
    .dram_req_valid, .dram_req_ready, .dram_req_we, .dram_req_addr, .dram_req_wdata, .dram_req_wstrb,
    .dram_rsp_valid, .dram_rsp_rdata, .perf,
    .bank_en, .bank_we, .bank_addr, .bank_wdata, .bank_rdata);

  for (genvar b = 0; b < NBANKS; b++) begin : g_bank
    sram_bank u_bank (
      .clk, .en(bank_en[b]), .we(bank_we[b]), .addr(bank_addr[b]), .wdata(bank_wdata[b]), .rdata(bank_rdata[b]));
  end
endmodule
