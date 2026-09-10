// sram_bank.sv — behavioral single-port SRAM bank, 16 B wide, byte enables,
// 1-cycle read latency. Simulation only: it sits outside the synthesized
// boundary (llaccel_core exposes the bank ports; llaccel_top instantiates 16).
module sram_bank
  import llaccel_pkg::*;
#(
  parameter int unsigned DEPTH = SRAM_BYTES / LINE_BYTES   // words per bank (4096 for 1 MiB)
) (
  input  logic                 clk,
  input  logic                 en,
  input  logic [BANK_BYTES-1:0] we,
  input  logic [BANK_AW-1:0]   addr,
  input  logic [BANK_DW-1:0]   wdata,
  output logic [BANK_DW-1:0]   rdata
);
  localparam int unsigned IW = $clog2(DEPTH);

  logic [BANK_DW-1:0] mem [DEPTH] /* verilator public_flat_rw */;
  logic [IW-1:0]      idx;
  assign idx = addr[IW-1:0];

  always_ff @(posedge clk) begin
    if (en) begin
      for (int unsigned i = 0; i < BANK_BYTES; i++)
        if (we[i]) mem[idx][8*i +: 8] <= wdata[8*i +: 8];
      rdata <= mem[idx];
    end
  end

  logic unused_ok;
  assign unused_ok = &{1'b0, addr[BANK_AW-1:IW]};
endmodule
