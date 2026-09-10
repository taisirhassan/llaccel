// attn_engine_stub.sv — placeholder with the real attn_engine interface. Pops any
// instruction and retires it one cycle later without touching SRAM. Used only by
// rtl/filelist_core_stubs.f until the real rtl/attn_engine.sv lands.
module attn_engine
  import llaccel_pkg::*;
(
  input  logic         clk,
  input  logic         rst_n,
  input  logic         instr_valid,
  input  instr_words_t instr,
  output logic         instr_ready,
  input  logic [31:0]  pos,
  output logic         busy,
  output logic         done_pulse,
  output logic [7:0]   done_sig_sem,
  output logic         attn_rd_valid,
  output sram_req_t    attn_rd_req,
  input  logic         attn_rd_grant,
  input  logic         attn_rd_rvalid,
  input  logic [511:0] attn_rd_rdata,
  output logic         attn_wr_valid,
  output sram_req_t    attn_wr_req,
  output logic [511:0] attn_wr_wdata,
  output logic [63:0]  attn_wr_wstrb,
  input  logic         attn_wr_grant,
  output logic         perf_busy,
  output logic         perf_sram_stall,
  output logic         perf_mac_cycles
);
  logic       active;
  logic [7:0] sig;

  assign instr_ready  = !active;
  assign busy         = active;
  assign done_pulse   = active;
  assign done_sig_sem = sig;
  assign perf_busy    = active;
  assign perf_sram_stall = 1'b0;
  assign perf_mac_cycles = 1'b0;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      active <= 1'b0;
      sig    <= NO_SEM;
    end else begin
      if (instr_valid && instr_ready) begin
        active <= 1'b1;
        sig    <= instr_sig_sem(instr);
      end else begin
        active <= 1'b0;
      end
    end
  end

  assign attn_rd_valid = 1'b0;
  assign attn_rd_req   = '0;
  assign attn_wr_valid = 1'b0;
  assign attn_wr_req   = '0;
  assign attn_wr_wdata = '0;
  assign attn_wr_wstrb = '0;

  logic unused_ok;
  assign unused_ok = &{1'b0, pos, attn_rd_grant, attn_rd_rvalid, attn_rd_rdata, attn_wr_grant};
endmodule
