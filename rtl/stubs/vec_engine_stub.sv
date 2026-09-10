// vec_engine_stub.sv — placeholder with the real vec_engine interface. Pops any
// instruction and retires it one cycle later without touching SRAM. Used only by
// rtl/filelist_core_stubs.f until the real rtl/vec_engine.sv lands.
module vec_engine
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
  output logic         vec_rd0_valid,
  output sram_req_t    vec_rd0_req,
  input  logic         vec_rd0_grant,
  input  logic         vec_rd0_rvalid,
  input  logic [511:0] vec_rd0_rdata,
  output logic         vec_rd1_valid,
  output sram_req_t    vec_rd1_req,
  input  logic         vec_rd1_grant,
  input  logic         vec_rd1_rvalid,
  input  logic [511:0] vec_rd1_rdata,
  output logic         vec_wr_valid,
  output sram_req_t    vec_wr_req,
  output logic [511:0] vec_wr_wdata,
  output logic [63:0]  vec_wr_wstrb,
  input  logic         vec_wr_grant,
  output logic         perf_busy,
  output logic         perf_sram_stall
);
  logic       active;
  logic [7:0] sig;

  assign instr_ready  = !active;
  assign busy         = active;
  assign done_pulse   = active;
  assign done_sig_sem = sig;
  assign perf_busy    = active;
  assign perf_sram_stall = 1'b0;

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

  assign vec_rd0_valid = 1'b0;
  assign vec_rd0_req   = '0;
  assign vec_rd1_valid = 1'b0;
  assign vec_rd1_req   = '0;
  assign vec_wr_valid  = 1'b0;
  assign vec_wr_req    = '0;
  assign vec_wr_wdata  = '0;
  assign vec_wr_wstrb  = '0;

  logic unused_ok;
  assign unused_ok = &{1'b0, pos, vec_rd0_grant, vec_rd0_rvalid, vec_rd0_rdata,
                       vec_rd1_grant, vec_rd1_rvalid, vec_rd1_rdata, vec_wr_grant};
endmodule
