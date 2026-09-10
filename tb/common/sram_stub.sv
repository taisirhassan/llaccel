// sram_stub.sv — behavioral SRAM harness for the engine unit testbenches.
//
// Implements the crossbar port contract of docs/ARCH.md / llaccel_pkg.sv for
// two read ports and one write port over a 1 MiB byte array (public, so the
// C++ testbench can load and inspect it directly):
//   * grant is combinational in the same cycle as the request;
//   * read data arrives the cycle after the grant, right-aligned
//     (byte i of the transfer at rdata[8*i +: 8]); bytes beyond the transfer
//     size are driven with 0xA5 so a consumer that reads them is caught;
//   * writes commit at the granting edge (byte i written iff wstrb[i]).
// Grants are denied (a) at random with probability deny_pct/100 per port per
// cycle, and (b) on a bank conflict with a higher-priority port granted in the
// same cycle (priority rd0 > rd1 > wr, matching the xbar's ordering of the
// vec_rd0/rd1/wr and attn_rd/wr ports). Requests are checked against the
// alignment / line-crossing rules; violations are fatal.
module sram_stub #(
  parameter int unsigned BYTES = llaccel_pkg::SRAM_BYTES
) (
  input  logic                   clk,
  input  logic                   rst_n,
  input  logic [7:0]             deny_pct,     // 0..100: probability (%) of withholding a grant

  input  logic                   rd0_valid,
  input  llaccel_pkg::sram_req_t rd0_req,
  output logic                   rd0_grant,
  output logic                   rd0_rvalid,
  output logic [511:0]           rd0_rdata,

  input  logic                   rd1_valid,
  input  llaccel_pkg::sram_req_t rd1_req,
  output logic                   rd1_grant,
  output logic                   rd1_rvalid,
  output logic [511:0]           rd1_rdata,

  input  logic                   wr_valid,
  input  llaccel_pkg::sram_req_t wr_req,
  input  logic [511:0]           wr_wdata,
  input  logic [63:0]            wr_wstrb,
  output logic                   wr_grant
);
  import llaccel_pkg::*;

  logic [7:0] mem [BYTES] /* verilator public_flat_rw */;

  // ---- request legality -----------------------------------------------------------
  function automatic bit req_legal(input logic valid, input sram_req_t r);
    int unsigned nb;
    if (!valid) return 1'b1;
    nb = size_bytes(r.size);
    if (r.size == SZ_256) return 1'b0;                       // engines never issue line requests
    if (r.addr[3:0] != 4'd0) return 1'b0;                    // 16-B aligned
    if ((32'(r.addr[7:0]) + nb) > 32'd256) return 1'b0;      // never crosses a 256-B line
    if ((32'(r.addr) + nb) > BYTES) return 1'b0;
    return 1'b1;
  endfunction

  // Bank mask (16 banks, 16 B each) touched by a request.
  function automatic logic [15:0] bank_mask(input logic valid, input sram_req_t r);
    logic [15:0] m;
    int unsigned nb, b0;
    m = '0;
    if (valid) begin
      nb = size_bytes(r.size) / 16;
      b0 = 32'(r.addr[7:4]);
      for (int unsigned i = 0; i < 16; i++) if (i >= b0 && i < b0 + nb) m[i] = 1'b1;
    end
    return m;
  endfunction

  // ---- grant logic -------------------------------------------------------------------
  logic rnd0, rnd1, rndw;   // per-port random denial for this cycle
  logic [15:0] bm0, bm1, bmw;

  always_comb begin
    bm0 = bank_mask(rd0_valid, rd0_req);
    bm1 = bank_mask(rd1_valid, rd1_req);
    bmw = bank_mask(wr_valid, wr_req);
    rd0_grant = rd0_valid && !rnd0;
    rd1_grant = rd1_valid && !rnd1 && !(rd0_grant && (bm1 & bm0) != 16'd0);
    wr_grant  = wr_valid  && !rndw && !(rd0_grant && (bmw & bm0) != 16'd0)
                                   && !(rd1_grant && (bmw & bm1) != 16'd0);
  end

  // Fresh random denial decisions each cycle (decided at the negative edge so
  // they are stable for the whole positive-edge cycle).
  always_ff @(negedge clk) begin
    rnd0 <= (32'($urandom % 100) < 32'(deny_pct));
    rnd1 <= (32'($urandom % 100) < 32'(deny_pct));
    rndw <= (32'($urandom % 100) < 32'(deny_pct));
  end

  // ---- data path -----------------------------------------------------------------------------
  function automatic logic [511:0] read_bytes(input sram_req_t r);
    logic [511:0] d;
    int unsigned nb;
    nb = size_bytes(r.size);
    d = {64{8'hA5}};
    for (int unsigned i = 0; i < 64; i++)
      if (i < nb) d[8*i +: 8] = mem[32'(r.addr) + i];
    return d;
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd0_rvalid <= 1'b0;
      rd1_rvalid <= 1'b0;
      rd0_rdata  <= '0;
      rd1_rdata  <= '0;
    end else begin
      rd0_rvalid <= rd0_grant;
      rd1_rvalid <= rd1_grant;
      if (rd0_grant) rd0_rdata <= read_bytes(rd0_req);
      if (rd1_grant) rd1_rdata <= read_bytes(rd1_req);
      if (wr_grant) begin
        for (int unsigned i = 0; i < 64; i++)
          if (i < size_bytes(wr_req.size) && wr_wstrb[i]) mem[32'(wr_req.addr) + i] <= wr_wdata[8*i +: 8];
      end
    end
  end

  // ---- contract checks ----------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (rst_n) begin
      if (!req_legal(rd0_valid, rd0_req)) $fatal(1, "sram_stub: illegal rd0 request addr=%h size=%0d", rd0_req.addr, rd0_req.size);
      if (!req_legal(rd1_valid, rd1_req)) $fatal(1, "sram_stub: illegal rd1 request addr=%h size=%0d", rd1_req.addr, rd1_req.size);
      if (!req_legal(wr_valid, wr_req))   $fatal(1, "sram_stub: illegal wr request addr=%h size=%0d", wr_req.addr, wr_req.size);
    end
  end
endmodule
