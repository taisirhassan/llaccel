// sram_xbar.sv — 16-bank scratchpad crossbar with fixed-priority per-bank arbitration.
//
// Every requester presents a line request {addr, size}; the banks it touches are
// addr[7:4] .. addr[7:4] + size/16 - 1 (a request never crosses a 256-B line and
// SZ_256 requests are 256-B aligned). A request is granted in the same cycle iff
// none of its banks was claimed by a higher-priority request that cycle; a denied
// request must be held by the requester. Granted reads return data the next
// cycle, right-aligned (byte i of the transfer at rdata[8*i +: 8]); granted writes
// complete that cycle (wdata/wstrb right-aligned the same way).
//
// Priority (high -> low), port index in the internal arrays:
//   0 gemm_w  1 gemm_a  2 gemm_wr  3 attn_rd  4 attn_wr
//   5 vec_rd0 6 vec_rd1 7 vec_wr   8 dma_wr   9 dma_rd
module sram_xbar
  import llaccel_pkg::*;
(
  input  logic                 clk,
  input  logic                 rst_n,
  // ---- gemm ----
  input  logic                 gemm_w_valid,
  input  sram_req_t            gemm_w_req,
  output logic                 gemm_w_grant,
  output logic                 gemm_w_rvalid,
  output logic [2047:0]        gemm_w_rdata,
  input  logic                 gemm_a_valid,
  input  sram_req_t            gemm_a_req,
  output logic                 gemm_a_grant,
  output logic                 gemm_a_rvalid,
  output logic [511:0]         gemm_a_rdata,
  input  logic                 gemm_wr_valid,
  input  sram_req_t            gemm_wr_req,
  input  logic [511:0]         gemm_wr_wdata,
  input  logic [63:0]          gemm_wr_wstrb,
  output logic                 gemm_wr_grant,
  // ---- attn ----
  input  logic                 attn_rd_valid,
  input  sram_req_t            attn_rd_req,
  output logic                 attn_rd_grant,
  output logic                 attn_rd_rvalid,
  output logic [511:0]         attn_rd_rdata,
  input  logic                 attn_wr_valid,
  input  sram_req_t            attn_wr_req,
  input  logic [511:0]         attn_wr_wdata,
  input  logic [63:0]          attn_wr_wstrb,
  output logic                 attn_wr_grant,
  // ---- vec ----
  input  logic                 vec_rd0_valid,
  input  sram_req_t            vec_rd0_req,
  output logic                 vec_rd0_grant,
  output logic                 vec_rd0_rvalid,
  output logic [511:0]         vec_rd0_rdata,
  input  logic                 vec_rd1_valid,
  input  sram_req_t            vec_rd1_req,
  output logic                 vec_rd1_grant,
  output logic                 vec_rd1_rvalid,
  output logic [511:0]         vec_rd1_rdata,
  input  logic                 vec_wr_valid,
  input  sram_req_t            vec_wr_req,
  input  logic [511:0]         vec_wr_wdata,
  input  logic [63:0]          vec_wr_wstrb,
  output logic                 vec_wr_grant,
  // ---- dma ----
  input  logic                 dma_wr_valid,
  input  sram_req_t            dma_wr_req,
  input  logic [511:0]         dma_wr_wdata,
  input  logic [63:0]          dma_wr_wstrb,
  output logic                 dma_wr_grant,
  input  logic                 dma_rd_valid,
  input  sram_req_t            dma_rd_req,
  output logic                 dma_rd_grant,
  output logic                 dma_rd_rvalid,
  output logic [511:0]         dma_rd_rdata,
  // ---- banks ----
  output logic                 bank_en    [NBANKS],
  output logic [BANK_BYTES-1:0] bank_we   [NBANKS],
  output logic [BANK_AW-1:0]   bank_addr  [NBANKS],
  output logic [BANK_DW-1:0]   bank_wdata [NBANKS],
  input  logic [BANK_DW-1:0]   bank_rdata [NBANKS],
  // ---- perf: denied-cycle pulses per port, bytes moved this cycle ----
  output logic                 gemm_w_stall,
  output logic                 gemm_a_stall,
  output logic                 gemm_wr_stall,
  output logic                 attn_rd_stall,
  output logic                 attn_wr_stall,
  output logic                 vec_rd0_stall,
  output logic                 vec_rd1_stall,
  output logic                 vec_wr_stall,
  output logic                 dma_wr_stall,
  output logic                 dma_rd_stall,
  output logic [15:0]          rd_bytes,
  output logic [15:0]          wr_bytes
);
  localparam int unsigned NP = 10;
  localparam logic [NP-1:0] IS_WRITE = 10'b0_1_1_0_1_0_1_0_0_0; // bit p set for write ports 2,4,7,8

  // ---- gather ports into arrays -------------------------------------------------
  logic        v   [NP];
  sram_req_t   r   [NP];
  logic [511:0] wd [NP];
  logic [63:0]  ws [NP];
  logic        g   [NP];
  logic [NBANKS-1:0] mask [NP];
  logic [NBANKS-1:0] taken;

  always_comb begin
    v[0] = gemm_w_valid;  r[0] = gemm_w_req;  wd[0] = '0;            ws[0] = '0;
    v[1] = gemm_a_valid;  r[1] = gemm_a_req;  wd[1] = '0;            ws[1] = '0;
    v[2] = gemm_wr_valid; r[2] = gemm_wr_req; wd[2] = gemm_wr_wdata; ws[2] = gemm_wr_wstrb;
    v[3] = attn_rd_valid; r[3] = attn_rd_req; wd[3] = '0;            ws[3] = '0;
    v[4] = attn_wr_valid; r[4] = attn_wr_req; wd[4] = attn_wr_wdata; ws[4] = attn_wr_wstrb;
    v[5] = vec_rd0_valid; r[5] = vec_rd0_req; wd[5] = '0;            ws[5] = '0;
    v[6] = vec_rd1_valid; r[6] = vec_rd1_req; wd[6] = '0;            ws[6] = '0;
    v[7] = vec_wr_valid;  r[7] = vec_wr_req;  wd[7] = vec_wr_wdata;  ws[7] = vec_wr_wstrb;
    v[8] = dma_wr_valid;  r[8] = dma_wr_req;  wd[8] = dma_wr_wdata;  ws[8] = dma_wr_wstrb;
    v[9] = dma_rd_valid;  r[9] = dma_rd_req;  wd[9] = '0;            ws[9] = '0;
  end

  // bank mask of a line request
  function automatic logic [NBANKS-1:0] bank_mask(input sram_req_t q);
    logic [NBANKS-1:0] m;
    case (q.size)
      SZ_16:   m = 16'h0001;
      SZ_32:   m = 16'h0003;
      SZ_64:   m = 16'h000F;
      default: m = 16'hFFFF;
    endcase
    return m << q.addr[7:4];
  endfunction

  // ---- fixed-priority arbitration (combinational) --------------------------------
  always_comb begin
    taken = '0;
    for (int unsigned p = 0; p < NP; p++) begin
      mask[p] = bank_mask(r[p]);
      g[p]    = v[p] && ((mask[p] & taken) == '0);
      if (g[p]) taken = taken | mask[p];
    end
  end

  assign gemm_w_grant  = g[0];
  assign gemm_a_grant  = g[1];
  assign gemm_wr_grant = g[2];
  assign attn_rd_grant = g[3];
  assign attn_wr_grant = g[4];
  assign vec_rd0_grant = g[5];
  assign vec_rd1_grant = g[6];
  assign vec_wr_grant  = g[7];
  assign dma_wr_grant  = g[8];
  assign dma_rd_grant  = g[9];

  assign gemm_w_stall  = v[0] && !g[0];
  assign gemm_a_stall  = v[1] && !g[1];
  assign gemm_wr_stall = v[2] && !g[2];
  assign attn_rd_stall = v[3] && !g[3];
  assign attn_wr_stall = v[4] && !g[4];
  assign vec_rd0_stall = v[5] && !g[5];
  assign vec_rd1_stall = v[6] && !g[6];
  assign vec_wr_stall  = v[7] && !g[7];
  assign dma_wr_stall  = v[8] && !g[8];
  assign dma_rd_stall  = v[9] && !g[9];

  // ---- bank drive ------------------------------------------------------------------
  // Grants are bank-disjoint, so at most one port matches each bank.
  always_comb begin
    for (int unsigned b = 0; b < NBANKS; b++) begin
      logic [3:0] chunk;
      bank_en[b]    = 1'b0;
      bank_we[b]    = '0;
      bank_addr[b]  = '0;
      bank_wdata[b] = '0;
      for (int unsigned p = 0; p < NP; p++) begin
        if (g[p] && mask[p][b]) begin
          chunk         = 4'(b) - r[p].addr[7:4];
          bank_en[b]    = 1'b1;
          bank_addr[b]  = r[p].addr[SRAM_AW-1:8];
          bank_we[b]    = IS_WRITE[p] ? ws[p][16*chunk[1:0] +: 16] : '0;
          bank_wdata[b] = wd[p][128*chunk[1:0] +: 128];
        end
      end
    end
  end

  // ---- read response (next cycle) ----------------------------------------------------
  logic       rg     [NP];
  logic [3:0] rstart [NP];

  always_ff @(posedge clk) begin
    for (int unsigned p = 0; p < NP; p++) begin
      if (!rst_n) begin
        rg[p]     <= 1'b0;
        rstart[p] <= '0;
      end else begin
        rg[p]     <= g[p] && !IS_WRITE[p];
        rstart[p] <= r[p].addr[7:4];
      end
    end
  end

  function automatic logic [511:0] gather4(input logic [3:0] start, input logic [BANK_DW-1:0] rd [NBANKS]);
    logic [511:0] d;
    for (int unsigned c = 0; c < 4; c++) d[128*c +: 128] = rd[4'(start + 4'(c))];
    return d;
  endfunction

  always_comb begin
    for (int unsigned c = 0; c < NBANKS; c++)
      gemm_w_rdata[128*c +: 128] = bank_rdata[4'(rstart[0] + 4'(c))];
  end
  assign gemm_w_rvalid  = rg[0];
  assign gemm_a_rvalid  = rg[1];
  assign gemm_a_rdata   = gather4(rstart[1], bank_rdata);
  assign attn_rd_rvalid = rg[3];
  assign attn_rd_rdata  = gather4(rstart[3], bank_rdata);
  assign vec_rd0_rvalid = rg[5];
  assign vec_rd0_rdata  = gather4(rstart[5], bank_rdata);
  assign vec_rd1_rvalid = rg[6];
  assign vec_rd1_rdata  = gather4(rstart[6], bank_rdata);
  assign dma_rd_rvalid  = rg[9];
  assign dma_rd_rdata   = gather4(rstart[9], bank_rdata);

  // ---- byte accounting ------------------------------------------------------------------
  function automatic logic [63:0] strb_mask(input sram_size_e s);
    case (s)
      SZ_16:   return 64'h0000_0000_0000_FFFF;
      SZ_32:   return 64'h0000_0000_FFFF_FFFF;
      default: return 64'hFFFF_FFFF_FFFF_FFFF;
    endcase
  endfunction

  always_comb begin
    rd_bytes = '0;
    wr_bytes = '0;
    for (int unsigned p = 0; p < NP; p++) begin
      if (g[p]) begin
        if (IS_WRITE[p]) wr_bytes = wr_bytes + 16'($countones(ws[p] & strb_mask(r[p].size)));
        else             rd_bytes = rd_bytes + 16'(size_bytes(r[p].size));
      end
    end
  end

  logic unused_ok;
  always_comb begin
    unused_ok = 1'b0;
    for (int unsigned p = 0; p < NP; p++) unused_ok = unused_ok | rg[p] | (|rstart[p]);
  end
endmodule
