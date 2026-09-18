// llaccel_pkg.sv — device parameters, ISA encoding and shared types.
// Mirrors include/llaccel/isa.h and docs/ISA.md / docs/ARCH.md exactly.
package llaccel_pkg;

  // ---- device parameters ------------------------------------------------------
  localparam int unsigned SRAM_BYTES  = 1 << 20;
  localparam int unsigned SRAM_AW     = 24;               // byte address width
  localparam int unsigned NBANKS      = 16;
  localparam int unsigned BANK_BYTES  = 16;
  localparam int unsigned BANK_DW     = BANK_BYTES * 8;   // 128
  localparam int unsigned LINE_BYTES  = NBANKS * BANK_BYTES; // 256
  localparam int unsigned BANK_AW     = SRAM_AW - 8;      // word index within a bank (addr[23:8])
  localparam int unsigned GEMM_TN     = 16;
  localparam int unsigned GEMM_TK     = 16;
  localparam int unsigned GEMM_TM     = 16;
  localparam int unsigned VEC_LANES   = 16;
  localparam int unsigned ATTN_LANES  = 64;
  localparam int unsigned ATTN_TMAX   = 4096;
  localparam int unsigned NSEM        = 32;
  localparam int unsigned QDEPTH      = 8;
  localparam int unsigned DRAM_BEAT   = 64;
  localparam int unsigned DRAM_DW     = DRAM_BEAT * 8;    // 512
  localparam int unsigned INSTR_BYTES = 64;
  localparam int unsigned INSTR_W     = INSTR_BYTES * 8;  // 512
  localparam int unsigned NUM_PERF    = 27;
  localparam logic [7:0]  NO_SEM      = 8'hFF;

  // ---- opcodes ---------------------------------------------------------------------
  typedef enum logic [7:0] {
    OP_NOP         = 8'h00,
    OP_HALT        = 8'h01,
    OP_DMA_LOAD    = 8'h10,
    OP_DMA_STORE   = 8'h11,
    OP_GEMM        = 8'h20,
    OP_VEC_RMSNORM = 8'h30,
    OP_VEC_ROPE    = 8'h31,
    OP_VEC_SILU    = 8'h32,
    OP_VEC_MUL     = 8'h33,
    OP_VEC_ADD     = 8'h34,
    OP_VEC_QUANT   = 8'h35,
    OP_ATTN        = 8'h40,
    OP_KV_WRITE    = 8'h41
  } opcode_e;

  typedef enum logic [2:0] { ENG_CP = 3'd0, ENG_DMA = 3'd1, ENG_GEMM = 3'd2, ENG_VEC = 3'd3, ENG_ATTN = 3'd4 } engine_e;

  function automatic engine_e engine_of(input logic [7:0] op);
    if (op < 8'h10) return ENG_CP;
    if (op < 8'h20) return ENG_DMA;
    if (op < 8'h30) return ENG_GEMM;
    if (op < 8'h40) return ENG_VEC;
    return ENG_ATTN;
  endfunction

  typedef enum logic [3:0] { EP_NONE = 4'd0, EP_RESADD = 4'd1, EP_SILU = 4'd2, EP_MUL = 4'd3 } epilogue_e;
  localparam int unsigned FLAG_HAS_BIAS = 0;  // bit index in flags

  // ---- instruction word: 16 x u32, little-endian word order ---------------------------
  // w[0] = opcode | flags<<8 | wait_sem<<16 | signal_sem<<24 ; w[1] = wait_val ; w[2..15] operands
  typedef logic [15:0][31:0] instr_words_t;   // w[i] = word i

  function automatic logic [7:0] instr_opcode(input instr_words_t w);   return w[0][7:0];   endfunction
  function automatic logic [7:0] instr_flags(input instr_words_t w);    return w[0][15:8];  endfunction
  function automatic logic [7:0] instr_wait_sem(input instr_words_t w); return w[0][23:16]; endfunction
  function automatic logic [7:0] instr_sig_sem(input instr_words_t w);  return w[0][31:24]; endfunction
  function automatic logic [31:0] instr_wait_val(input instr_words_t w); return w[1];       endfunction

  // Operand word indices (docs/ISA.md)
  // DMA_LOAD/STORE : 2 sram, 3 dram, 4 rows, 5 row_bytes, 6 src_stride, 7 dst_stride
  // GEMM           : 2 a, 3 w, 4 out, 5 rq, 6 bias, 7 aux, 8 M, 9 N, 10 K, 11 ep, 12 silu_Mi, 13 silu_Si_sh
  //                  ep = mode[3:0] | out_i8[4] | aux_shift[15:8] ; silu_Si_sh = Si[7:0] | sh_out[15:8]
  // VEC_RMSNORM    : 2 src, 3 gamma, 4 dst, 5 M, 6 K, 7 eps_t, 8 C, 9 sh_post
  // VEC_ROPE       : 2 src, 3 dst, 4 M, 5 H, 6 D, 7 table, 8 table_stride
  // VEC_SILU       : 2 src, 3 dst, 4 count, 5 Mi, 6 Si, 7 sh_out
  // VEC_MUL        : 2 a, 3 b, 4 dst, 5 count, 6 sh
  // VEC_ADD        : 2 a, 3 b, 4 dst, 5 count, 6 sh_b
  // VEC_QUANT      : 2 src, 3 dst, 4 count, 5 M, 6 S
  // ATTN           : 2 q, 3 out, 4 kbase, 5 vbase, 6 M, 7 H, 8 Hkv, 9 D, 10 kv_stride, 11 Ms, 12 Ss, 13 Mo, 14 So
  // KV_WRITE       : 2 src, 3 base, 4 M, 5 Hkv, 6 D, 7 kv_stride

  // ---- SRAM crossbar port contract ---------------------------------------------------------
  // Line request: addr is 16-B aligned (256-B aligned for SZ_256); never crosses a 256-B line.
  typedef enum logic [1:0] { SZ_16 = 2'd0, SZ_32 = 2'd1, SZ_64 = 2'd2, SZ_256 = 2'd3 } sram_size_e;
  typedef struct packed {
    logic [SRAM_AW-1:0] addr;
    sram_size_e         size;
  } sram_req_t;
  // Per port (see sram_xbar.sv):
  //   read  port : req_valid, req (sram_req_t), grant (combinational, same cycle),
  //                rsp_valid (next cycle), rsp_data [DW-1:0]  (DW = 2048 for gemm_w, else 512;
  //                data is right-aligned: byte i of the transfer is rsp_data[8*i +: 8])
  //   write port : req_valid, req, grant, wdata[511:0], wstrb[63:0] (byte i valid if wstrb[i])
  // Priority high→low: gemm_w, gemm_a, gemm_wr, attn_rd, attn_wr, vec_rd0, vec_rd1, vec_wr, dma_wr, dma_rd.
  function automatic int unsigned size_bytes(input sram_size_e s);
    case (s)
      SZ_16:  return 16;
      SZ_32:  return 32;
      SZ_64:  return 64;
      default: return 256;
    endcase
  endfunction

  // ---- DRAM interface (to the C++ model) ----------------------------------------------------
  // req_valid/req_ready handshake; req_we, req_addr[31:0] (64-B aligned), req_wdata[511:0], req_wstrb[63:0]
  // rsp_valid, rsp_rdata[511:0] : read responses in request order, at most one per cycle.

  // ---- perf counter indices ---------------------------------------------------------------------
  localparam int unsigned PERF_CYCLES = 0, PERF_INSTR_ISSUED = 1, PERF_CP_STALL_WAIT = 2, PERF_CP_STALL_QFULL = 3,
    PERF_CP_STALL_FETCH = 4, PERF_GEMM_BUSY = 5, PERF_GEMM_MAC_CYCLES = 6, PERF_GEMM_SRAM_STALL = 7,
    PERF_GEMM_EPILOGUE_CYCLES = 8, PERF_VEC_BUSY = 9, PERF_VEC_SRAM_STALL = 10, PERF_ATTN_BUSY = 11,
    PERF_ATTN_SRAM_STALL = 12, PERF_ATTN_MAC_CYCLES = 13, PERF_DMA_BUSY = 14, PERF_DMA_SRAM_STALL = 15,
    PERF_DMA_DRAM_WAIT = 16, PERF_SRAM_RD_BYTES = 17, PERF_SRAM_WR_BYTES = 18, PERF_DRAM_RD_BYTES = 19,
    PERF_DRAM_WR_BYTES = 20, PERF_GEMM_IDLE_QEMPTY = 21, PERF_VEC_IDLE_QEMPTY = 22, PERF_ATTN_IDLE_QEMPTY = 23, PERF_ATTN_DRAM_WAIT = 24, PERF_ATTN_DRAM_RD_BYTES = 25, PERF_ATTN_DRAM_WR_BYTES = 26;

  // ---- shared arithmetic helpers (docs/NUMERICS.md) --------------------------------------------------
  // round-half-up arithmetic right shift of a 64-bit signed value
  function automatic logic signed [63:0] rshr64(input logic signed [63:0] v, input logic [5:0] s);
    if (s == 0) return v;
    return (v >>> s) + $signed({63'd0, v[s - 1]});
  endfunction
  function automatic logic signed [15:0] sat16(input logic signed [63:0] v);
    if (v > 64'sd32767)  return 16'sd32767;
    if (v < -64'sd32768) return -16'sd32768;
    return v[15:0];
  endfunction
  function automatic logic signed [7:0] sat8(input logic signed [63:0] v);
    if (v > 64'sd127)  return 8'sd127;
    if (v < -64'sd128) return -8'sd128;
    return v[7:0];
  endfunction
  function automatic logic [7:0] satu8(input logic signed [63:0] v);
    if (v > 64'sd255) return 8'd255;
    if (v < 0)        return 8'd0;
    return v[7:0];
  endfunction
  // sat(rshr(v * M, S)) with v up to 48-bit signed and M a positive 31-bit multiplier
  // The true product needs at most 48 + 31 = 79 bits; a 64-bit multiply keeps its
  // low 64 bits, which is exact whenever |v * m| < 2^63 (every use in NUMERICS.md).
  function automatic logic signed [63:0] mulshift64(input logic signed [47:0] v, input logic [31:0] m, input logic [5:0] s);
    logic signed [63:0] p;
    p = $signed({{16{v[47]}}, v}) * $signed({32'd0, m});
    return rshr64(p, s);
  endfunction

endpackage
