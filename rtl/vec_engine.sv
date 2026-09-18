// 16-lane i16 vector engine; see docs/ARCH.md and docs/NUMERICS.md.
// three walkers drive rd0, rd1 and compute/write steps. response FIFO credits
// include in-flight reads; writes remain held until granted.
// read -> compute -> write ordering keeps in-place operations safe.
// 32-B accesses crossing a 256-B SRAM line split into two 16-B pieces.
// RMSNorm uses a 48-bit sum, isqrt/udiv, then a scaling pass per row.
// RoPE reads x1/cos, then x2/sin, and writes both halves in three substeps.
// SiLU interpolates two sigmoid LUT entries per lane.

/* verilator lint_off DECLFILENAME */  // private helper modules live in the engine's file

// ---------------------------------------------------------------------------------------
// step walker. ROLE 0 = rd0 address, 1 = rd1 address, 2 = write address.
// ---------------------------------------------------------------------------------------
module vec_engine_walker
  import llaccel_pkg::*;
#(
  parameter int ROLE = 0
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        load,          // start a new instruction (parameters below are sampled)
  input  logic        load_empty,    // the new instruction has zero steps
  input  logic [23:0] base_in,       // rd0: src; rd1: gamma / RoPE table row for cm=0; wr: dst
  input  logic [23:0] stride_in,     // added to the row base when m advances
  // instruction parameters (registered by the engine, stable while running)
  input  logic [7:0]  op,
  input  logic [7:0]  M,
  input  logic [7:0]  H,
  input  logic [8:0]  D,
  input  logic [15:0] K16,           // K / 16 (RMSNorm)
  input  logic [15:0] nsteps,        // count / 16 (flat ops)
  input  logic        advance,       // consume the current step
  output logic        active,        // steps remain
  output logic        en_rd0,        // current step reads on rd0
  output logic        en_rd1,        // current step reads on rd1
  output logic        en_wr,         // current step writes
  output logic [1:0]  sub,           // sub-step (RMSNorm: pass; RoPE: 0/1/2)
  output logic        j_last,        // last group of the current row/pass (RMSNorm pass boundary)
  output logic [23:0] addr,          // address for this ROLE
  output sram_size_e  size
);
  logic [7:0]  cm, ch;
  logic [15:0] cj;
  logic [1:0]  sub_r;
  logic [23:0] rowbase, stride;

  logic is_flat, is_norm, is_rope, is_quant;
  logic [15:0] jmax;
  logic last;
  logic [23:0] hoff, joff, half;

  assign sub = sub_r;
  assign is_flat  = (op == 8'(OP_VEC_SILU)) || (op == 8'(OP_VEC_MUL)) || (op == 8'(OP_VEC_ADD)) || (op == 8'(OP_VEC_QUANT));
  assign is_norm  = (op == 8'(OP_VEC_RMSNORM));
  assign is_rope  = (op == 8'(OP_VEC_ROPE));
  assign is_quant = (op == 8'(OP_VEC_QUANT));

  always_comb begin
    // RoPE uses 1/1/2/4/8 groups per half for D=16/32/64/128/256.
    // the three substeps save x1/cos, compute both halves, then write y2;
    // all FIFOs and lane registers retain the same fixed 16-lane width.
    // per-op loop bounds and offsets
    if (is_flat)      jmax = nsteps - 16'd1;
    else if (is_norm) jmax = K16 - 16'd1;
    else              jmax = (D == 9'd16) ? 16'd0 : (16'(D) >> 5) - 16'd1; // ceil((D/2)/16)-1
    j_last = (cj == jmax);
    hoff = (24'(ch) * 24'(D)) << 1;                                   // 2 * ch * D bytes
    joff = {3'd0, cj, 5'd0};                                 // 32 bytes per 16-lane half-group
    half = 24'(D);                                            // D/2 elements * 2 bytes

    en_rd0 = 1'b0; en_rd1 = 1'b0; en_wr = 1'b0; last = 1'b0; addr = '0; size = SZ_32;
    if (is_flat) begin
      en_rd0 = 1'b1;
      en_rd1 = (op == 8'(OP_VEC_MUL)) || (op == 8'(OP_VEC_ADD));
      en_wr  = 1'b1;
      last   = j_last;
      if (ROLE == 2 && is_quant) begin
        addr = rowbase + {4'd0, cj, 4'd0};                     // 16 B of i8 per step
        size = SZ_16;
      end else begin
        addr = rowbase + {3'd0, cj, 5'd0};                     // 32 B per step
      end
    end else if (is_norm) begin
      en_rd0 = 1'b1;
      en_rd1 = sub_r[0];
      en_wr  = sub_r[0];
      last   = sub_r[0] && j_last && (cm == M - 8'd1);
      addr   = rowbase + {3'd0, cj, 5'd0};
    end else if (is_rope) begin
      en_rd0 = (sub_r != 2'd2);
      en_rd1 = (sub_r != 2'd2);
      en_wr  = (sub_r != 2'd0);
      last   = (sub_r == 2'd2) && j_last && (ch == H - 8'd1) && (cm == M - 8'd1);
      size   = (D == 9'd16) ? SZ_16 : SZ_32;
      if (ROLE == 1)      addr = rowbase + joff + ((sub_r == 2'd1) ? half : 24'd0);
      else if (ROLE == 0) addr = rowbase + hoff + joff + ((sub_r == 2'd1) ? half : 24'd0);
      else                addr = rowbase + hoff + joff + ((sub_r == 2'd2) ? half : 24'd0);
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      active <= 1'b0;
      cm <= '0; ch <= '0; cj <= '0; sub_r <= '0;
      rowbase <= '0; stride <= '0;
    end else if (load) begin
      active  <= !load_empty;
      cm <= '0; ch <= '0; cj <= '0; sub_r <= '0;
      rowbase <= base_in;
      stride  <= stride_in;
    end else if (advance && active) begin
      if (last) begin
        active <= 1'b0;
      end else if (is_flat) begin
        cj <= cj + 16'd1;
      end else if (is_norm) begin
        if (j_last) begin
          cj <= '0;
          if (!sub_r[0]) begin
            sub_r <= 2'd1;
          end else begin
            sub_r   <= 2'd0;
            cm       <= cm + 8'd1;
            rowbase <= rowbase + stride;
          end
        end else begin
          cj <= cj + 16'd1;
        end
      end else begin // RoPE
        if (sub_r != 2'd2) begin
          sub_r <= sub_r + 2'd1;
        end else begin
          sub_r <= 2'd0;
          if (!j_last) begin
            cj <= cj + 16'd1;
          end else begin
            cj <= '0;
            if (ch != H - 8'd1) begin
              ch <= ch + 8'd1;
            end else begin
              ch       <= '0;
              cm       <= cm + 8'd1;
              rowbase <= rowbase + stride;
            end
          end
        end
      end
    end
  end
endmodule

// ---------------------------------------------------------------------------------------
// read-port line splitter. A 32-B step at the last 16-B slot of a 256-B line is
// issued as two 16-B pieces (addr, addr + 16); the walker advances only when the
// last piece is granted, and the two response halves are merged into one 256-bit
// FIFO entry. Responses return exactly one cycle after the grant, so a
// registered copy of the granted piece's role tags the arriving data.
// ---------------------------------------------------------------------------------------
module vec_engine_rdsplit
  import llaccel_pkg::*;
(
  input  logic         clk,
  input  logic         rst_n,
  input  logic         clear,        // new instruction: no request is in flight
  input  logic         req,          // the current step wants a read (active/credit already applied)
  input  logic [23:0]  addr,         // step address, 16-B aligned
  input  sram_size_e   size,         // SZ_16 or SZ_32
  output logic         step_grant,   // the whole step has been granted this cycle
  output logic         port_valid,
  output sram_req_t    port_req,
  input  logic         port_grant,
  input  logic         port_rvalid,
  input  logic [255:0] port_rdata,   // low 32 B of the port's right-aligned response
  output logic         push,         // one complete step response is available on din
  output logic [255:0] din
);
  logic         split, ph, last, rsp_first, rsp_second;
  logic [127:0] hold;

  assign split      = (size == SZ_32) && (addr[7:4] == 4'hF);
  assign last       = !split || ph;
  assign port_valid = req;
  assign port_req   = '{addr: split ? addr + (ph ? 24'd16 : 24'd0) : addr, size: split ? SZ_16 : size};
  assign step_grant = req && port_grant && last;
  assign push       = port_rvalid && !rsp_first;
  assign din        = rsp_second ? {port_rdata[127:0], hold} : port_rdata;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ph <= 1'b0; rsp_first <= 1'b0; rsp_second <= 1'b0; hold <= '0;
    end else begin
      rsp_first  <= req && port_grant && split && !ph;
      rsp_second <= req && port_grant && split && ph;
      if (clear)                  ph <= 1'b0;
      else if (req && port_grant) ph <= split && !ph;
      if (port_rvalid && rsp_first) hold <= port_rdata[127:0];
    end
  end
endmodule

// ---------------------------------------------------------------------------------------
// Small response FIFO (registered storage, combinational head).
// ---------------------------------------------------------------------------------------
module vec_engine_fifo #(
  parameter int W = 256,
  parameter int DEPTH = 4
) (
  input  logic                     clk,
  input  logic                     rst_n,
  input  logic                     flush,
  input  logic                     push,
  input  logic [W-1:0]             din,
  input  logic                     pop,
  output logic [W-1:0]             dout,
  output logic                     empty,
  output logic [$clog2(DEPTH+1)-1:0] count
);
  localparam int PW = $clog2(DEPTH);
  logic [W-1:0] mem [DEPTH];
  logic [PW-1:0] wp, rp;

  assign dout  = mem[rp];
  assign empty = (count == '0);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wp <= '0; rp <= '0; count <= '0;
    end else if (flush) begin
      wp <= '0; rp <= '0; count <= '0;
    end else begin
      if (push) wp <= wp + PW'(1);
      if (pop)  rp <= rp + PW'(1);
      count <= count + ($clog2(DEPTH+1))'(push) - ($clog2(DEPTH+1))'(pop);
    end
  end

  always_ff @(posedge clk) begin
    if (push) mem[wp] <= din;
  end
endmodule

// ---------------------------------------------------------------------------------------
// vector engine
// ---------------------------------------------------------------------------------------
module vec_engine
  import llaccel_pkg::*;
  import llaccel_luts_pkg::*;
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
  localparam int LANES = VEC_LANES;   // 16
  localparam int FIFO_DEPTH = 4;

  // ---- helpers -------------------------------------------------------------------------
  function automatic logic signed [63:0] sx16(input logic [15:0] v);
    return {{48{v[15]}}, v};
  endfunction
  function automatic logic signed [63:0] sx32(input logic [31:0] v);
    return {{32{v[31]}}, v};
  endfunction
  function automatic logic signed [47:0] sx16_48(input logic [15:0] v);
    return {{32{v[15]}}, v};
  endfunction
  // i16 x i16 -> i32
  function automatic logic signed [31:0] mul16(input logic [15:0] a, input logic [15:0] b);
    return 32'($signed(a)) * 32'($signed(b));
  endfunction

  // SiLU of one lane (NUMERICS.md "SiLU"); SIGMOID_LUT[idx] and [idx+1] are two ROM reads.
  function automatic logic [15:0] silu_lane(input logic [15:0] x, input logic [31:0] mi,
                                            input logic [5:0] si, input logic [5:0] sho);
    logic [15:0] u;
    logic [8:0]  idx;
    logic [7:0]  f;
    logic [15:0] l0, l1;
    logic [23:0] d;
    logic [15:0] sg;
    logic signed [63:0] prod;
    u   = sat16(mulshift64(sx16_48(x), mi, si));
    idx = {1'b0, ~u[15], u[14:8]};                 // (u >>> 8) + 128, in [0, 255]
    f   = u[7:0];
    l0  = SIGMOID_LUT[idx];
    l1  = SIGMOID_LUT[idx + 9'd1];
    d   = (24'(l1) - 24'(l0)) * 24'(f);            // sigmoid is monotone: l1 >= l0
    sg  = l0 + 16'(d >> 8);                        // <= 65514
    prod = sx16(x) * $signed({48'd0, sg});
    return sat16(rshr64(prod, sho));
  endfunction

  // ---- instruction state -----------------------------------------------------------------
  typedef enum logic [1:0] { S_IDLE, S_RUN, S_DONE } state_e;
  state_e state;

  logic [7:0]  op_r, sig_r;
  logic [31:0] p0_r, p1_r, p2_r;           // op-specific scalar params (see decode)
  logic [7:0]  M_r, H_r;
  logic [8:0]  D_r;
  logic [15:0] K16_r, nsteps_r;
  logic        is_norm, is_rope, is_silu, is_mul, is_add, is_quant;

  assign is_norm  = (op_r == 8'(OP_VEC_RMSNORM));
  assign is_rope  = (op_r == 8'(OP_VEC_ROPE));
  assign is_silu  = (op_r == 8'(OP_VEC_SILU));
  assign is_mul   = (op_r == 8'(OP_VEC_MUL));
  assign is_add   = (op_r == 8'(OP_VEC_ADD));
  assign is_quant = (op_r == 8'(OP_VEC_QUANT));

  // Decode of the incoming instruction (combinational, used only in the accept cycle).
  logic [7:0]  d_op;
  logic [31:0] w2, w3, w4, w5, w6, w7, w8, w9;
  logic        d_norm, d_rope, d_two_src;
  logic [23:0] d_base0, d_base1, d_basew, d_stride0, d_stride1, d_stridew;
  logic        d_empty;
  logic [7:0]  d_M, d_H;
  logic [8:0]  d_D;
  logic [15:0] d_K16, d_nsteps;
  logic [31:0] d_tab_row0;

  assign d_op = instr_opcode(instr);
  assign w2 = instr[2]; assign w3 = instr[3]; assign w4 = instr[4]; assign w5 = instr[5];
  assign w6 = instr[6]; assign w7 = instr[7]; assign w8 = instr[8]; assign w9 = instr[9];
  assign d_norm    = (d_op == 8'(OP_VEC_RMSNORM));
  assign d_rope    = (d_op == 8'(OP_VEC_ROPE));
  assign d_two_src = (d_op == 8'(OP_VEC_MUL)) || (d_op == 8'(OP_VEC_ADD));

  always_comb begin
    d_M = 8'd0; d_H = 8'd0; d_D = 9'd0; d_K16 = 16'd0; d_nsteps = 16'd0;
    d_base0 = w2[23:0]; d_base1 = w3[23:0]; d_basew = w3[23:0];
    d_stride0 = '0; d_stride1 = '0; d_stridew = '0;
    d_tab_row0 = w7 + pos * w8;                            // table + POS * table_stride
    d_empty = 1'b1;
    if (d_norm) begin
      d_basew   = w4[23:0];
      d_M       = w5[7:0];
      d_K16     = w6[19:4];
      d_stride0 = {w6[22:0], 1'b0};                        // 2K bytes per row
      d_stridew = {w6[22:0], 1'b0};
      d_empty   = (w5[7:0] == 8'd0) || (w6[19:4] == 16'd0);
    end else if (d_rope) begin
      d_M       = w4[7:0];
      d_H       = w5[7:0];
      d_D       = w6[8:0];
      d_base1   = d_tab_row0[23:0];
      d_stride1 = w8[23:0];
      d_stride0 = (24'(w5[7:0]) * 24'(w6[8:0])) << 1;             // 2*H*D bytes per row
      d_stridew = (24'(w5[7:0]) * 24'(w6[8:0])) << 1;
      d_empty   = (w4[7:0] == 8'd0) || (w5[7:0] == 8'd0);
    end else begin
      d_basew   = d_two_src ? w4[23:0] : w3[23:0];
      d_nsteps  = w4[19:4];
      if (d_two_src) d_nsteps = w5[19:4];
      d_empty   = (d_nsteps == 16'd0);
    end
  end

  logic accept;
  assign instr_ready = (state == S_IDLE);
  assign accept      = instr_ready && instr_valid;
  assign busy        = (state != S_IDLE);
  assign done_pulse  = (state == S_DONE);
  assign done_sig_sem = sig_r;
  assign perf_busy   = busy;

  // ---- walkers ---------------------------------------------------------------------------
  logic w0_active, w0_en_rd0, w0_adv;
  logic [23:0] w0_addr;
  sram_size_e  w0_size;
  logic w1_active, w1_en_rd1, w1_adv;
  logic [23:0] w1_addr;
  sram_size_e  w1_size;
  logic wc_active, wc_en_rd0, wc_en_rd1, wc_en_wr, wc_j_last, wc_adv;
  logic [1:0]  wc_sub;
  logic [23:0] wc_addr;
  sram_size_e  wc_size;

  /* verilator lint_off PINCONNECTEMPTY */
  vec_engine_walker #(.ROLE(0)) u_w0 (
    .clk(clk), .rst_n(rst_n), .load(accept), .load_empty(d_empty), .base_in(d_base0), .stride_in(d_stride0),
    .op(op_r), .M(M_r), .H(H_r), .D(D_r), .K16(K16_r), .nsteps(nsteps_r), .advance(w0_adv),
    .active(w0_active), .en_rd0(w0_en_rd0), .en_rd1(), .en_wr(), .sub(), .j_last(), .addr(w0_addr), .size(w0_size));
  vec_engine_walker #(.ROLE(1)) u_w1 (
    .clk(clk), .rst_n(rst_n), .load(accept), .load_empty(d_empty), .base_in(d_base1), .stride_in(d_stride1),
    .op(op_r), .M(M_r), .H(H_r), .D(D_r), .K16(K16_r), .nsteps(nsteps_r), .advance(w1_adv),
    .active(w1_active), .en_rd0(), .en_rd1(w1_en_rd1), .en_wr(), .sub(), .j_last(), .addr(w1_addr), .size(w1_size));
  vec_engine_walker #(.ROLE(2)) u_wc (
    .clk(clk), .rst_n(rst_n), .load(accept), .load_empty(d_empty), .base_in(d_basew), .stride_in(d_stridew),
    .op(op_r), .M(M_r), .H(H_r), .D(D_r), .K16(K16_r), .nsteps(nsteps_r), .advance(wc_adv),
    .active(wc_active), .en_rd0(wc_en_rd0), .en_rd1(wc_en_rd1), .en_wr(wc_en_wr), .sub(wc_sub), .j_last(wc_j_last),
    .addr(wc_addr), .size(wc_size));
  /* verilator lint_on PINCONNECTEMPTY */

  // ---- response FIFOs, line splitters and read issue -----------------------------------------
  logic [255:0] f0_dout, f1_dout, f0_din, f1_din;
  logic f0_empty, f1_empty, f0_pop, f1_pop, f0_push, f1_push;
  logic [2:0] f0_count, f1_count;
  logic credit0, credit1, rd0_req, rd1_req, rd0_step, rd1_step;

  vec_engine_fifo #(.W(256), .DEPTH(FIFO_DEPTH)) u_f0 (
    .clk(clk), .rst_n(rst_n), .flush(accept), .push(f0_push), .din(f0_din),
    .pop(f0_pop), .dout(f0_dout), .empty(f0_empty), .count(f0_count));
  vec_engine_fifo #(.W(256), .DEPTH(FIFO_DEPTH)) u_f1 (
    .clk(clk), .rst_n(rst_n), .flush(accept), .push(f1_push), .din(f1_din),
    .pop(f1_pop), .dout(f1_dout), .empty(f1_empty), .count(f1_count));

  // A granted read returns next cycle, so the only response in flight is the
  // one whose rvalid is high now (not yet counted by the FIFO). Counting the
  // first half of a split step as a pending push is conservative. A request
  // that is presented and not granted keeps its credit (the count can only
  // grow through an rvalid, which needs a grant), so it is held as required.
  assign credit0 = (4'(f0_count) + 4'(vec_rd0_rvalid)) < 4'(FIFO_DEPTH);
  assign credit1 = (4'(f1_count) + 4'(vec_rd1_rvalid)) < 4'(FIFO_DEPTH);

  assign rd0_req = (state == S_RUN) && w0_active && w0_en_rd0 && credit0;
  assign rd1_req = (state == S_RUN) && w1_active && w1_en_rd1 && credit1;
  assign w0_adv  = (state == S_RUN) && w0_active && (!w0_en_rd0 || rd0_step);
  assign w1_adv  = (state == S_RUN) && w1_active && (!w1_en_rd1 || rd1_step);

  vec_engine_rdsplit u_s0 (
    .clk(clk), .rst_n(rst_n), .clear(accept), .req(rd0_req), .addr(w0_addr), .size(w0_size), .step_grant(rd0_step),
    .port_valid(vec_rd0_valid), .port_req(vec_rd0_req), .port_grant(vec_rd0_grant),
    .port_rvalid(vec_rd0_rvalid), .port_rdata(vec_rd0_rdata[255:0]), .push(f0_push), .din(f0_din));
  vec_engine_rdsplit u_s1 (
    .clk(clk), .rst_n(rst_n), .clear(accept), .req(rd1_req), .addr(w1_addr), .size(w1_size), .step_grant(rd1_step),
    .port_valid(vec_rd1_valid), .port_req(vec_rd1_req), .port_grant(vec_rd1_grant),
    .port_rvalid(vec_rd1_rvalid), .port_rdata(vec_rd1_rdata[255:0]), .push(f1_push), .din(f1_din));

  // ---- RMSNorm scalar path ------------------------------------------------------------------------
  typedef enum logic [2:0] { N_IDLE, N_SQRT_START, N_SQRT_WAIT, N_DIV_START, N_DIV_WAIT, N_READY } norm_e;
  norm_e norm_state;
  logic [47:0] ss;                 // sum of squares of the current row
  logic [23:0] r_r;                // isqrt result
  logic [15:0] inv_r;              // min(65535, C / max(r, 1))
  logic        isqrt_start, isqrt_busy, isqrt_done;
  logic [47:0] isqrt_a;
  logic [23:0] isqrt_q;
  logic        udiv_start, udiv_busy, udiv_done;
  logic [23:0] udiv_b;
  logic [31:0] udiv_q;

  assign isqrt_start = (norm_state == N_SQRT_START);
  assign isqrt_a     = ss + {16'd0, p0_r};                     // ss + eps_t
  assign udiv_start  = (norm_state == N_DIV_START);
  assign udiv_b      = (r_r == 24'd0) ? 24'd1 : r_r;           // max(r, 1)

  isqrt u_isqrt (.clk(clk), .rst_n(rst_n), .start(isqrt_start), .a(isqrt_a), .busy(isqrt_busy), .done(isqrt_done), .q(isqrt_q));
  udiv #(.AW(32), .BW(24)) u_udiv (.clk(clk), .rst_n(rst_n), .start(udiv_start), .a(p1_r), .b(udiv_b),
                                   .busy(udiv_busy), .done(udiv_done), .q(udiv_q));

  // ---- write holding register and line splitter ---------------------------------------------------
  logic        wr_valid_r;
  logic [23:0] wr_addr_r;
  sram_size_e  wr_size_r;
  logic [255:0] wr_data_r;
  logic        wr_cross, wr_ph, wr_last, wr_done;

  assign wr_cross = (wr_size_r == SZ_32) && (wr_addr_r[7:4] == 4'hF);
  assign wr_last  = !wr_cross || wr_ph;
  assign wr_done  = vec_wr_grant && wr_last;                   // the holding register frees this cycle

  assign vec_wr_valid = wr_valid_r;
  assign vec_wr_req   = '{addr: wr_cross ? wr_addr_r + (wr_ph ? 24'd16 : 24'd0) : wr_addr_r,
                          size: wr_cross ? SZ_16 : wr_size_r};
  assign vec_wr_wdata = (wr_cross && wr_ph) ? {384'd0, wr_data_r[255:128]} : {256'd0, wr_data_r};
  assign vec_wr_wstrb = (vec_wr_req.size == SZ_16) ? 64'h0000_0000_0000_FFFF : 64'h0000_0000_FFFF_FFFF;

  // ---- compute stage --------------------------------------------------------------------------------
  logic [255:0] x1_r, c_r, y2_r;   // RoPE: saved x1 / cos from sub 0, y2 from sub 1

  logic need0, need1, norm_ok, wr_ok, do_step;
  assign need0   = wc_en_rd0;
  assign need1   = wc_en_rd1;
  assign norm_ok = !(is_norm && wc_sub[0]) || (norm_state == N_READY);
  assign wr_ok   = !wc_en_wr || !wr_valid_r || wr_done;
  assign do_step = (state == S_RUN) && wc_active && (!need0 || !f0_empty) && (!need1 || !f1_empty) && wr_ok && norm_ok;
  assign f0_pop  = do_step && need0;
  assign f1_pop  = do_step && need1;
  assign wc_adv  = do_step;

  // Lane arithmetic (combinational from the FIFO heads and saved registers).
  logic [255:0] y_vec;      // i16 results
  logic [255:0] y2_vec;     // RoPE second half
  logic [127:0] q_vec;      // i8 results (QUANT)
  logic [47:0]  sq_sum;     // sum of 16 squares (<= 2^34)

  always_comb begin
    y_vec  = '0;
    y2_vec = '0;
    q_vec  = '0;
    sq_sum = '0;
    for (int l = 0; l < LANES; l++) begin
      logic [15:0] xa, xb, x1, cc;
      logic signed [63:0] t;
      logic signed [31:0] xg;
      logic signed [63:0] xgi;
      logic signed [63:0] r1, r2;
      xg = '0; xgi = '0; r1 = '0; r2 = '0;
      xa = f0_dout[16*l +: 16];
      xb = f1_dout[16*l +: 16];
      x1 = x1_r[16*l +: 16];
      cc = c_r[16*l +: 16];
      // sum of squares (RMSNorm pass 0)
      sq_sum = sq_sum + 48'(32'(mul16(xa, xa)));
      // per-op result
      t = 64'sd0;
      if (is_silu) begin
        y_vec[16*l +: 16] = silu_lane(xa, p0_r, p1_r[5:0], p2_r[5:0]);
      end else if (is_mul) begin
        t = sx32(mul16(xa, xb));
        y_vec[16*l +: 16] = sat16(rshr64(t, p0_r[5:0]));
      end else if (is_add) begin
        t = sx16(xa) + rshr64(sx16(xb), p0_r[5:0]);
        y_vec[16*l +: 16] = sat16(t);
      end else if (is_quant) begin
        q_vec[8*l +: 8] = sat8(mulshift64(sx16_48(xa), p0_r, p1_r[5:0]));
      end else if (is_norm) begin
        xg  = mul16(xa, xb);                                   // x * g : i32
        xgi = sx32(xg) * $signed({48'd0, inv_r});              // * inv : i48
        y_vec[16*l +: 16] = sat16(rshr64(xgi, p2_r[5:0]));     // sh_post
      end else begin // RoPE (sub 1): xa = x2, xb = sin, x1/cc saved from sub 0
        r1 = sx32(mul16(x1, cc)) - sx32(mul16(xa, xb));        // x1*c - x2*s
        r2 = sx32(mul16(xa, cc)) + sx32(mul16(x1, xb));        // x2*c + x1*s
        y_vec[16*l +: 16]  = sat16(rshr64(r1, 6'd14));
        y2_vec[16*l +: 16] = sat16(rshr64(r2, 6'd14));
      end
    end
  end

  // ---- sequential control ---------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      op_r <= '0; sig_r <= '0; p0_r <= '0; p1_r <= '0; p2_r <= '0;
      M_r <= '0; H_r <= '0; D_r <= '0; K16_r <= '0; nsteps_r <= '0;
      norm_state <= N_IDLE; ss <= '0; r_r <= '0; inv_r <= '0;
      wr_valid_r <= 1'b0; wr_addr_r <= '0; wr_size_r <= SZ_32; wr_data_r <= '0; wr_ph <= 1'b0;
      x1_r <= '0; c_r <= '0; y2_r <= '0;
    end else begin
`ifndef SYNTHESIS
      // simulation-only contract checks (never evaluated in reset)
      if (accept) begin
        assert (d_op == 8'(OP_VEC_RMSNORM) || d_op == 8'(OP_VEC_ROPE) || d_op == 8'(OP_VEC_SILU) ||
                d_op == 8'(OP_VEC_MUL) || d_op == 8'(OP_VEC_ADD) || d_op == 8'(OP_VEC_QUANT))
          else $error("vec_engine: unsupported opcode %h", d_op);
        if (d_rope) assert (w6 == 32'd16 || w6 == 32'd32 || w6 == 32'd64 ||
                            w6 == 32'd128 || w6 == 32'd256)
          else $error("vec_engine: RoPE D must be 16/32/64/128/256 (got %0d)", w6);
        if (d_norm) assert (w6[3:0] == 4'd0) else $error("vec_engine: RMSNorm K must be a multiple of 16");
        if (!d_norm && !d_rope) assert ((d_two_src ? w5[3:0] : w4[3:0]) == 4'd0)
          else $error("vec_engine: count must be a multiple of 16");
      end
      if (vec_rd0_valid) assert (vec_rd0_req.addr[3:0] == 4'd0 && (32'(vec_rd0_req.addr[7:0]) + size_bytes(vec_rd0_req.size)) <= 32'd256)
        else $error("vec_engine: rd0 request violates alignment/line rule");
      if (vec_rd1_valid) assert (vec_rd1_req.addr[3:0] == 4'd0 && (32'(vec_rd1_req.addr[7:0]) + size_bytes(vec_rd1_req.size)) <= 32'd256)
        else $error("vec_engine: rd1 request violates alignment/line rule");
      if (vec_wr_valid) assert (vec_wr_req.addr[3:0] == 4'd0 && (32'(vec_wr_req.addr[7:0]) + size_bytes(vec_wr_req.size)) <= 32'd256)
        else $error("vec_engine: wr request violates alignment/line rule");
`endif
      case (state)
        S_IDLE: begin
          if (accept) begin
            state    <= S_RUN;
            op_r     <= d_op;
            sig_r    <= instr_sig_sem(instr);
            M_r      <= d_M;
            H_r      <= d_H;
            D_r      <= d_D;
            K16_r    <= d_K16;
            nsteps_r <= d_nsteps;
            // scalar params: RMSNORM {eps_t, C, sh_post}; SILU {Mi, Si, sh_out};
            // MUL {sh}; ADD {sh_b}; QUANT {M, S}
            if (d_norm)          begin p0_r <= w7; p1_r <= w8; p2_r <= w9; end
            else if (d_two_src)  begin p0_r <= w6; p1_r <= '0; p2_r <= '0; end
            else                 begin p0_r <= w5; p1_r <= w6; p2_r <= w7; end
            norm_state <= N_IDLE;
            ss         <= '0;
            wr_valid_r <= 1'b0;
            wr_ph      <= 1'b0;
          end
        end
        S_RUN: begin
          // write holding register (a split write frees only when its second piece is granted)
          if (vec_wr_grant && wr_cross && !wr_ph) wr_ph <= 1'b1;
          if (do_step && wc_en_wr) begin
            wr_valid_r <= 1'b1;
            wr_addr_r  <= wc_addr;
            wr_size_r  <= wc_size;
            wr_ph      <= 1'b0;
            if (is_quant)                   wr_data_r <= {128'd0, q_vec};
            else if (is_rope && wc_sub == 2'd2) wr_data_r <= y2_r;
            else                            wr_data_r <= y_vec;
          end else if (wr_done) begin
            wr_valid_r <= 1'b0;
          end
          // RoPE saved operands
          if (do_step && is_rope) begin
            if (wc_sub == 2'd0) begin x1_r <= f0_dout; c_r <= f1_dout; end
            if (wc_sub == 2'd1) y2_r <= y2_vec;
          end
          // RMSNorm scalar sequence
          case (norm_state)
            N_IDLE: if (do_step && is_norm && !wc_sub[0]) begin
              ss <= ss + sq_sum;
              if (wc_j_last) norm_state <= N_SQRT_START;
            end
            N_SQRT_START: norm_state <= N_SQRT_WAIT;
            N_SQRT_WAIT: if (isqrt_done) begin r_r <= isqrt_q; norm_state <= N_DIV_START; end
            N_DIV_START: norm_state <= N_DIV_WAIT;
            N_DIV_WAIT: if (udiv_done) begin
              inv_r <= (udiv_q > 32'd65535) ? 16'd65535 : udiv_q[15:0];
              norm_state <= N_READY;
            end
            N_READY: if (do_step && is_norm && wc_j_last) begin
              norm_state <= N_IDLE;
              ss <= '0;
            end
            default: norm_state <= N_IDLE;
          endcase
          // retire when every walker is exhausted and the last write has been granted
          if (!w0_active && !w1_active && !wc_active && !wr_valid_r) state <= S_DONE;
        end
        S_DONE: state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end

  assign perf_sram_stall = (vec_rd0_valid && !vec_rd0_grant) || (vec_rd1_valid && !vec_rd1_grant) ||
                           (vec_wr_valid && !vec_wr_grant);

  // isqrt/udiv busy flags are implied by norm_state; keep the ports connected for lint.
  logic unused_ok;
  assign unused_ok = isqrt_busy | udiv_busy | (|vec_rd0_rdata[511:256]) | (|vec_rd1_rdata[511:256]) |
                     (|instr[1]) | (|instr[0][23:8]) | (|w2[31:24]) | (|w3[31:24]) | (|w4[31:20]) |
                     (|w5[31:8]) | (|w6[31:23]) | (|w7[31:24]) | (|w8[31:24]) | (|w9[31:6]) | (|p2_r[31:6]) |
                     (|d_tab_row0[31:24]) | (|instr[15:10]);

endmodule
