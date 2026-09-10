// attn_engine.sv — causal GQA attention over the INT8 KV cache, and KV_WRITE.
//
// ATTN, per (m, h) with P = POS + m, T = P + 1, kvh = h / (H / Hkv):
//   S_Q    read the D-byte q slice (one SZ_16/32/64 read).
//   S_K    stream T key rows (one read per key, issued back to back); each
//          returned row goes through the 64-wide i8 dot product (products +
//          adder tree, registered) and is written to the score buffer while
//          the running max is updated. perf_mac_cycles pulses per row.
//   S_P2   walk the score buffer: z = mulshift(mx - s, Ms, Ss), p from the two
//          exp LUTs, write p back over s, accumulate the sum (3-stage pipeline:
//          buffer read, z, p).
//   S_DIV  inv = floor(2^31 / sum) on the 32/32 restoring divider (udiv.sv).
//   S_V    stream T value rows; pn[t] = satu8((p[t]*inv + 2^22) >> 23) is
//          computed when the row arrives (p[t] is read from the buffer in
//          lockstep with the request) and 64 u8×i8 MACs accumulate o[d].
//   S_OUT  requantize o[d] eight lanes per cycle (sat8(mulshift(o, Mo, So))),
//          then push the D-byte row into the write queue; the next head starts
//          while the write drains.
// KV_WRITE copies M*Hkv rows of D bytes through the same write queue (read
// issue is credit-limited by the queue depth).
//
// Every intermediate follows numerics.h::attention_head exactly via the
// llaccel_pkg helpers (mulshift64, sat8, satu8).

/* verilator lint_off DECLFILENAME */  // private helper module lives in the engine's file

// ---------------------------------------------------------------------------------------
// Write queue: {addr, 64 B data} entries, registered storage, combinational head.
// ---------------------------------------------------------------------------------------
module attn_engine_wfifo #(
  parameter int DEPTH = 2
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         flush,
  input  logic         push,
  input  logic [23:0]  din_addr,
  input  logic [511:0] din_data,
  input  logic         pop,
  output logic [23:0]  dout_addr,
  output logic [511:0] dout_data,
  output logic         empty,
  output logic         full,
  output logic [$clog2(DEPTH+1)-1:0] count
);
  localparam int PW = (DEPTH > 1) ? $clog2(DEPTH) : 1;
  localparam int CW = $clog2(DEPTH+1);
  logic [23:0]  mem_addr [DEPTH];
  logic [511:0] mem_data [DEPTH];
  logic [PW-1:0] wp, rp;

  assign dout_addr = mem_addr[rp];
  assign dout_data = mem_data[rp];
  assign empty = (count == '0);
  assign full  = (count == CW'(DEPTH));

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wp <= '0; rp <= '0; count <= '0;
    end else if (flush) begin
      wp <= '0; rp <= '0; count <= '0;
    end else begin
      if (push) wp <= (wp == PW'(DEPTH-1)) ? '0 : wp + PW'(1);
      if (pop)  rp <= (rp == PW'(DEPTH-1)) ? '0 : rp + PW'(1);
      count <= count + CW'(push) - CW'(pop);
    end
  end

  always_ff @(posedge clk) begin
    if (push) begin
      mem_addr[wp] <= din_addr;
      mem_data[wp] <= din_data;
    end
  end
endmodule

// ---------------------------------------------------------------------------------------
// Attention engine
// ---------------------------------------------------------------------------------------
module attn_engine (
  input  logic                      clk,
  input  logic                      rst_n,
  input  logic                      instr_valid,
  input  llaccel_pkg::instr_words_t instr,
  output logic                      instr_ready,
  input  logic [31:0]               pos,
  output logic                      busy,
  output logic                      done_pulse,
  output logic [7:0]                done_sig_sem,
  output logic                      attn_rd_valid,
  output llaccel_pkg::sram_req_t    attn_rd_req,
  input  logic                      attn_rd_grant,
  input  logic                      attn_rd_rvalid,
  input  logic [511:0]              attn_rd_rdata,
  output logic                      attn_wr_valid,
  output llaccel_pkg::sram_req_t    attn_wr_req,
  output logic [511:0]              attn_wr_wdata,
  output logic [63:0]               attn_wr_wstrb,
  input  logic                      attn_wr_grant,
  output logic                      perf_busy,
  output logic                      perf_sram_stall,
  output logic                      perf_mac_cycles
);
  import llaccel_pkg::*;

  // The generated LUT file also defines the sigmoid table, unused here.
  /* verilator lint_off UNUSEDPARAM */
  `include "llaccel_luts.svh"
  /* verilator lint_on UNUSEDPARAM */

  localparam int LANES = ATTN_LANES;     // 64
  localparam int TMAX  = ATTN_TMAX;      // 256
  localparam int WQ_DEPTH = 2;
  localparam int OUT_LANES = 8;          // requant lanes per cycle

  // ---- helpers -------------------------------------------------------------------------
  function automatic logic signed [47:0] sx32_48(input logic [31:0] v);
    return {{16{v[31]}}, v};
  endfunction
  function automatic logic signed [31:0] sx8_32(input logic [7:0] v);
    return {{24{v[7]}}, v};
  endfunction
  function automatic sram_size_e size_of_d(input logic [6:0] d);
    if (d == 7'd16) return SZ_16;
    if (d == 7'd32) return SZ_32;
    return SZ_64;
  endfunction
  function automatic logic [63:0] strb_of_d(input logic [6:0] d);
    if (d == 7'd16) return 64'h0000_0000_0000_FFFF;
    if (d == 7'd32) return 64'h0000_0000_FFFF_FFFF;
    return 64'hFFFF_FFFF_FFFF_FFFF;
  endfunction

  // ---- state ------------------------------------------------------------------------------
  typedef enum logic [3:0] {
    S_IDLE, S_Q, S_K, S_P2, S_DIV_START, S_DIV_WAIT, S_V, S_OUT, S_PUSH, S_KV, S_DRAIN, S_DONE
  } state_e;
  state_e state;

  logic        is_attn;                 // else KV_WRITE
  logic [7:0]  sig_r;
  logic [23:0] out_r, kbase_r, vbase_r, kvs_r;
  logic [7:0]  M_r, H_r, Hkv_r, G_r;
  logic [6:0]  D_r;
  logic [23:0] HD_r;                    // H * D bytes (q / out row stride)
  logic [31:0] Ms_r, Mo_r;
  logic [5:0]  Ss_r, So_r;
  sram_size_e  sz_r;
  logic [63:0] strb_r;

  // ATTN loop bookkeeping
  logic [7:0]  cm, ch, hg;              // row, head, index within the GQA group
  logic [8:0]  T_r;                     // keys for the current row (POS + m + 1)
  logic [23:0] qrow, orow, hoff, kkv, vkv;
  logic        q_issued;
  logic [8:0]  t_iss, t_rsp, t_wr, t_a, t_acc, p_cnt;
  logic [23:0] row_addr;
  logic [511:0] qv;                     // q lanes (zero beyond D)
  // score pipeline
  logic signed [31:0] s_reg, mx;
  logic        s_valid;
  logic [8:0]  s_t;
  logic [31:0] sbuf [TMAX];             // scores, then probabilities
  logic [31:0] s_rd;                    // synchronous read data
  logic [7:0]  sbuf_raddr;
  // softmax pipeline
  logic        va, vz;
  logic [8:0]  ta_q, tz_q;
  logic        z_big;                   // z >= 4096
  logic [11:0] z_lo;
  logic [31:0] sum;
  logic [31:0] inv_r;
  // value pipeline
  logic [7:0]  pn_q;
  logic [511:0] v_q;
  logic        vv;
  logic signed [31:0] o [LANES];
  // output requant
  logic [2:0]  og;
  logic [511:0] wdata_acc;
  // KV_WRITE bookkeeping
  logic [15:0] r_iss, R_r;
  logic [7:0]  kvh;
  logic [23:0] src_addr, dst_m, dst_cur, dst_pending;

  // ---- decode ---------------------------------------------------------------------------------
  logic [7:0]  d_op;
  logic        d_attn;
  logic [31:0] w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14;
  logic [7:0]  d_G;
  logic [23:0] d_HD, d_posD;
  logic [15:0] d_R;

  assign d_op = instr_opcode(instr);
  assign d_attn = (d_op == 8'(OP_ATTN));
  assign w2 = instr[2]; assign w3 = instr[3]; assign w4 = instr[4]; assign w5 = instr[5]; assign w6 = instr[6];
  assign w7 = instr[7]; assign w8 = instr[8]; assign w9 = instr[9]; assign w10 = instr[10]; assign w11 = instr[11];
  assign w12 = instr[12]; assign w13 = instr[13]; assign w14 = instr[14];
  assign d_G    = (w8[7:0] == 8'd0) ? 8'd1 : (w7[7:0] / w8[7:0]);          // H / Hkv (ATTN)
  assign d_HD   = 24'(w7[7:0] * w9[6:0]);                                  // H * D (ATTN)
  assign d_posD = 24'(pos[15:0] * w6[6:0]);                                // POS * D (KV_WRITE)
  assign d_R    = 16'(w4[7:0] * w5[7:0]);                                  // M * Hkv (KV_WRITE)

  logic accept;
  assign instr_ready  = (state == S_IDLE);
  assign accept       = instr_ready && instr_valid;
  assign busy         = (state != S_IDLE);
  assign done_pulse   = (state == S_DONE);
  assign done_sig_sem = sig_r;
  assign perf_busy    = busy;

  // ---- write queue ------------------------------------------------------------------------------
  logic        wq_push, wq_pop, wq_empty, wq_full;
  logic [23:0] wq_din_addr, wq_dout_addr;
  logic [511:0] wq_din_data, wq_dout_data;
  logic [1:0]  wq_count;

  attn_engine_wfifo #(.DEPTH(WQ_DEPTH)) u_wq (
    .clk(clk), .rst_n(rst_n), .flush(accept), .push(wq_push), .din_addr(wq_din_addr), .din_data(wq_din_data),
    .pop(wq_pop), .dout_addr(wq_dout_addr), .dout_data(wq_dout_data), .empty(wq_empty), .full(wq_full), .count(wq_count));

  assign attn_wr_valid = !wq_empty;
  assign attn_wr_req   = '{addr: wq_dout_addr, size: sz_r};
  assign attn_wr_wdata = wq_dout_data;
  assign attn_wr_wstrb = strb_r;
  assign wq_pop        = attn_wr_valid && attn_wr_grant;

  // KV_WRITE: rows land in the queue the cycle after the grant, so credit = free slots minus that response.
  logic kv_credit;
  assign kv_credit = (3'(wq_count) + 3'(attn_rd_rvalid)) < 3'(WQ_DEPTH);

  always_comb begin
    wq_push     = 1'b0;
    wq_din_addr = dst_pending;
    wq_din_data = attn_rd_rdata;
    if (!is_attn) begin
      wq_push = (state == S_KV || state == S_DRAIN) && attn_rd_rvalid;
    end else if (state == S_PUSH) begin
      wq_push     = !wq_full;
      wq_din_addr = orow + hoff;
      wq_din_data = wdata_acc;
    end
  end

  // ---- read request mux -------------------------------------------------------------------------------
  logic [23:0] rd_addr;
  always_comb begin
    attn_rd_valid = 1'b0;
    rd_addr = row_addr;
    case (state)
      S_Q:  begin attn_rd_valid = !q_issued;                 rd_addr = qrow + hoff; end
      S_K:  begin attn_rd_valid = (t_iss != T_r);            rd_addr = row_addr;    end
      S_V:  begin attn_rd_valid = (t_iss != T_r);            rd_addr = row_addr;    end
      S_KV: begin attn_rd_valid = (r_iss != R_r) && kv_credit; rd_addr = src_addr;  end
      default: ;
    endcase
  end
  assign attn_rd_req = '{addr: rd_addr, size: sz_r};

  // ---- arithmetic ------------------------------------------------------------------------------------
  // i8 dot product of q and the arriving key row (64 products + adder tree).
  logic signed [31:0] dot;
  always_comb begin
    dot = 32'sd0;
    for (int d = 0; d < LANES; d++)
      dot = dot + (sx8_32(qv[8*d +: 8]) * sx8_32(attn_rd_rdata[8*d +: 8]));
  end

  // z = mulshift(mx - s, Ms, Ss); p = z >= 4096 ? 0 : (EXPI[z>>8] * EXPF[z&255] + 2^15) >> 16
  logic signed [31:0] diff;
  logic signed [63:0] z64;
  logic [15:0] p_val;
  logic [31:0] p_prod;
  always_comb begin
    diff = mx - s_rd;
    z64  = mulshift64(sx32_48(diff), Ms_r, Ss_r);
    p_prod = 32'(EXP_INT_LUT[z_lo[11:8]]) * 32'(EXP_FRAC_LUT[z_lo[7:0]]) + 32'd32768;
    p_val  = z_big ? 16'd0 : p_prod[31:16];
  end

  // pn = satu8((p * inv + 2^22) >> 23) for the arriving value row
  logic [63:0] pn_prod;
  logic [7:0]  pn;
  always_comb begin
    pn_prod = 64'(s_rd[15:0]) * 64'(inv_r) + 64'd4194304;
    pn = satu8($signed(pn_prod >> 23));
  end

  // output requant, OUT_LANES lanes per cycle
  logic [63:0] out8;
  always_comb begin
    for (int i = 0; i < OUT_LANES; i++)
      out8[8*i +: 8] = sat8(mulshift64(sx32_48(o[{og, 3'(i)}]), Mo_r, So_r));
  end

  // ---- score buffer (synchronous read, one write port) ---------------------------------------------
  assign sbuf_raddr = (state == S_V) ? t_iss[7:0] : t_a[7:0];
  always_ff @(posedge clk) begin
    s_rd <= sbuf[sbuf_raddr];
    if (state == S_K && s_valid) sbuf[s_t[7:0]]  <= s_reg;
    if (state == S_P2 && vz)     sbuf[tz_q[7:0]] <= {16'd0, p_val};
  end

  // ---- udiv: inv = floor(2^31 / sum) ------------------------------------------------------------------
  logic udiv_start, udiv_busy, udiv_done;
  logic [31:0] udiv_q;
  assign udiv_start = (state == S_DIV_START);
  udiv #(.AW(32), .BW(32)) u_udiv (.clk(clk), .rst_n(rst_n), .start(udiv_start), .a(32'h8000_0000), .b(sum),
                                   .busy(udiv_busy), .done(udiv_done), .q(udiv_q));

  // ---- main sequential control ----------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      is_attn <= 1'b0; sig_r <= '0;
      out_r <= '0; kbase_r <= '0; vbase_r <= '0; kvs_r <= '0;
      M_r <= '0; H_r <= '0; Hkv_r <= '0; G_r <= '0; D_r <= '0; HD_r <= '0;
      Ms_r <= '0; Mo_r <= '0; Ss_r <= '0; So_r <= '0; sz_r <= SZ_16; strb_r <= '0;
      cm <= '0; ch <= '0; hg <= '0; T_r <= '0;
      qrow <= '0; orow <= '0; hoff <= '0; kkv <= '0; vkv <= '0;
      q_issued <= 1'b0; t_iss <= '0; t_rsp <= '0; t_wr <= '0; t_a <= '0; t_acc <= '0; p_cnt <= '0;
      row_addr <= '0; qv <= '0;
      s_reg <= '0; mx <= '0; s_valid <= 1'b0; s_t <= '0;
      va <= 1'b0; vz <= 1'b0; ta_q <= '0; tz_q <= '0; z_big <= 1'b0; z_lo <= '0; sum <= '0; inv_r <= '0;
      pn_q <= '0; v_q <= '0; vv <= 1'b0;
      for (int d = 0; d < LANES; d++) o[d] <= '0;
      og <= '0; wdata_acc <= '0;
      r_iss <= '0; R_r <= '0; kvh <= '0; src_addr <= '0; dst_m <= '0; dst_cur <= '0; dst_pending <= '0;
    end else begin
`ifndef SYNTHESIS
      if (accept) begin
        assert (d_op == 8'(OP_ATTN) || d_op == 8'(OP_KV_WRITE)) else $error("attn_engine: unsupported opcode %h", d_op);
        if (d_attn) begin
          assert (w9[6:0] == 7'd16 || w9[6:0] == 7'd32 || w9[6:0] == 7'd64) else $error("attn_engine: D must be 16/32/64");
          assert (64'(pos) + 64'(w6) <= 64'(TMAX)) else $error("attn_engine: POS + M = %0d exceeds ATTN_TMAX", pos + w6);
          assert (w8[7:0] != 8'd0 && 16'(d_G) * 16'(w8[7:0]) == 16'(w7[7:0])) else $error("attn_engine: Hkv must divide H");
        end else begin
          assert (w6[6:0] == 7'd16 || w6[6:0] == 7'd32 || w6[6:0] == 7'd64) else $error("attn_engine: D must be 16/32/64");
        end
      end
      if (attn_rd_valid) assert (attn_rd_req.addr[3:0] == 4'd0 && (32'(attn_rd_req.addr[7:0]) + size_bytes(attn_rd_req.size)) <= 32'd256)
        else $error("attn_engine: rd request violates alignment/line rule");
      if (attn_wr_valid) assert (attn_wr_req.addr[3:0] == 4'd0 && (32'(attn_wr_req.addr[7:0]) + size_bytes(attn_wr_req.size)) <= 32'd256)
        else $error("attn_engine: wr request violates alignment/line rule");
`endif
      // pipeline valid defaults
      s_valid <= 1'b0;
      vv      <= 1'b0;

      case (state)
        // ------------------------------------------------------------------------------
        S_IDLE: if (accept) begin
          is_attn <= d_attn;
          sig_r   <= instr_sig_sem(instr);
          if (d_attn) begin
            out_r   <= w3[23:0];  kbase_r <= w4[23:0]; vbase_r <= w5[23:0];
            M_r     <= w6[7:0];   H_r     <= w7[7:0];  Hkv_r   <= w8[7:0];  G_r <= d_G;
            D_r     <= w9[6:0];   kvs_r   <= w10[23:0];
            Ms_r    <= w11;       Ss_r    <= w12[5:0]; Mo_r    <= w13;      So_r <= w14[5:0];
            HD_r    <= d_HD;
            sz_r    <= size_of_d(w9[6:0]);
            strb_r  <= strb_of_d(w9[6:0]);
            cm <= '0; ch <= '0; hg <= '0;
            T_r   <= 9'(pos[8:0]) + 9'd1;
            qrow  <= w2[23:0];
            orow  <= w3[23:0];
            hoff  <= '0;
            kkv   <= w4[23:0];
            vkv   <= w5[23:0];
            q_issued <= 1'b0;
            state <= (w6[7:0] == 8'd0 || w7[7:0] == 8'd0) ? S_DRAIN : S_Q;
          end else begin
            D_r     <= w6[6:0];
            kvs_r   <= w7[23:0];
            Hkv_r   <= w5[7:0];
            sz_r    <= size_of_d(w6[6:0]);
            strb_r  <= strb_of_d(w6[6:0]);
            R_r     <= d_R;
            r_iss   <= '0;
            kvh     <= '0;
            src_addr <= w2[23:0];
            dst_m   <= w3[23:0] + d_posD;
            dst_cur <= w3[23:0] + d_posD;
            state   <= (d_R == 16'd0) ? S_DRAIN : S_KV;
          end
        end
        // ------------------------------------------------------------------------------
        S_Q: begin
          if (attn_rd_valid && attn_rd_grant) q_issued <= 1'b1;
          if (attn_rd_rvalid) begin
            for (int d = 0; d < LANES; d++) qv[8*d +: 8] <= (d < 32'(D_r)) ? attn_rd_rdata[8*d +: 8] : 8'd0;
            q_issued <= 1'b0;
            t_iss    <= '0;
            t_rsp    <= '0;
            t_wr     <= '0;
            row_addr <= kkv;
            mx       <= 32'sh8000_0000;
            state    <= S_K;
          end
        end
        // ------------------------------------------------------------------------------
        S_K: begin
          if (attn_rd_valid && attn_rd_grant) begin
            t_iss    <= t_iss + 9'd1;
            row_addr <= row_addr + 24'(D_r);
          end
          if (attn_rd_rvalid) begin
            s_reg   <= dot;
            s_valid <= 1'b1;
            s_t     <= t_rsp;
            t_rsp   <= t_rsp + 9'd1;
          end
          if (s_valid) begin
            if (s_reg > mx) mx <= s_reg;
            t_wr <= t_wr + 9'd1;
          end
          if (t_wr == T_r) begin
            t_a   <= '0;
            va    <= 1'b0;
            vz    <= 1'b0;
            p_cnt <= '0;
            sum   <= '0;
            state <= S_P2;
          end
        end
        // ------------------------------------------------------------------------------
        S_P2: begin
          // address stage
          if (t_a != T_r) begin
            va   <= 1'b1;
            ta_q <= t_a;
            t_a  <= t_a + 9'd1;
          end else begin
            va <= 1'b0;
          end
          // z stage
          vz <= va;
          if (va) begin
            tz_q  <= ta_q;
            z_big <= (z64[63:12] != 52'd0);
            z_lo  <= z64[11:0];
          end
          // p stage (writes sbuf, see the buffer block)
          if (vz) begin
            sum   <= sum + 32'(p_val);
            p_cnt <= p_cnt + 9'd1;
          end
          if (p_cnt == T_r) state <= S_DIV_START;
        end
        S_DIV_START: state <= S_DIV_WAIT;
        S_DIV_WAIT: if (udiv_done) begin
          inv_r    <= udiv_q;
          t_iss    <= '0;
          t_acc    <= '0;
          row_addr <= vkv;
          for (int d = 0; d < LANES; d++) o[d] <= '0;
          state    <= S_V;
        end
        // ------------------------------------------------------------------------------
        S_V: begin
          if (attn_rd_valid && attn_rd_grant) begin
            t_iss    <= t_iss + 9'd1;
            row_addr <= row_addr + 24'(D_r);
          end
          if (attn_rd_rvalid) begin
            pn_q <= pn;
            v_q  <= attn_rd_rdata;
            vv   <= 1'b1;
          end
          if (vv) begin
            for (int d = 0; d < LANES; d++)
              o[d] <= o[d] + (32'($signed({24'd0, pn_q})) * sx8_32(v_q[8*d +: 8]));
            t_acc <= t_acc + 9'd1;
          end
          if (t_acc == T_r) begin
            og    <= '0;
            state <= S_OUT;
          end
        end
        // ------------------------------------------------------------------------------
        S_OUT: begin
          wdata_acc[64*og +: 64] <= out8;
          og <= og + 3'd1;
          if (32'(og) == (32'(D_r) / OUT_LANES) - 1) state <= S_PUSH;
        end
        S_PUSH: if (!wq_full) begin
          // advance (m, h)
          if (ch == H_r - 8'd1) begin
            ch   <= '0;
            hg   <= '0;
            hoff <= '0;
            kkv  <= kbase_r;
            vkv  <= vbase_r;
            qrow <= qrow + HD_r;
            orow <= orow + HD_r;
            T_r  <= T_r + 9'd1;
            if (cm == M_r - 8'd1) begin
              state <= S_DRAIN;
            end else begin
              cm    <= cm + 8'd1;
              state <= S_Q;
            end
          end else begin
            ch   <= ch + 8'd1;
            hoff <= hoff + 24'(D_r);
            if (hg == G_r - 8'd1) begin
              hg  <= '0;
              kkv <= kkv + kvs_r;
              vkv <= vkv + kvs_r;
            end else begin
              hg <= hg + 8'd1;
            end
            state <= S_Q;
          end
        end
        // ------------------------------------------------------------------------------
        S_KV: begin
          if (attn_rd_valid && attn_rd_grant) begin
            dst_pending <= dst_cur;
            src_addr    <= src_addr + 24'(D_r);
            r_iss       <= r_iss + 16'd1;
            if (kvh == Hkv_r - 8'd1) begin
              kvh     <= '0;
              dst_m   <= dst_m + 24'(D_r);
              dst_cur <= dst_m + 24'(D_r);
            end else begin
              kvh     <= kvh + 8'd1;
              dst_cur <= dst_cur + kvs_r;
            end
          end
          if (r_iss == R_r) state <= S_DRAIN;
        end
        // ------------------------------------------------------------------------------
        S_DRAIN: if (wq_empty && !attn_rd_rvalid) state <= S_DONE;
        S_DONE:  state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end

  assign perf_sram_stall = (attn_rd_valid && !attn_rd_grant) || (attn_wr_valid && !attn_wr_grant);
  assign perf_mac_cycles = attn_rd_rvalid && (state == S_K || state == S_V);

  logic unused_ok;
  assign unused_ok = udiv_busy | (|instr[1]) | (|instr[0][23:8]) | (|instr[15]) |
                     (|w2[31:24]) | (|w3[31:24]) | (|w4[31:8]) | (|w5[31:24]) | (|w6[31:8]) | (|w7[31:24]) |
                     (|w8[31:8]) | (|w9[31:7]) | (|w10[31:24]) | (|w12[31:6]) | (|w14[31:6]) |
                     (|pos[31:16]) | (|z64[11:0]) | (|p_prod[15:0]) | (|pn_prod[22:0]) | (|s_rd[31:16]);
endmodule
