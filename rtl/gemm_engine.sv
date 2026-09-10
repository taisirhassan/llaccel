// gemm_engine.sv — INT8 GEMM with per-channel requantization and (v2) fused epilogue.
//
//   for nt in N/16:                       (output channel tile)
//     for kt in K/16:                     (reduction tile)
//       load W tile (256 B, gemm_w) into the free half of the double-buffered
//       tile register; stream M rows of A (16 B each, gemm_a) through the 16x16
//       multiplier array + 16 adder trees, accumulating acc[m][n] (i32).
//     read rq (128 B) and bias (64 B) for this nt while the MAC pipeline flushes,
//     then drain M rows through the epilogue pipeline, one row per cycle:
//       acc(+bias) -> mulshift(M_n,S_n) -> sat16 | sat8 -> epilogue -> 32 B / 16 B write (gemm_wr)
//
// Tile loads run ahead of the stream (at most one tile), so tile kt+1 is fetched
// while tile kt streams; the crossbar serializes the 256-B read against the A
// row reads. The drain tail (epilogue pipeline + write FIFO) overlaps the next
// nt's streaming. Every arithmetic step follows docs/NUMERICS.md exactly
// (helpers from llaccel_pkg; SIGMOID_LUT from llaccel_luts_pkg).
module gemm_engine
  import llaccel_pkg::*;
#(
  parameter bit EPILOGUE_FUSION = 1'b0
) (
  input  logic          clk,
  input  logic          rst_n,
  // instruction pop interface
  input  logic          instr_valid,
  input  instr_words_t  instr,
  output logic          instr_ready,
  output logic          busy,
  output logic          done_pulse,
  output logic [7:0]    done_sig_sem,
  // weight tile read port (256 B)
  output logic          gemm_w_valid,
  output sram_req_t     gemm_w_req,
  input  logic          gemm_w_grant,
  input  logic          gemm_w_rvalid,
  input  logic [2047:0] gemm_w_rdata,
  // A / rq / bias / aux read port
  output logic          gemm_a_valid,
  output sram_req_t     gemm_a_req,
  input  logic          gemm_a_grant,
  input  logic          gemm_a_rvalid,
  input  logic [511:0]  gemm_a_rdata,
  // output write port
  output logic          gemm_wr_valid,
  output sram_req_t     gemm_wr_req,
  output logic [511:0]  gemm_wr_wdata,
  output logic [63:0]   gemm_wr_wstrb,
  input  logic          gemm_wr_grant,
  // perf
  output logic          perf_busy,
  output logic          perf_mac_cycle,
  output logic          perf_epilogue_cycle,
  output logic [1:0]    perf_sram_stall
);
  typedef enum logic [2:0] { S_IDLE, S_SETUP, S_RUN, S_RQ, S_WAIT, S_DRAIN, S_FINISH, S_DONE } state_e;
  typedef enum logic [2:0] { T_NONE, T_AROW, T_RQ_LO, T_RQ_HI, T_BIAS, T_AUX } tag_e;

  state_e state;

  // ---- instruction registers -------------------------------------------------------
  logic [23:0] a_base, w_base, out_base, rq_base, bias_base, aux_base;
  logic [4:0]  m_rows;            // 1..16
  logic [15:0] n_cols, k_len;
  logic [11:0] nt_cnt, kt_cnt;    // N/16, K/16
  logic [3:0]  mode;
  logic        out_i8, has_bias, need_aux;
  logic [5:0]  aux_sh, silu_si, silu_sh;
  logic [31:0] silu_mi;
  logic [7:0]  sig;
  logic [23:0] out_stride, aux_stride;

  // ---- weight tile loader --------------------------------------------------------------
  logic          ld_active, ld_inflight, ld_par;
  logic [11:0]   ld_nt, ld_kt;
  logic [23:0]   w_ptr;
  logic          wbuf_valid [2];
  logic [2047:0] wbuf       [2];

  // ---- A stream ---------------------------------------------------------------------------
  logic [11:0] st_nt, st_kt;
  logic [4:0]  st_m;
  logic        st_par;
  logic [23:0] a_addr, a_kt_base;
  logic        wbuf_ready, st_last_row, st_last_kt;

  // ---- gemm_a response tag -------------------------------------------------------------------
  tag_e        a_tag;
  logic [4:0]  a_tag_m;
  logic        a_tag_par;
  logic        a_row_rvalid;

  // ---- rq / bias -----------------------------------------------------------------------------------
  logic [1:0]  rq_step;
  logic [1:0]  rq_pending;
  logic [23:0] rq_ptr, bias_ptr;
  logic [31:0] rq_m   [16];
  logic [5:0]  rq_s   [16];
  logic [31:0] bias_v [16];

  // ---- MAC pipeline ---------------------------------------------------------------------------------
  logic               p_valid, s_valid;
  logic [4:0]         p_m, s_m;
  logic signed [15:0] prod [16][16];
  logic signed [19:0] psum [16];
  logic signed [31:0] acc  [16][16];

  // ---- epilogue pipeline -----------------------------------------------------------------------------
  logic        adv;
  logic        d0_fire, d1_v, d2_v, d3_v, d4_v;
  logic [4:0]  dr_m;
  logic [23:0] out_ptr, aux_ptr, out_nt_base, aux_nt_base;
  logic signed [31:0] d1_val [16];
  logic [23:0]        d1_addr, d2_addr, d3_addr, d4_addr;
  logic signed [63:0] d2_p   [16];
  logic [255:0]       aux_d1;
  logic [255:0]       d2_aux;
  logic signed [15:0] d3_t   [16];
  logic signed [7:0]  d3_t8  [16];
  logic [255:0]       d3_aux;
  logic signed [15:0] d4_y   [16];
  logic signed [15:0] d4_u   [16];
  logic signed [15:0] d4_t   [16];
  logic signed [7:0]  d4_y8  [16];
  logic signed [15:0] d3_y_c [16];   // combinational stage-3 result
  logic signed [15:0] d3_u_c [16];
  logic signed [15:0] d4_y_c [16];   // combinational stage-4 result
  logic [255:0]       out_pack;

  // ---- output FIFO ------------------------------------------------------------------------------------------
  localparam int unsigned OUT_W = 24 + 256;
  logic             out_push, out_pop, out_full, out_empty;
  logic [OUT_W-1:0] out_in, out_out;
  logic [1:0]       out_count;

  // ======================================================================================
  // Weight tile loader: loads tiles in linear (nt, kt) order into alternating halves.
  // ======================================================================================
  assign gemm_w_valid    = (state == S_RUN || state == S_RQ || state == S_WAIT || state == S_DRAIN) &&
                           ld_active && !wbuf_valid[ld_par] && !ld_inflight;
  assign gemm_w_req.addr = w_ptr;
  assign gemm_w_req.size = SZ_256;

  // ======================================================================================
  // gemm_a request mux (A rows / rq / bias / aux are in mutually exclusive states)
  // ======================================================================================
  assign wbuf_ready  = wbuf_valid[st_par] || (gemm_w_rvalid && (ld_par == st_par));
  assign st_last_row = (st_m == m_rows - 5'd1);
  assign st_last_kt  = (st_kt == kt_cnt - 12'd1);

  always_comb begin
    gemm_a_valid    = 1'b0;
    gemm_a_req.addr = a_addr;
    gemm_a_req.size = SZ_16;
    case (state)
      S_RUN: begin
        gemm_a_valid    = wbuf_ready;
        gemm_a_req.addr = a_addr;
        gemm_a_req.size = SZ_16;
      end
      S_RQ: begin
        gemm_a_valid    = !d1_v && !d2_v && !d3_v && !d4_v;   // previous nt's drain no longer needs rq regs
        gemm_a_req.addr = (rq_step == 2'd0) ? rq_ptr : (rq_step == 2'd1) ? rq_ptr + 24'd64 : bias_ptr;
        gemm_a_req.size = SZ_64;
      end
      S_DRAIN: begin
        gemm_a_valid    = need_aux && adv;
        gemm_a_req.addr = aux_ptr;
        gemm_a_req.size = SZ_32;
      end
      default: ;
    endcase
  end

  assign a_row_rvalid = gemm_a_rvalid && (a_tag == T_AROW);

  // ======================================================================================
  // Main control
  // ======================================================================================
  assign instr_ready  = (state == S_IDLE);
  assign busy         = (state != S_IDLE);
  assign done_pulse   = (state == S_DONE);
  assign done_sig_sem = sig;
  assign adv          = !out_full;
  assign d0_fire      = (state == S_DRAIN) && adv && (!need_aux || gemm_a_grant);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= S_IDLE;
      a_base <= '0; w_base <= '0; out_base <= '0; rq_base <= '0; bias_base <= '0; aux_base <= '0;
      m_rows <= '0; n_cols <= '0; k_len <= '0; nt_cnt <= '0; kt_cnt <= '0;
      mode <= '0; out_i8 <= 1'b0; has_bias <= 1'b0; need_aux <= 1'b0;
      aux_sh <= '0; silu_si <= '0; silu_sh <= '0; silu_mi <= '0; sig <= NO_SEM;
      out_stride <= '0; aux_stride <= '0;
      ld_active <= 1'b0; ld_inflight <= 1'b0; ld_par <= 1'b0; ld_nt <= '0; ld_kt <= '0; w_ptr <= '0;
      wbuf_valid[0] <= 1'b0; wbuf_valid[1] <= 1'b0;
      st_nt <= '0; st_kt <= '0; st_m <= '0; st_par <= 1'b0; a_addr <= '0; a_kt_base <= '0;
      a_tag <= T_NONE; a_tag_m <= '0; a_tag_par <= 1'b0;
      rq_step <= '0; rq_pending <= '0; rq_ptr <= '0; bias_ptr <= '0;
      dr_m <= '0; out_ptr <= '0; aux_ptr <= '0; out_nt_base <= '0; aux_nt_base <= '0;
    end else begin
      // ---- tile loader (independent of the main FSM) ----
      if (gemm_w_valid && gemm_w_grant) ld_inflight <= 1'b1;
      if (gemm_w_rvalid) begin
        wbuf_valid[ld_par] <= 1'b1;
        ld_inflight        <= 1'b0;
        ld_par             <= ~ld_par;
        w_ptr              <= w_ptr + 24'd256;
        if (ld_kt == kt_cnt - 12'd1) begin
          ld_kt <= '0;
          ld_nt <= ld_nt + 12'd1;
          if (ld_nt == nt_cnt - 12'd1) ld_active <= 1'b0;
        end else begin
          ld_kt <= ld_kt + 12'd1;
        end
      end

      // ---- gemm_a response tag ----
      if (gemm_a_valid && gemm_a_grant) begin
        a_tag_m   <= st_m;
        a_tag_par <= st_par;
        case (state)
          S_RUN:   a_tag <= T_AROW;
          S_RQ:    a_tag <= (rq_step == 2'd0) ? T_RQ_LO : (rq_step == 2'd1) ? T_RQ_HI : T_BIAS;
          default: a_tag <= T_AUX;
        endcase
      end else begin
        a_tag <= T_NONE;
      end

      // ---- rq / bias capture ----
      if (gemm_a_rvalid) begin
        case (a_tag)
          T_RQ_LO: for (int unsigned i = 0; i < 8; i++) begin
            rq_m[i] <= gemm_a_rdata[64*i +: 32];
            rq_s[i] <= gemm_a_rdata[64*i + 32 +: 6];
          end
          T_RQ_HI: for (int unsigned i = 0; i < 8; i++) begin
            rq_m[i + 8] <= gemm_a_rdata[64*i +: 32];
            rq_s[i + 8] <= gemm_a_rdata[64*i + 32 +: 6];
          end
          T_BIAS: for (int unsigned i = 0; i < 16; i++) bias_v[i] <= gemm_a_rdata[32*i +: 32];
          default: ;
        endcase
      end
      case ({(state == S_RQ) && gemm_a_grant, gemm_a_rvalid && (a_tag == T_RQ_LO || a_tag == T_RQ_HI || a_tag == T_BIAS)})
        2'b10:   rq_pending <= rq_pending + 2'd1;
        2'b01:   rq_pending <= rq_pending - 2'd1;
        default: ;
      endcase

      case (state)
        S_IDLE: begin
          if (instr_valid) begin
            a_base    <= instr[2][23:0];
            w_base    <= instr[3][23:0];
            out_base  <= instr[4][23:0];
            rq_base   <= instr[5][23:0];
            bias_base <= instr[6][23:0];
            aux_base  <= instr[7][23:0];
            m_rows    <= instr[8][4:0];
            n_cols    <= instr[9][15:0];
            k_len     <= instr[10][15:0];
            mode      <= instr[11][3:0];
            out_i8    <= instr[11][4];
            aux_sh    <= instr[11][13:8];
            silu_mi   <= instr[12];
            silu_si   <= instr[13][5:0];
            silu_sh   <= instr[13][13:8];
            has_bias  <= instr[0][8 + FLAG_HAS_BIAS];
            sig       <= instr_sig_sem(instr);
            state     <= S_SETUP;
          end
        end

        S_SETUP: begin
          nt_cnt      <= n_cols[15:4];
          kt_cnt      <= k_len[15:4];
          need_aux    <= EPILOGUE_FUSION && (mode == EP_RESADD || mode == EP_MUL);
          out_stride  <= out_i8 ? {8'd0, n_cols} : {7'd0, n_cols, 1'b0};
          aux_stride  <= {7'd0, n_cols, 1'b0};
          ld_active   <= 1'b1; ld_inflight <= 1'b0; ld_par <= 1'b0; ld_nt <= '0; ld_kt <= '0;
          w_ptr       <= w_base;
          wbuf_valid[0] <= 1'b0; wbuf_valid[1] <= 1'b0;
          st_nt <= '0; st_kt <= '0; st_m <= '0; st_par <= 1'b0;
          a_addr <= a_base; a_kt_base <= a_base;
          rq_ptr <= rq_base; bias_ptr <= bias_base; rq_pending <= '0;
          out_nt_base <= out_base; aux_nt_base <= aux_base;
          for (int unsigned i = 0; i < 16; i++) bias_v[i] <= '0;
          if (m_rows == 5'd0 || n_cols[15:4] == 12'd0 || k_len[15:4] == 12'd0) begin
            ld_active <= 1'b0;
            state     <= S_DONE;
          end else begin
            state     <= S_RUN;
          end
        end

        S_RUN: begin
          if (gemm_a_grant) begin
            if (!st_last_row) begin
              st_m   <= st_m + 5'd1;
              a_addr <= a_addr + {8'd0, k_len};
            end else begin
              st_m               <= '0;
              wbuf_valid[st_par] <= 1'b0;    // tile fully consumed: free the half
              st_par             <= ~st_par;
              if (st_last_kt) begin
                st_kt     <= '0;
                a_kt_base <= a_base;
                a_addr    <= a_base;
                rq_step   <= '0;
                state     <= S_RQ;
              end else begin
                st_kt     <= st_kt + 12'd1;
                a_kt_base <= a_kt_base + 24'd16;
                a_addr    <= a_kt_base + 24'd16;
              end
            end
          end
        end

        S_RQ: begin
          if (gemm_a_grant) begin
            case (rq_step)
              2'd0: rq_step <= 2'd1;
              2'd1: begin
                if (has_bias) rq_step <= 2'd2;
                else          state   <= S_WAIT;
              end
              default: state <= S_WAIT;
            endcase
          end
        end

        S_WAIT: begin
          if (!p_valid && !s_valid && !a_row_rvalid && rq_pending == 2'd0 &&
              !(gemm_a_rvalid && (a_tag == T_RQ_LO || a_tag == T_RQ_HI || a_tag == T_BIAS))) begin
            dr_m    <= '0;
            out_ptr <= out_nt_base;
            aux_ptr <= aux_nt_base;
            state   <= S_DRAIN;
          end
        end

        S_DRAIN: begin
          if (d0_fire) begin
            out_ptr <= out_ptr + out_stride;
            aux_ptr <= aux_ptr + aux_stride;
            if (dr_m == m_rows - 5'd1) begin
              out_nt_base <= out_nt_base + (out_i8 ? 24'd16 : 24'd32);
              aux_nt_base <= aux_nt_base + 24'd32;
              rq_ptr      <= rq_ptr + 24'd128;
              bias_ptr    <= bias_ptr + 24'd64;
              if (st_nt == nt_cnt - 12'd1) begin
                state <= S_FINISH;
              end else begin
                st_nt <= st_nt + 12'd1;
                state <= S_RUN;
              end
            end else begin
              dr_m <= dr_m + 5'd1;
            end
          end
        end

        S_FINISH: begin
          if (!d1_v && !d2_v && !d3_v && !d4_v && out_empty && !ld_inflight) state <= S_DONE;
        end

        S_DONE: state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end

  // weight tile registers (no reset: pure data)
  always_ff @(posedge clk) begin
    if (gemm_w_rvalid) wbuf[ld_par] <= gemm_w_rdata;
  end

  // ======================================================================================
  // MAC pipeline: P (products) -> S (adder trees) -> ACC
  // ======================================================================================
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      p_valid <= 1'b0;
      s_valid <= 1'b0;
      p_m     <= '0;
      s_m     <= '0;
    end else begin
      p_valid <= a_row_rvalid;
      s_valid <= p_valid;
      if (a_row_rvalid) p_m <= a_tag_m;
      if (p_valid)      s_m <= p_m;
    end
  end

  always_ff @(posedge clk) begin
    if (a_row_rvalid) begin
      for (int unsigned n = 0; n < 16; n++)
        for (int unsigned k = 0; k < 16; k++)
          prod[n][k] <= $signed(gemm_a_rdata[8*k +: 8]) * $signed(wbuf[a_tag_par][8*(16*n + k) +: 8]);
    end
    if (p_valid) begin
      for (int unsigned n = 0; n < 16; n++) begin
        logic signed [19:0] t;
        t = '0;
        for (int unsigned k = 0; k < 16; k++) t = t + 20'(prod[n][k]);
        psum[n] <= t;
      end
    end
  end

  // accumulators: cleared in SETUP and as each row is drained; accumulated by the S stage
  always_ff @(posedge clk) begin
    if (state == S_SETUP) begin
      for (int unsigned m = 0; m < 16; m++)
        for (int unsigned n = 0; n < 16; n++) acc[m][n] <= '0;
    end else begin
      if (d0_fire) for (int unsigned n = 0; n < 16; n++) acc[dr_m][n] <= '0;
      if (s_valid) for (int unsigned n = 0; n < 16; n++) acc[s_m][n] <= acc[s_m][n] + 32'(psum[n]);
    end
  end

  // ======================================================================================
  // Epilogue pipeline (advances when the output FIFO is not full)
  // ======================================================================================
  // D0 -> D1 : v = acc + bias
  // D1 -> D2 : p = v * M_n
  // D2 -> D3 : t = sat16(rshr(p, S_n)), t8 = sat8(...)  (+ aux row)
  // D3 -> D4 : NONE/RESADD/MUL result, SILU step 1 (u)
  // D4 -> out: SILU step 2 (LUT + final multiply), pack, push
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      d1_v <= 1'b0; d2_v <= 1'b0; d3_v <= 1'b0; d4_v <= 1'b0;
    end else if (adv) begin
      d1_v <= d0_fire;
      d2_v <= d1_v;
      d3_v <= d2_v;
      d4_v <= d3_v;
    end
  end

  always_ff @(posedge clk) begin
    if (gemm_a_rvalid && a_tag == T_AUX) aux_d1 <= gemm_a_rdata[255:0];
    if (adv) begin
      if (d0_fire) begin
        for (int unsigned n = 0; n < 16; n++) d1_val[n] <= acc[dr_m][n] + $signed(bias_v[n]);
        d1_addr <= out_ptr;
      end
      if (d1_v) begin
        for (int unsigned n = 0; n < 16; n++) begin
          logic signed [79:0] pp;
          pp = $signed({{16{d1_val[n][31]}}, d1_val[n]}) * $signed({48'd0, rq_m[n]});
          d2_p[n] <= pp[63:0];
        end
        d2_addr <= d1_addr;
        d2_aux  <= (gemm_a_rvalid && a_tag == T_AUX) ? gemm_a_rdata[255:0] : aux_d1;
      end
      if (d2_v) begin
        for (int unsigned n = 0; n < 16; n++) begin
          logic signed [63:0] r;
          r = rshr64(d2_p[n], rq_s[n]);
          d3_t[n]  <= sat16(r);
          d3_t8[n] <= sat8(r);
        end
        d3_addr <= d2_addr;
        d3_aux  <= d2_aux;
      end
      if (d3_v) begin
        for (int unsigned n = 0; n < 16; n++) begin
          d4_y[n]  <= d3_y_c[n];
          d4_u[n]  <= d3_u_c[n];
          d4_t[n]  <= d3_t[n];
          d4_y8[n] <= d3_t8[n];
        end
        d4_addr <= d3_addr;
      end
    end
  end

  // signed 16 x signed 16 -> signed 32 ; signed 16 x unsigned 16 -> signed 32
  function automatic logic signed [31:0] mul_s16_s16(input logic signed [15:0] x, input logic signed [15:0] y);
    return $signed({{16{x[15]}}, x}) * $signed({{16{y[15]}}, y});
  endfunction
  function automatic logic signed [31:0] mul_s16_u16(input logic signed [15:0] x, input logic [15:0] y);
    logic signed [32:0] p;
    p = $signed({{17{x[15]}}, x}) * $signed({17'd0, y});
    return p[31:0];
  endfunction

  generate
    if (EPILOGUE_FUSION) begin : g_fusion
      // stage 3
      always_comb begin
        for (int unsigned n = 0; n < 16; n++) begin
          logic signed [15:0] ax;
          logic signed [16:0] sum17;
          logic signed [31:0] pm;
          ax    = d3_aux[16*n +: 16];
          sum17 = {d3_t[n][15], d3_t[n]} + {ax[15], ax};
          pm    = mul_s16_s16(d3_t[n], ax);
          d3_u_c[n] = sat16(mulshift64({{32{d3_t[n][15]}}, d3_t[n]}, silu_mi, silu_si));
          case (mode)
            EP_RESADD: d3_y_c[n] = sat16({{47{sum17[16]}}, sum17});
            EP_MUL:    d3_y_c[n] = sat16(rshr64({{32{pm[31]}}, pm}, aux_sh));
            default:   d3_y_c[n] = d3_t[n];
          endcase
        end
      end
      // stage 4: SiLU second half
      always_comb begin
        for (int unsigned n = 0; n < 16; n++) begin
          logic [7:0]  idx, f;
          logic [15:0] l0, l1, sg;
          logic [16:0] diff;
          logic [24:0] pf;
          logic signed [31:0] px;
          idx  = {~d4_u[n][15], d4_u[n][14:8]};          // (u >> 8) + 128
          f    = d4_u[n][7:0];
          l0   = llaccel_luts_pkg::SIGMOID_LUT[idx];
          l1   = llaccel_luts_pkg::SIGMOID_LUT[9'(idx) + 9'd1];
          diff = {1'b0, l1} - {1'b0, l0};                  // >= 0 (monotone table)
          pf   = diff[15:0] * f;
          sg   = l0 + pf[23:8];
          px   = mul_s16_u16(d4_t[n], sg);
          d4_y_c[n] = (mode == EP_SILU) ? sat16(rshr64({{32{px[31]}}, px}, silu_sh)) : d4_y[n];
        end
      end
    end else begin : g_nofusion
      always_comb begin
        for (int unsigned n = 0; n < 16; n++) begin
          d3_y_c[n] = d3_t[n];
          d3_u_c[n] = '0;
          d4_y_c[n] = d4_y[n];
        end
      end
      logic unused_fusion;
      assign unused_fusion = &{1'b0, d3_aux, silu_mi, silu_si, silu_sh, aux_sh, d4_u[0], d4_t[0]};
    end
  endgenerate

  // pack the output row
  always_comb begin
    out_pack = '0;
    for (int unsigned n = 0; n < 16; n++) begin
      if (out_i8) out_pack[8*n +: 8]   = d4_y8[n];
      else        out_pack[16*n +: 16] = d4_y_c[n];
    end
  end

  assign out_push = d4_v && adv;
  assign out_in   = {d4_addr, out_pack};
  assign out_pop  = gemm_wr_grant;

  sync_fifo #(.WIDTH(OUT_W), .DEPTH(2)) u_out (
    .clk, .rst_n, .clr(1'b0), .push(out_push), .wdata(out_in), .pop(out_pop),
    .rdata(out_out), .full(out_full), .empty(out_empty), .count(out_count));

  assign gemm_wr_valid    = !out_empty;
  assign gemm_wr_req.addr = out_out[OUT_W-1 -: 24];
  assign gemm_wr_req.size = out_i8 ? SZ_16 : SZ_32;
  assign gemm_wr_wdata    = {256'd0, out_out[255:0]};
  assign gemm_wr_wstrb    = out_i8 ? 64'h0000_0000_0000_FFFF : 64'h0000_0000_FFFF_FFFF;

  // ======================================================================================
  // perf
  // ======================================================================================
  assign perf_busy           = busy;
  assign perf_mac_cycle      = a_row_rvalid;
  assign perf_epilogue_cycle = (state == S_DRAIN) || d1_v || d2_v || d3_v || d4_v || !out_empty;
  assign perf_sram_stall     = {1'b0, gemm_w_valid && !gemm_w_grant} + {1'b0, gemm_a_valid && !gemm_a_grant} +
                               {1'b0, gemm_wr_valid && !gemm_wr_grant};

  logic unused_ok;
  assign unused_ok = &{1'b0, out_count, instr[1], instr[0][23:9], instr[0][7:0], instr[2][31:24], instr[3][31:24],
                       instr[4][31:24], instr[5][31:24], instr[6][31:24], instr[7][31:24], instr[8][31:5],
                       instr[9][31:16], instr[10][31:16], instr[11][31:14], instr[11][7:5], instr[13][31:14],
                       instr[13][7:6], instr[15:14], gemm_a_rdata[511:256], n_cols[3:0], k_len[3:0]};
endmodule
