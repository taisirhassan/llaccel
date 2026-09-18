// causal GQA with a fixed 256-entry score/probability tile and DRAM-backed KV.
// three K passes find the global maximum, sum and probabilities; V accumulates
// one output across tiles. Q, output and KV_WRITE sources remain in SRAM.
// up to 32 tagged pieces may be outstanding. wide heads serialize rows and
// reuse 64 arithmetic lanes; complete dot products precede softmax.
// wide rows use 16-B pieces in 64-B DRAM reads, causing repeated-beat overfetch.

/* verilator lint_off DECLFILENAME */  // private helper module lives in the engine's file

// ---------------------------------------------------------------------------------------
// write queue: {addr, up to 256 B data} entries, registered storage, combinational head.
// ---------------------------------------------------------------------------------------
module attn_engine_wfifo #(
  parameter int DEPTH = 2
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         flush,
  input  logic         push,
  input  logic [31:0]  din_addr,
  input  logic [2047:0] din_data,
  input  logic         pop,
  output logic [31:0]  dout_addr,
  output logic [2047:0] dout_data,
  output logic         empty,
  output logic         full,
  output logic [$clog2(DEPTH+1)-1:0] count
);
  localparam int PW = (DEPTH > 1) ? $clog2(DEPTH) : 1;
  localparam int CW = $clog2(DEPTH+1);
  logic [31:0]  mem_addr [DEPTH];
  logic [2047:0] mem_data [DEPTH];
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
// attention engine
// ---------------------------------------------------------------------------------------
module attn_engine
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
  output logic         dram_req_valid,
  output logic         dram_req_we,
  output logic [31:0]  dram_req_addr,
  output logic [511:0] dram_req_wdata,
  output logic [63:0]  dram_req_wstrb,
  input  logic         dram_req_ready,
  input  logic         dram_rsp_valid,
  input  logic [511:0] dram_rsp_rdata,
  output logic         perf_dram_wait,
  output logic         perf_busy,
  output logic         perf_sram_stall,
  output logic         perf_mac_cycles
);
  localparam int LANES = ATTN_LANES;     // 64 physical arithmetic lanes
  localparam int MAX_D = 256;
  localparam int ROW_BITS = 8 * MAX_D;
  localparam int TMAX  = ATTN_TMAX;
  localparam int TILE = 256;
  localparam int WQ_DEPTH = 2;
  localparam int OUT_LANES = 8;          // requant lanes per cycle

  // ---- helpers -------------------------------------------------------------------------
  function automatic logic signed [47:0] sx32_48(input logic [31:0] v);
    return {{16{v[31]}}, v};
  endfunction
  function automatic logic signed [31:0] sx8_32(input logic [7:0] v);
    return {{24{v[7]}}, v};
  endfunction
  function automatic sram_size_e size_of_d(input logic [8:0] d);
    if (d == 9'd16) return SZ_16;
    if (d == 9'd32) return SZ_32;
    return SZ_64;
  endfunction
  function automatic logic [63:0] strb_of_d(input logic [8:0] d);
    if (d == 9'd16) return 64'h0000_0000_0000_FFFF;
    if (d == 9'd32) return 64'h0000_0000_FFFF_FFFF;
    return 64'hFFFF_FFFF_FFFF_FFFF;
  endfunction
  // A row of `s` bytes whose first 16-B slot of its 256-B line is `slot` (addr[7:4])
  // crosses the line iff its last slot lies beyond the line.
  function automatic logic line_cross(input logic [3:0] slot, input sram_size_e s);
    if (s == SZ_32) return (slot == 4'hF);
    if (s == SZ_64) return (slot >= 4'hD);
    return 1'b0;
  endfunction

  // ---- state ------------------------------------------------------------------------------
  typedef enum logic [3:0] {
    S_IDLE, S_Q, S_K, S_P2, S_DIV_START, S_DIV_WAIT, S_V, S_OUT, S_PUSH, S_KV, S_DRAIN, S_DONE
  } state_e;
  state_e state;

  logic        is_attn;                 // else KV_WRITE
  logic [7:0]  sig_r;
  logic [31:0] kbase_r, vbase_r, kvs_r;
  logic [7:0]  M_r, H_r, Hkv_r, G_r;
  logic [8:0]  D_r;
  logic [23:0] HD_r;                    // H * D bytes (q / out row stride)
  logic [31:0] Ms_r, Mo_r;
  logic [5:0]  Ss_r, So_r;
  sram_size_e  sz_r;
  logic [63:0] strb_r;
  logic [3:0]  np1_r;                   // D/16 - 1, up to sixteen pieces

  // ATTN loop bookkeeping
  logic [7:0]  cm, ch, hg;              // row, head, index within the GQA group
  logic [31:0] total_r, tile_start;
  logic [1:0] pass_r; // max, sum, normalized value accumulation
  logic [8:0]  T_r;                     // keys for the current row (POS + m + 1)
  logic [23:0] qrow, orow, hoff;
  logic [31:0] kkv, vkv;
  logic        q_issued;
  logic [8:0]  t_iss, t_rsp, t_wr, t_a, t_acc, p_cnt;
  logic [31:0] row_addr;
  logic [ROW_BITS-1:0] qv;                     // q lanes (zero beyond D)
  // score pipeline
  logic signed [31:0] s_reg, mx;
  logic        s_valid;
  logic [7:0]  s_t;
  logic [31:0] sbuf [TILE];             // scores, then probabilities
  logic [31:0] s_rd;                    // synchronous read data
  logic [7:0]  sbuf_raddr;
  // softmax pipeline
  logic        va, vz;
  logic [7:0]  ta_q, tz_q;
  logic        z_big;                   // z >= 4096
  logic [11:0] z_lo;
  logic [31:0] sum;
  logic [31:0] inv_r;
  // value pipeline
  logic [14:0] pn_q;
  logic wide_prob;
  logic [ROW_BITS-1:0] v_q;
  logic        vv;
  // Q15 coefficient sum <=32768+TMAX/2; each i8 accumulator stays within
  // 128*(32768+2048), independent of head width.
  logic signed [31:0] o [MAX_D];
  // output requant
  logic [4:0]  og;
  logic [ROW_BITS-1:0] wdata_acc;
  // KV_WRITE bookkeeping
  logic [15:0] r_iss, rows_r;
  logic [7:0]  kvh;
  logic [31:0] src_addr, dst_m, dst_cur, dst_pending;

  // ---- decode ---------------------------------------------------------------------------------
  logic [7:0]  d_op;
  logic        d_attn;
  logic [31:0] w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14;
  logic [7:0]  d_G;
  logic [23:0] d_HD;
  logic [31:0] d_posD;
  logic [15:0] d_rows;
  logic [8:0]  d_D;
  logic [31:0] d_D_word;

  assign d_op = instr_opcode(instr);
  assign d_attn = (d_op == 8'(OP_ATTN));
  assign w2 = instr[2]; assign w3 = instr[3]; assign w4 = instr[4]; assign w5 = instr[5]; assign w6 = instr[6];
  assign w7 = instr[7]; assign w8 = instr[8]; assign w9 = instr[9]; assign w10 = instr[10]; assign w11 = instr[11];
  assign w12 = instr[12]; assign w13 = instr[13]; assign w14 = instr[14];
  assign d_G    = (w8[7:0] == 8'd0) ? 8'd1 : (w7[7:0] / w8[7:0]);          // H / Hkv (ATTN)
  assign d_HD   = 24'(w7[7:0] * w9[8:0]);                                  // H * D (ATTN)
  assign d_posD = pos * 32'(w6[8:0]);                                // POS * D (KV_WRITE)
  assign d_rows = 16'(w4[7:0] * w5[7:0]);                                  // M * Hkv (KV_WRITE)
  assign d_D_word = d_attn ? w9 : w6;
  assign d_D = d_D_word[8:0];

  logic accept;
  assign instr_ready  = (state == S_IDLE);
  assign accept       = instr_ready && instr_valid;
  assign busy         = (state != S_IDLE);
  assign done_pulse   = (state == S_DONE);
  assign done_sig_sem = sig_r;
  assign perf_busy    = busy;

  // ---- read port: row request -> pieces, pieces -> row -----------------------------------------
  logic        rd_want;                 // the FSM presents a row read
  logic [31:0] rd_addr;                 // row address (16-B aligned)
  logic        rd_cross, rd_last, rd_take;
  logic [3:0]  rd_ph;                   // piece being presented
  logic        rsp_cross, rsp_last;     // tags of the response arriving this cycle
  logic [3:0]  rsp_ph;
  logic [ROW_BITS-1:0] row_asm;          // assembled 16-B pieces
  logic        row_valid;               // a complete row is on row_data this cycle
  logic [ROW_BITS-1:0] row_data;

  logic kv_read, dram_tag_full, dram_tag_empty, dram_row_valid;
  logic [15:0] dram_tag;
  logic [5:0] dram_tag_count;
  logic [31:0] piece_addr;
  logic [ROW_BITS-1:0] dram_row_data;
  logic [ROW_BITS-1:0] dram_asm;
  logic dram_read_take;
  assign kv_read = (state == S_K || state == S_V);
  assign rd_cross = (D_r > 9'd64) || (kv_read ? ((32'(rd_addr[5:0]) + 32'(D_r)) > 64) : line_cross(rd_addr[7:4], sz_r));
  assign rd_last = !rd_cross || (rd_ph == np1_r);
  assign piece_addr = rd_addr + (rd_cross ? {24'd0,rd_ph,4'd0} : 32'd0);
  assign attn_rd_valid = rd_want && !kv_read;
  assign attn_rd_req = '{addr:piece_addr[23:0], size:rd_cross ? SZ_16 : sz_r};
  assign dram_read_take = kv_read && rd_want && !dram_tag_full && dram_req_ready;
  assign rd_take = (kv_read ? dram_read_take : (attn_rd_valid && attn_rd_grant)) && rd_last;
  sync_fifo #(.WIDTH(16), .DEPTH(32)) u_dram_tags (
    .clk,.rst_n,.clr(accept),.push(dram_read_take),
    .wdata({t_iss[7:0],rd_cross,rd_last,rd_ph,piece_addr[5:4]}),
    .pop(dram_rsp_valid),.rdata(dram_tag),.full(dram_tag_full),.empty(dram_tag_empty),.count(dram_tag_count));
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin dram_row_valid <= 0; dram_row_data <= 0; dram_asm <= 0; end
    else begin
      dram_row_valid <= dram_rsp_valid && dram_tag[6];
      if (dram_rsp_valid) begin
        if (dram_tag[7]) begin
          if (!dram_tag[6]) dram_asm[128*dram_tag[5:2] +:128] <= dram_rsp_rdata[128*dram_tag[1:0] +:128];
          dram_row_data <= dram_asm;
          dram_row_data[128*dram_tag[5:2] +:128] <= dram_rsp_rdata[128*dram_tag[1:0] +:128];
        end else dram_row_data <= ROW_BITS'(dram_rsp_rdata) >> (128*dram_tag[1:0]);
      end
    end
  end
  assign row_valid = kv_read ? dram_row_valid : (attn_rd_rvalid && rsp_last);
  always_comb begin
    row_data = kv_read ? dram_row_data : ROW_BITS'(attn_rd_rdata);
    if (!kv_read && rsp_cross) begin
      row_data = row_asm;
      row_data[128*rsp_ph +:128] = attn_rd_rdata[127:0];
    end
  end

  // ---- write queue and write-port splitter --------------------------------------------------------
  logic        wq_push, wq_pop, wq_empty, wq_full;
  logic [31:0] wq_din_addr, wq_dout_addr;
  logic [ROW_BITS-1:0] wq_din_data, wq_dout_data;
  logic [1:0]  wq_count;
  logic        wr_cross, wr_last;
  logic [3:0]  wr_ph;

  attn_engine_wfifo #(.DEPTH(WQ_DEPTH)) u_wq (
    .clk(clk), .rst_n(rst_n), .flush(accept), .push(wq_push), .din_addr(wq_din_addr), .din_data(wq_din_data),
    .pop(wq_pop), .dout_addr(wq_dout_addr), .dout_data(wq_dout_data), .empty(wq_empty), .full(wq_full), .count(wq_count));

  assign wr_cross = (D_r > 9'd64) || (is_attn ? line_cross(wq_dout_addr[7:4], sz_r) : ((32'(wq_dout_addr[5:0]) + 32'(D_r)) > 64));
  assign wr_last       = !wr_cross || (wr_ph == np1_r);
  assign attn_wr_valid = is_attn && !wq_empty;
  assign attn_wr_req   = '{addr: wr_cross ? wq_dout_addr[23:0] + {16'd0, wr_ph, 4'd0} : wq_dout_addr[23:0],
                           size: wr_cross ? SZ_16 : sz_r};
  assign attn_wr_wdata = wr_cross ? {384'd0, wq_dout_data[128*wr_ph +: 128]} : wq_dout_data[511:0];
  assign attn_wr_wstrb = wr_cross ? 64'h0000_0000_0000_FFFF : strb_r;
  assign wq_pop = (is_attn ? (attn_wr_valid && attn_wr_grant) : (!wq_empty && dram_req_ready)) && wr_last;
  logic [31:0] write_piece;
  assign write_piece = wq_dout_addr + (wr_cross ? {24'd0,wr_ph,4'd0} : 32'd0);
  assign dram_req_valid = is_attn ? (kv_read && rd_want && !dram_tag_full) : !wq_empty;
  assign dram_req_we = !is_attn;
  assign dram_req_addr = is_attn ? {piece_addr[31:6],6'd0} : {write_piece[31:6],6'd0};
  assign dram_req_wdata = (wr_cross ? {384'd0, wq_dout_data[128*wr_ph +:128]} : wq_dout_data[511:0]) << (128*write_piece[5:4]);
  assign dram_req_wstrb = (wr_cross ? 64'hFFFF : strb_r) << (16*write_piece[5:4]);

  // KV_WRITE: a row lands in the queue the cycle after its last piece is granted, so
  // credit = free slots minus the response in flight (a first piece counts too; conservative).
  logic kv_credit;
  assign kv_credit = (3'(wq_count) + 3'(attn_rd_rvalid)) < 3'(WQ_DEPTH);

  always_comb begin
    wq_push     = 1'b0;
    wq_din_addr = dst_pending;
    wq_din_data = row_data;
    if (!is_attn) begin
      wq_push = (state == S_KV || state == S_DRAIN) && row_valid;
    end else if (state == S_PUSH) begin
      wq_push     = !wq_full;
      wq_din_addr = 32'(orow + hoff);
      wq_din_data = wdata_acc;
    end
  end

  // one wide K/V row remains pending until all arithmetic groups consume it.
  // pieces pipeline through 32 tags; narrow heads still overlap rows.
  logic wide_head, wide_pending, k_active;
  logic [1:0] k_group, v_group;
  logic [ROW_BITS-1:0] key_row;
  logic signed [31:0] dot_partial;
  logic k_last_group, v_last_group;
  assign wide_head = D_r > 9'd64;
  assign k_last_group = k_group == 2'((D_r >> 6) - 9'd1);
  assign v_last_group = !wide_head || v_group == 2'((D_r >> 6) - 9'd1);

  // ---- read request mux -------------------------------------------------------------------------------
  always_comb begin
    rd_want = 1'b0;
    rd_addr = row_addr;
    case (state)
      S_Q:  begin rd_want = !q_issued;                      rd_addr = 32'(qrow + hoff); end
      S_K:  begin rd_want = (t_iss != T_r);                 rd_addr = row_addr;    end
      S_V:  begin rd_want = (t_iss != T_r);                 rd_addr = row_addr;    end
      S_KV: begin rd_want = (r_iss != rows_r) && kv_credit; rd_addr = src_addr;    end
      default: ;
    endcase
    if (wide_head && kv_read && wide_pending) rd_want = 1'b0;
  end

  // ---- arithmetic ------------------------------------------------------------------------------------
  // 64 products per cycle; publish one score only after all channel groups.
  // D<=256 bounds |dot|<=2^22 and score differences<2^23, within signed32.
  logic signed [31:0] dot;
  always_comb begin
    dot = 32'sd0;
    for (int d = 0; d < LANES; d++)
      dot = dot + (sx8_32(qv[8*(wide_head ? (int'(k_group)*LANES+d) : d) +: 8]) *
                   sx8_32(wide_head ? key_row[8*(int'(k_group)*LANES+d) +: 8] : row_data[8*d +: 8]));
  end

  // z = mulshift(mx - s, Ms, Ss); p = z >= 4096 ? 0 : (EXPI[z>>8] * EXPF[z&255] + 2^15) >> 16
  logic signed [31:0] diff;
  logic signed [63:0] z64;
  logic [15:0] p_val;
  logic [31:0] p_prod;
  always_comb begin
    diff = mx - $signed(s_rd);
    z64  = mulshift64(sx32_48(diff), Ms_r, Ss_r);
    p_prod = 32'(EXP_INT_LUT[z_lo[11:8]]) * 32'(EXP_FRAC_LUT[z_lo[7:0]]) + 32'd32768;
    p_val  = z_big ? 16'd0 : p_prod[31:16];
  end

  // pn = satu8((p * inv + 2^22) >> 23) for the arriving value row
  logic [63:0] pn_prod;
  logic [14:0] pn;
  always_comb begin
    pn_prod = 64'(s_rd[15:0]) * 64'(inv_r) + (wide_prob ? 64'd32768 : 64'd4194304);
    if (wide_prob) pn = (pn_prod >> 16) > 64'd32767 ? 15'd32767 : 15'(pn_prod >> 16);
    else pn = {7'd0, satu8($signed(pn_prod >> 23))};
  end

  // output requant, OUT_LANES lanes per cycle
  logic [63:0] out8;
  always_comb begin
    for (int i = 0; i < OUT_LANES; i++)
      out8[8*i +: 8] = sat8(mulshift64(wide_prob ? 48'(rshr64(64'($signed(o[{og, 3'(i)}])), 6'd7)) : sx32_48(o[{og, 3'(i)}]), Mo_r, So_r));
  end

  // ---- score buffer (synchronous read, one write port) ---------------------------------------------
  // in S_V the read address is the row whose (last) piece is being issued: its data
  // and p[t] then arrive in the same cycle (t_iss only advances on rd_take).
  assign sbuf_raddr = (state == S_V) ? dram_tag[15:8] : t_a[7:0];
  always_ff @(posedge clk) begin
    s_rd <= sbuf[sbuf_raddr];
    if (state == S_K && s_valid) sbuf[s_t]  <= s_reg;
    if (state == S_P2 && vz)     sbuf[tz_q] <= {16'd0, p_val};
  end

  // ---- udiv: inv = floor(2^31 / sum) ------------------------------------------------------------------
  logic udiv_start, udiv_busy, udiv_done;
  logic [31:0] udiv_q;
  assign udiv_start = (state == S_DIV_START);
  udiv #(.AW(32), .BW(32)) u_udiv (.clk(clk), .rst_n(rst_n), .start(udiv_start), .a(32'h8000_0000), .b(sum),
                                   .busy(udiv_busy), .done(udiv_done), .q(udiv_q));

  // ---- port splitter state ------------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_ph <= '0; rsp_cross <= 1'b0; rsp_last <= 1'b1; rsp_ph <= '0; row_asm <= '0; wr_ph <= '0;
    end else begin
      // response tags: a granted piece returns exactly one cycle later
      rsp_cross <= rd_cross;
      rsp_last  <= rd_last;
      rsp_ph    <= rd_ph;
      if (accept)                              rd_ph <= '0;
      else if (dram_read_take || (attn_rd_valid && attn_rd_grant)) rd_ph <= rd_last ? 4'd0 : rd_ph + 4'd1;
      if (attn_rd_rvalid && rsp_cross && !rsp_last) row_asm[128*rsp_ph +: 128] <= attn_rd_rdata[127:0];
      if (accept)                              wr_ph <= '0;
      else if ((attn_wr_valid && attn_wr_grant) || (!is_attn && dram_req_valid && dram_req_ready)) wr_ph <= wr_last ? 4'd0 : wr_ph + 4'd1;
    end
  end

  // ---- main sequential control ----------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      is_attn <= 1'b0; sig_r <= '0;
      kbase_r <= '0; vbase_r <= '0; kvs_r <= '0;
      M_r <= '0; H_r <= '0; Hkv_r <= '0; G_r <= '0; D_r <= '0; HD_r <= '0;
      Ms_r <= '0; Mo_r <= '0; Ss_r <= '0; So_r <= '0; sz_r <= SZ_16; strb_r <= '0; np1_r <= '0;
      cm <= '0; ch <= '0; hg <= '0; T_r <= '0; total_r <= 0; tile_start <= 0; pass_r <= 0;
      qrow <= '0; orow <= '0; hoff <= '0; kkv <= '0; vkv <= '0;
      q_issued <= 1'b0; t_iss <= '0; t_rsp <= '0; t_wr <= '0; t_a <= '0; t_acc <= '0; p_cnt <= '0;
      row_addr <= '0; qv <= '0;
      s_reg <= '0; mx <= '0; s_valid <= 1'b0; s_t <= '0;
      va <= 1'b0; vz <= 1'b0; ta_q <= '0; tz_q <= '0; z_big <= 1'b0; z_lo <= '0; sum <= '0; inv_r <= '0;
      wide_prob <= 1'b0;
      pn_q <= '0; v_q <= '0; vv <= 1'b0;
      wide_pending <= 1'b0; k_active <= 1'b0; k_group <= '0; v_group <= '0;
      key_row <= '0; dot_partial <= '0;
      for (int d = 0; d < MAX_D; d++) o[d] <= '0;
      og <= '0; wdata_acc <= '0;
      r_iss <= '0; rows_r <= '0; kvh <= '0; src_addr <= '0; dst_m <= '0; dst_cur <= '0; dst_pending <= '0;
    end else begin
`ifndef SYNTHESIS
      if (accept) begin
        assert (d_op == 8'(OP_ATTN) || d_op == 8'(OP_KV_WRITE)) else $error("attn_engine: unsupported opcode %h", d_op);
        assert (d_D_word == 32'd16 || d_D_word == 32'd32 || d_D_word == 32'd64 ||
                d_D_word == 32'd128 || d_D_word == 32'd256)
          else $error("attn_engine: D must be 16/32/64/128/256 (got %0d)", d_D_word);
        if (d_attn) begin
          assert (64'(pos) + 64'(w6) <= 64'(TMAX)) else $error("attn_engine: POS + M = %0d exceeds ATTN_TMAX", pos + w6);
          assert (w8[7:0] != 8'd0 && 16'(d_G) * 16'(w8[7:0]) == 16'(w7[7:0])) else $error("attn_engine: Hkv must divide H");
        end else begin
          assert (64'(pos) + 64'(w4[7:0]) <= 64'(TMAX)) else $error("attn_engine: KV_WRITE POS + M exceeds ATTN_TMAX");
        end
      end
      if (wide_head && kv_read && row_valid) begin
        assert (wide_pending) else $error("attn_engine: wide response without pending row");
        assert (!k_active && !vv) else $error("attn_engine: wide row overwrote active arithmetic");
      end
      if (dram_rsp_valid) assert (!dram_tag_empty)
        else $error("attn_engine: DRAM response without a request tag");
      if (attn_rd_valid) assert (attn_rd_req.addr[3:0] == 4'd0 && (32'(attn_rd_req.addr[7:0]) + size_bytes(attn_rd_req.size)) <= 32'd256)
        else $error("attn_engine: rd request violates alignment/line rule");
      if (attn_wr_valid) assert (attn_wr_req.addr[3:0] == 4'd0 && (32'(attn_wr_req.addr[7:0]) + size_bytes(attn_wr_req.size)) <= 32'd256)
        else $error("attn_engine: wr request violates alignment/line rule");
`endif
      // pipeline valid defaults
      s_valid <= 1'b0;
      vv      <= 1'b0;
      if (wide_head && kv_read && rd_take) wide_pending <= 1'b1;

      case (state)
        // ------------------------------------------------------------------------------
        S_IDLE: if (accept) begin
          wide_pending <= 1'b0; k_active <= 1'b0; k_group <= '0; v_group <= '0;
          wide_prob <= instr[0][9];
          is_attn <= d_attn;
          sig_r   <= instr_sig_sem(instr);
          sz_r    <= size_of_d(d_D);
          strb_r  <= strb_of_d(d_D);
          np1_r   <= 4'((d_D >> 4) - 9'd1);
          D_r     <= d_D;
          if (d_attn) begin
            kbase_r <= w4; vbase_r <= w5;
            M_r     <= w6[7:0];   H_r     <= w7[7:0];  Hkv_r   <= w8[7:0];  G_r <= d_G;
            kvs_r   <= w10;
            Ms_r    <= w11;       Ss_r    <= w12[5:0]; Mo_r    <= w13;      So_r <= w14[5:0];
            HD_r    <= d_HD;
            cm <= '0; ch <= '0; hg <= '0;
            total_r <= pos + 1;
            T_r <= (pos + 1 > TILE) ? 9'(TILE) : 9'(pos+1);
            tile_start <= 0; pass_r <= 0;
            qrow  <= w2[23:0];
            orow  <= w3[23:0];
            hoff  <= '0;
            kkv   <= w4;
            vkv   <= w5;
            q_issued <= 1'b0;
            state <= (w6[7:0] == 8'd0 || w7[7:0] == 8'd0) ? S_DRAIN : S_Q;
          end else begin
            kvs_r   <= w7;
            Hkv_r   <= w5[7:0];
            rows_r  <= d_rows;
            r_iss   <= '0;
            kvh     <= '0;
            src_addr <= {8'd0,w2[23:0]};
            dst_m   <= w3 + d_posD;
            dst_cur <= w3 + d_posD;
            state   <= (d_rows == 16'd0) ? S_DRAIN : S_KV;
          end
        end
        // ------------------------------------------------------------------------------
        S_Q: begin
          if (rd_take) q_issued <= 1'b1;
          if (row_valid) begin
            for (int d = 0; d < MAX_D; d++) qv[8*d +: 8] <= (d < 32'(D_r)) ? row_data[8*d +: 8] : 8'd0;
            q_issued <= 1'b0;
            t_iss    <= '0;
            t_rsp    <= '0;
            t_wr     <= '0;
            row_addr <= kkv;
            mx       <= 32'sh8000_0000;
            tile_start <= 0; pass_r <= 0; sum <= 0;
            T_r <= (total_r > TILE) ? 9'(TILE) : 9'(total_r);
            state    <= S_K;
          end
        end
        // ------------------------------------------------------------------------------
        S_K: begin
          if (rd_take) begin
            t_iss    <= t_iss + 9'd1;
            row_addr <= row_addr + 32'(D_r);
          end
          if (row_valid) begin
            if (wide_head) begin
              key_row <= row_data;
              k_group <= 0;
              dot_partial <= 0;
              k_active <= 1'b1;
            end else begin
              s_reg   <= dot;
              s_valid <= 1'b1;
              s_t     <= t_rsp[7:0];
              t_rsp   <= t_rsp + 9'd1;
            end
          end
          if (k_active) begin
            // no score is published until every channel contributes.
            dot_partial <= dot_partial + dot;
            if (k_last_group) begin
              s_reg <= dot_partial + dot;
              s_valid <= 1'b1;
              s_t <= t_rsp[7:0];
              t_rsp <= t_rsp + 9'd1;
              k_active <= 1'b0;
              wide_pending <= 1'b0;
            end else k_group <= k_group + 2'd1;
          end
          if (s_valid) begin
            if (pass_r == 0 && s_reg > mx) mx <= s_reg;
            t_wr <= t_wr + 9'd1;
          end
          if (t_wr == T_r) begin
            t_a <= 0; va <= 0; vz <= 0; p_cnt <= 0;
            if (pass_r == 0) begin
              t_iss <= 0; t_rsp <= 0; t_wr <= 0;
              if (tile_start + 32'(T_r) == total_r) begin
                pass_r <= 1; tile_start <= 0; row_addr <= kkv;
                T_r <= total_r > TILE ? 9'(TILE) : 9'(total_r);
              end else begin
                tile_start <= tile_start + 32'(T_r);
                row_addr <= kkv + (tile_start + 32'(T_r))*32'(D_r);
                T_r <= (total_r-tile_start-32'(T_r)) > TILE ? 9'(TILE) : 9'(total_r-tile_start-32'(T_r));
              end
            end else state <= S_P2;
          end
        end
        // ------------------------------------------------------------------------------
        S_P2: begin
          // address stage
          if (t_a != T_r) begin
            va   <= 1'b1;
            ta_q <= t_a[7:0];
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
            if (pass_r == 1) sum <= sum + 32'(p_val);
            p_cnt <= p_cnt + 9'd1;
          end
          if (p_cnt == T_r) begin
            if (pass_r == 1) begin
              if (tile_start + 32'(T_r) == total_r) state <= S_DIV_START;
              else begin
                tile_start <= tile_start + 32'(T_r);
                row_addr <= kkv + (tile_start + 32'(T_r))*32'(D_r);
                T_r <= (total_r-tile_start-32'(T_r)) > TILE ? 9'(TILE) : 9'(total_r-tile_start-32'(T_r));
                t_iss <= 0; t_rsp <= 0; t_wr <= 0; state <= S_K;
              end
            end else begin
              t_iss <= 0; t_acc <= 0; row_addr <= vkv + tile_start*32'(D_r); state <= S_V;
            end
          end
        end
        S_DIV_START: state <= S_DIV_WAIT;
        S_DIV_WAIT: if (udiv_done) begin
          inv_r    <= udiv_q;
          t_iss    <= '0;
          t_acc    <= '0;
          row_addr <= kkv; pass_r <= 2; tile_start <= 0; t_rsp <= 0; t_wr <= 0;
          T_r <= total_r > TILE ? 9'(TILE) : 9'(total_r);
          for (int d = 0; d < MAX_D; d++) o[d] <= '0;
          state    <= S_K;
        end
        // ------------------------------------------------------------------------------
        S_V: begin
          if (rd_take) begin
            t_iss    <= t_iss + 9'd1;
            row_addr <= row_addr + 32'(D_r);
          end
          if (row_valid) begin
            pn_q <= pn;
            v_q  <= row_data;
            vv   <= 1'b1;
            v_group <= 0;
          end
          if (vv) begin
            for (int d = 0; d < LANES; d++)
              o[int'(v_group)*LANES+d] <= o[int'(v_group)*LANES+d] +
                  (32'($signed({17'd0, pn_q})) * sx8_32(v_q[8*(int'(v_group)*LANES+d) +: 8]));
            if (v_last_group) begin
              t_acc <= t_acc + 9'd1;
              if (wide_head) wide_pending <= 1'b0;
            end else begin
              v_group <= v_group + 2'd1;
              vv <= 1'b1;
            end
          end
          if (t_acc == T_r) begin
            if (tile_start + 32'(T_r) == total_r) begin og <= 0; state <= S_OUT; end
            else begin
              tile_start <= tile_start + 32'(T_r);
              row_addr <= kkv + (tile_start + 32'(T_r))*32'(D_r);
              T_r <= (total_r-tile_start-32'(T_r)) > TILE ? 9'(TILE) : 9'(total_r-tile_start-32'(T_r));
              t_iss <= 0; t_rsp <= 0; t_wr <= 0; state <= S_K;
            end
          end
        end
        // ------------------------------------------------------------------------------
        S_OUT: begin
          wdata_acc[64*og +: 64] <= out8;
          og <= og + 5'd1;
          if (og == 5'((D_r >> 3) - 9'd1)) state <= S_PUSH;   // D/8 groups of OUT_LANES
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
            total_r <= total_r + 1;
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
          if (rd_take) begin
            dst_pending <= dst_cur;
            src_addr    <= src_addr + 32'(D_r);
            r_iss       <= r_iss + 16'd1;
            if (kvh == Hkv_r - 8'd1) begin
              kvh     <= '0;
              dst_m   <= dst_m + 32'(D_r);
              dst_cur <= dst_m + 32'(D_r);
            end else begin
              kvh     <= kvh + 8'd1;
              dst_cur <= dst_cur + kvs_r;
            end
          end
          if (r_iss == rows_r) state <= S_DRAIN;
        end
        // ------------------------------------------------------------------------------
        S_DRAIN: if (wq_empty && !attn_rd_rvalid) state <= S_DONE;
        S_DONE:  state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end

  assign perf_sram_stall = (attn_rd_valid && !attn_rd_grant) || (attn_wr_valid && !attn_wr_grant);
  assign perf_dram_wait = (dram_req_valid && !dram_req_ready) || (kv_read && t_iss == T_r && !row_valid && !dram_tag_empty);
  assign perf_mac_cycles = ((state == S_K) && (wide_head ? k_active : row_valid)) ||
                           ((state == S_V) && (wide_head ? vv : row_valid));

  logic unused_ok;
  assign unused_ok = (|dram_tag_count) | (|write_piece[3:0]) | udiv_busy | (|instr[1]) | (|instr[0][23:8]) | (|instr[15]) |
                     (|w2[31:24]) | (|w3[31:24]) | (|w4[31:8]) | (|w5[31:24]) | (|w6[31:8]) | (|w7[31:24]) |
                     (|w8[31:8]) | (|w9[31:9]) | (|w10[31:24]) | (|w12[31:6]) | (|w14[31:6]) |
                     (|pos[31:16]) | (|z64[11:0]) | (|p_prod[15:0]) | (|pn_prod[22:0]) | (|s_rd[31:16]);
endmodule
