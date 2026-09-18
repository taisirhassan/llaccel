// dma_engine.sv — 2-D DRAM <-> SRAM copy engine (DMA_LOAD / DMA_STORE).
//
// A descriptor is rows x row_bytes with independent source/destination strides.
// All addresses and row_bytes are 16-B aligned; DRAM traffic is 64-B beats, so a
// row may start/end in the middle of a beat: the engine walks the containing
// beats and moves only the useful 16-B chunks.
//
// LOAD : DRAM read beats (<= MAX_OUTSTANDING in flight) -> response FIFO ->
//        one or two 64-B-aligned SRAM writes per beat (byte strobes select the
//        useful chunks; two writes when the chunks straddle a 64-B SRAM block).
//        Completion = last SRAM write granted.
// STORE: per DRAM beat, one or two 64-B-aligned SRAM reads -> merge into a beat
//        register -> DRAM write with byte strobes. Completion = last DRAM write
//        accepted.
module dma_engine
  import llaccel_pkg::*;
#(
  parameter int unsigned MAX_OUTSTANDING = 16
) (
  input  logic               clk,
  input  logic               rst_n,
  // instruction pop interface
  input  logic               instr_valid,
  input  instr_words_t       instr,
  output logic               instr_ready,
  output logic               busy,
  output logic               done_pulse,
  output logic [7:0]         done_sig_sem,
  // DRAM (through dram_arb)
  output logic               dram_req_valid,
  output logic               dram_req_we,
  output logic [31:0]        dram_req_addr,
  output logic [DRAM_DW-1:0] dram_req_wdata,
  output logic [DRAM_BEAT-1:0] dram_req_wstrb,
  input  logic               dram_req_ready,
  input  logic               dram_rsp_valid,
  input  logic [DRAM_DW-1:0] dram_rsp_rdata,
  // SRAM write port
  output logic               dma_wr_valid,
  output sram_req_t          dma_wr_req,
  output logic [511:0]       dma_wr_wdata,
  output logic [63:0]        dma_wr_wstrb,
  input  logic               dma_wr_grant,
  // SRAM read port
  output logic               dma_rd_valid,
  output sram_req_t          dma_rd_req,
  input  logic               dma_rd_grant,
  input  logic               dma_rd_rvalid,
  input  logic [511:0]       dma_rd_rdata,
  // perf
  output logic               perf_busy,
  output logic               perf_sram_stall,
  output logic               perf_dram_wait
);
  typedef enum logic [1:0] { S_IDLE, S_RUN, S_DONE } state_e;
  state_e state;

  // ---- instruction registers ----------------------------------------------------
  logic        is_store;
  logic [31:0] row_bytes, dram_stride;
  logic [23:0] sram_stride;
  logic [7:0]  sig;

  // ---- beat sequencer -----------------------------------------------------------------
  logic        seq_active;
  logic [31:0] ds, de, b;          // DRAM row start, row end, current beat address
  logic [23:0] srow, scur;         // SRAM row start, SRAM address of the beat's first useful chunk
  logic [31:0] rows_left;
  logic        row_first;          // b is the first beat of the row
  logic        rd_phase;           // store: which SRAM read of the beat is being issued

  logic [31:0] rem;
  logic [1:0]  rem_m1_c;           // chunk index of the row's last byte within its beat: ((rem - 1) mod 64) / 16
  logic        row_last, last_beat;
  logic [1:0]  first_c, last_c, p0;
  logic [2:0]  n_c;
  logic        two;                // chunks straddle two aligned 64-B SRAM blocks
  logic        seq_adv;

  always_comb begin
    rem       = de - b;
    rem_m1_c  = 2'((rem[5:0] - 6'd1) >> 4);
    row_last  = (rem <= 32'd64);
    first_c   = row_first ? ds[5:4] : 2'd0;
    last_c    = row_last ? rem_m1_c : 2'd3;
    n_c       = {1'b0, last_c} - {1'b0, first_c} + 3'd1;
    p0        = scur[5:4];
    two       = ({1'b0, p0} + n_c) > 3'd4;
    last_beat = row_last && (rows_left == 32'd1);
  end

  // ---- LOAD: DRAM reads -> FIFOs -> SRAM writes ------------------------------------------
  localparam int unsigned LMETA_W = 1 + 2 + 3 + 20;   // last, first_c, n_c, scur[23:4]
  logic               lmeta_push, lmeta_pop, lmeta_full, lmeta_empty;
  logic [LMETA_W-1:0] lmeta_in, lmeta_out;
  logic [$clog2(MAX_OUTSTANDING+1)-1:0] lmeta_count, ldata_count;
  logic               ldata_full, ldata_empty;
  logic [511:0]       ldata_out;
  logic               h_last;
  logic [1:0]         h_first, h_p0;
  logic [2:0]         h_nc;
  logic [23:0]        h_scur;
  logic               h_two;
  logic               wr_phase;

  assign lmeta_in = {last_beat, first_c, n_c, scur[23:4]};
  assign {h_last, h_first, h_nc, h_scur[23:4]} = lmeta_out;
  assign h_scur[3:0] = 4'd0;
  assign h_p0  = h_scur[5:4];
  assign h_two = ({1'b0, h_p0} + h_nc) > 3'd4;

  sync_fifo #(.WIDTH(LMETA_W), .DEPTH(MAX_OUTSTANDING)) u_lmeta (
    .clk, .rst_n, .clr(1'b0), .push(lmeta_push), .wdata(lmeta_in), .pop(lmeta_pop),
    .rdata(lmeta_out), .full(lmeta_full), .empty(lmeta_empty), .count(lmeta_count));
  sync_fifo #(.WIDTH(512), .DEPTH(MAX_OUTSTANDING)) u_ldata (
    .clk, .rst_n, .clr(1'b0), .push(dram_rsp_valid), .wdata(dram_rsp_rdata), .pop(lmeta_pop),
    .rdata(ldata_out), .full(ldata_full), .empty(ldata_empty), .count(ldata_count));

  logic ld_issue, ld_accept, ld_have, ld_wr_last_of_beat;
  assign ld_issue  = (state == S_RUN) && !is_store && seq_active && !lmeta_full;
  assign ld_accept = ld_issue && dram_req_ready;
  assign lmeta_push = ld_accept;
  assign ld_have   = !ldata_empty;
  assign ld_wr_last_of_beat = !(h_two && !wr_phase);
  assign lmeta_pop = ld_have && dma_wr_grant && ld_wr_last_of_beat;

  // SRAM write assembly: position pos (0..3) of the aligned block takes chunk j = pos + 4*phase - p0
  always_comb begin
    dma_wr_valid    = ld_have;
    dma_wr_req.addr = {h_scur[23:6], 6'd0} + (wr_phase ? 24'd64 : 24'd0);
    dma_wr_req.size = SZ_64;
    dma_wr_wdata    = '0;
    dma_wr_wstrb    = '0;
    for (int unsigned pos = 0; pos < 4; pos++) begin
      logic signed [4:0] j;
      logic [1:0]        src;
      j   = $signed({3'b0, pos[1:0]}) + (wr_phase ? 5'sd4 : 5'sd0) - $signed({3'b0, h_p0});
      src = h_first + j[1:0];
      if (j >= 0 && j < $signed({2'b0, h_nc})) begin
        dma_wr_wdata[128*pos +: 128] = ldata_out[128*src +: 128];
        dma_wr_wstrb[16*pos +: 16]   = 16'hFFFF;
      end
    end
  end

  // ---- STORE: SRAM reads -> FIFOs -> merge -> DRAM write ------------------------------------
  localparam int unsigned RMETA_W = 1 + 1 + 1 + 2 + 3 + 2 + 1 + 32; // last, first_of_beat, last_of_beat, first_c, n_c, p0, phase, b
  logic               rmeta_push, rmeta_pop, rmeta_full, rmeta_empty;
  logic [RMETA_W-1:0] rmeta_in, rmeta_out;
  logic [2:0]         rmeta_count, rdata_count;
  logic               rdata_full, rdata_empty;
  logic [511:0]       rdata_out;
  logic               r_last, r_first_of_beat, r_last_of_beat, r_phase;
  logic [1:0]         r_first, r_p0;
  logic [2:0]         r_nc;
  logic [31:0]        r_b;

  assign rmeta_in = {last_beat, ~rd_phase, (rd_phase || !two), first_c, n_c, p0, rd_phase, b};
  assign {r_last, r_first_of_beat, r_last_of_beat, r_first, r_nc, r_p0, r_phase, r_b} = rmeta_out;

  sync_fifo #(.WIDTH(RMETA_W), .DEPTH(4)) u_rmeta (
    .clk, .rst_n, .clr(1'b0), .push(rmeta_push), .wdata(rmeta_in), .pop(rmeta_pop),
    .rdata(rmeta_out), .full(rmeta_full), .empty(rmeta_empty), .count(rmeta_count));
  sync_fifo #(.WIDTH(512), .DEPTH(4)) u_rdata (
    .clk, .rst_n, .clr(1'b0), .push(dma_rd_rvalid), .wdata(dma_rd_rdata), .pop(rmeta_pop),
    .rdata(rdata_out), .full(rdata_full), .empty(rdata_empty), .count(rdata_count));

  logic         wr_valid, wr_last;
  logic [31:0]  wr_addr;
  logic [511:0] wr_data;
  logic [63:0]  wr_strb;
  logic         st_rd_issue, st_merge, st_wr_accept;

  assign st_rd_issue  = (state == S_RUN) && is_store && seq_active && !rmeta_full;
  assign dma_rd_valid = st_rd_issue;
  assign dma_rd_req.addr = {scur[23:6], 6'd0} + (rd_phase ? 24'd64 : 24'd0);
  assign dma_rd_req.size = SZ_64;
  assign rmeta_push   = st_rd_issue && dma_rd_grant;
  assign st_wr_accept = (state == S_RUN) && is_store && wr_valid && dram_req_ready;
  assign st_merge     = !rdata_empty && (!wr_valid || dram_req_ready);
  assign rmeta_pop    = st_merge;

  // ---- DRAM request mux -----------------------------------------------------------------------
  always_comb begin
    if (is_store) begin
      dram_req_valid = (state == S_RUN) && wr_valid;
      dram_req_we    = 1'b1;
      dram_req_addr  = wr_addr;
      dram_req_wdata = wr_data;
      dram_req_wstrb = wr_strb;
    end else begin
      dram_req_valid = ld_issue;
      dram_req_we    = 1'b0;
      dram_req_addr  = b;
      dram_req_wdata = '0;
      dram_req_wstrb = '0;
    end
  end

  // ---- sequencer advance -------------------------------------------------------------------------
  assign seq_adv = is_store ? (rmeta_push && (rd_phase || !two)) : ld_accept;

  // ---- control ----------------------------------------------------------------------------------------
  assign instr_ready  = (state == S_IDLE);
  assign busy         = (state != S_IDLE);
  assign done_pulse   = (state == S_DONE);
  assign done_sig_sem = sig;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= S_IDLE;
      is_store    <= 1'b0;
      row_bytes   <= '0;
      dram_stride <= '0;
      sram_stride <= '0;
      sig         <= NO_SEM;
      seq_active  <= 1'b0;
      ds <= '0; de <= '0; b <= '0; srow <= '0; scur <= '0; rows_left <= '0;
      row_first   <= 1'b0;
      rd_phase    <= 1'b0;
      wr_phase    <= 1'b0;
      wr_valid    <= 1'b0;
      wr_last     <= 1'b0;
      wr_addr     <= '0;
      wr_data     <= '0;
      wr_strb     <= '0;
    end else begin
      case (state)
        S_IDLE: begin
          if (instr_valid) begin
            is_store    <= (instr[0][7:0] == OP_DMA_STORE);
            sig         <= instr_sig_sem(instr);
            row_bytes   <= instr[5];
            dram_stride <= (instr[0][7:0] == OP_DMA_STORE) ? instr[7] : instr[6];
            sram_stride <= (instr[0][7:0] == OP_DMA_STORE) ? instr[6][23:0] : instr[7][23:0];
            ds          <= instr[3];
            de          <= instr[3] + instr[5];
            b           <= {instr[3][31:6], 6'd0};
            srow        <= instr[2][23:0];
            scur        <= instr[2][23:0];
            rows_left   <= instr[4];
            row_first   <= 1'b1;
            rd_phase    <= 1'b0;
            wr_phase    <= 1'b0;
            wr_valid    <= 1'b0;
            seq_active  <= (instr[4] != 32'd0) && (instr[5] != 32'd0);
            state       <= ((instr[4] != 32'd0) && (instr[5] != 32'd0)) ? S_RUN : S_DONE;
          end
        end

        S_RUN: begin
          // sequencer
          if (seq_adv) begin
            b         <= b + 32'd64;
            scur      <= scur + {17'd0, n_c, 4'd0};
            row_first <= 1'b0;
            rd_phase  <= 1'b0;
            if (row_last) begin
              if (rows_left == 32'd1) begin
                seq_active <= 1'b0;
              end else begin
                logic [31:0] ds_next;
                ds_next   = ds + dram_stride;
                rows_left <= rows_left - 32'd1;
                ds        <= ds_next;
                de        <= ds_next + row_bytes;
                b         <= {ds_next[31:6], 6'd0};
                srow      <= srow + sram_stride;
                scur      <= srow + sram_stride;
                row_first <= 1'b1;
              end
            end
          end else if (is_store && rmeta_push) begin
            rd_phase <= 1'b1;   // first of two SRAM reads issued
          end

          // load write stage
          if (!is_store && ld_have && dma_wr_grant) begin
            if (h_two && !wr_phase) begin
              wr_phase <= 1'b1;
            end else begin
              wr_phase <= 1'b0;
              if (h_last) state <= S_DONE;
            end
          end

          // store: DRAM write accept, then merge (merge may re-arm wr_valid in the same cycle)
          if (st_wr_accept) begin
            wr_valid <= 1'b0;
            if (wr_last) state <= S_DONE;
          end
          if (is_store && st_merge) begin
            if (r_first_of_beat) wr_strb <= '0;
            for (int unsigned j = 0; j < 4; j++) begin
              logic [2:0] q;
              logic [1:0] pos;
              q   = {1'b0, r_p0} + 3'(j);
              pos = r_first + 2'(j);
              if (j < r_nc && q[2] == r_phase) begin
                wr_data[128*pos +: 128] <= rdata_out[128*q[1:0] +: 128];
                wr_strb[16*pos +: 16]   <= 16'hFFFF;
              end
            end
            if (r_last_of_beat) begin
              wr_valid <= 1'b1;
              wr_addr  <= r_b;
              wr_last  <= r_last;
            end
          end
        end

        S_DONE: state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end

  // ---- perf ----------------------------------------------------------------------------------------
  assign perf_busy       = busy;
  assign perf_sram_stall = (dma_wr_valid && !dma_wr_grant) || (dma_rd_valid && !dma_rd_grant);
  assign perf_dram_wait  = (state == S_RUN) &&
                           (is_store ? (wr_valid && !dram_req_ready)
                                     : ((ld_issue && !dram_req_ready) || (ldata_empty && !lmeta_empty)));

  logic unused_ok;
  assign unused_ok = &{1'b0, lmeta_count, ldata_count, ldata_full, rmeta_count, rdata_count, rdata_full,
                       rmeta_empty, instr[1], instr[15:8], instr[0][23:8], instr[2][31:24], instr[6][31:24],
                       instr[7][31:24], h_scur[3:0]};
endmodule
