// cmd_proc.sv — command processor: instruction fetch (DRAM, 4-deep prefetch),
// decode, semaphore waits, in-order issue into the four engine queues, HALT/done.
//
// * Fetch: 64-B reads at pc via dram_arb (low priority). Up to 4 instructions are
//   in the prefetch FIFO or in flight. Fetching stops as soon as a HALT is *seen*
//   in a fetch response (so at most 3 instructions past HALT are ever requested);
//   responses for those are discarded.
// * Decode (head of the prefetch FIFO): wait_sem/wait_val checked against the
//   semaphore file; stalls counted (cp_stall_wait / cp_stall_qfull / cp_stall_fetch).
//   NOP and HALT are consumed here (they signal their signal_sem at consumption).
// * Semaphores: 32 x u32, incremented by engine done pulses; a WAIT sees a SIGNAL
//   from the previous cycle.
// * HALT: after the HALT is consumed, done rises (level, until the next start)
//   once every queue is empty, every engine is idle and no fetch is in flight.
// * start: reloads pc from pc_start, clears semaphores / queues / done.
module cmd_proc
  import llaccel_pkg::*;
#(
  parameter int unsigned PREFETCH = 4
) (
  input  logic          clk,
  input  logic          rst_n,
  input  logic          start,
  input  logic [31:0]   pc_start,
  output logic          done,
  output logic          running,
  // instruction fetch (through dram_arb, reads only)
  output logic          fetch_req_valid,
  output logic [31:0]   fetch_req_addr,
  input  logic          fetch_req_ready,
  input  logic          fetch_rsp_valid,
  input  logic [INSTR_W-1:0] fetch_rsp_rdata,
  // engine queues (head presented; engine pops with instr_ready)
  output logic          dma_instr_valid,
  output instr_words_t  dma_instr,
  input  logic          dma_instr_ready,
  input  logic          dma_busy,
  input  logic          dma_done_pulse,
  input  logic [7:0]    dma_done_sig_sem,
  output logic          gemm_instr_valid,
  output instr_words_t  gemm_instr,
  input  logic          gemm_instr_ready,
  input  logic          gemm_busy,
  input  logic          gemm_done_pulse,
  input  logic [7:0]    gemm_done_sig_sem,
  output logic          vec_instr_valid,
  output instr_words_t  vec_instr,
  input  logic          vec_instr_ready,
  input  logic          vec_busy,
  input  logic          vec_done_pulse,
  input  logic [7:0]    vec_done_sig_sem,
  output logic          attn_instr_valid,
  output instr_words_t  attn_instr,
  input  logic          attn_instr_ready,
  input  logic          attn_busy,
  input  logic          attn_done_pulse,
  input  logic [7:0]    attn_done_sig_sem,
  // perf pulses
  output logic          perf_instr_issued,
  output logic          perf_stall_wait,
  output logic          perf_stall_qfull,
  output logic          perf_stall_fetch,
  output logic          dma_q_empty,
  output logic          gemm_q_empty,
  output logic          vec_q_empty,
  output logic          attn_q_empty
);
  typedef enum logic [1:0] { CP_IDLE, CP_RUN, CP_HALT, CP_DONE } state_e;
  state_e state;

  // ---- semaphores ------------------------------------------------------------------------
  logic [31:0] sem [NSEM];

  // ---- fetch -------------------------------------------------------------------------------
  logic [31:0] pc;
  logic [2:0]  outstanding, discard;
  logic        halt_seen, fetch_held;
  logic        pf_push, pf_pop, pf_full, pf_empty;
  logic [INSTR_W-1:0] pf_out;
  logic [$clog2(PREFETCH+1)-1:0] pf_count;
  logic        fetch_accept, rsp_is_halt, rsp_keep;

  // Once presented, a fetch remains valid until accepted even if a previously
  // issued HALT returns meanwhile. dram_arb may already be holding this owner.
  assign fetch_req_valid = fetch_held || ((state == CP_RUN) && !halt_seen &&
                           ({1'b0, pf_count} + {1'b0, outstanding} < 4'(PREFETCH)));
  assign fetch_req_addr  = pc;
  assign fetch_accept    = fetch_req_valid && fetch_req_ready;
  assign rsp_keep        = fetch_rsp_valid && (discard == 3'd0);
  assign rsp_is_halt     = (fetch_rsp_rdata[7:0] == OP_HALT);
  assign pf_push         = rsp_keep;

  sync_fifo #(.WIDTH(INSTR_W), .DEPTH(PREFETCH)) u_pf (
    .clk, .rst_n, .clr(start), .push(pf_push), .wdata(fetch_rsp_rdata), .pop(pf_pop),
    .rdata(pf_out), .full(pf_full), .empty(pf_empty), .count(pf_count));

  // ---- decode --------------------------------------------------------------------------------
  instr_words_t head;
  logic [7:0]   op, wsem, ssem;
  logic [31:0]  wval;
  engine_e      eng;
  logic         wait_ok, head_valid, q_target_full, issue;

  assign head       = instr_words_t'(pf_out);
  assign op         = instr_opcode(head);
  assign wsem       = instr_wait_sem(head);
  assign ssem       = instr_sig_sem(head);
  assign wval       = instr_wait_val(head);
  assign eng        = engine_of(op);
  assign head_valid = (state == CP_RUN) && !pf_empty;
  assign wait_ok    = (wsem == NO_SEM) || (sem[wsem[4:0]] >= wval);

  // ---- engine queues ---------------------------------------------------------------------------
  logic         q_push [4], q_pop [4], q_full [4], q_empty [4];
  logic [INSTR_W-1:0] q_out [4];
  logic [$clog2(QDEPTH+1)-1:0] q_count [4];

  always_comb begin
    case (eng)
      ENG_DMA:  q_target_full = q_full[0];
      ENG_GEMM: q_target_full = q_full[1];
      ENG_VEC:  q_target_full = q_full[2];
      ENG_ATTN: q_target_full = q_full[3];
      default:  q_target_full = 1'b0;
    endcase
    issue = head_valid && wait_ok && !q_target_full;
    for (int unsigned i = 0; i < 4; i++) q_push[i] = 1'b0;
    if (issue) begin
      case (eng)
        ENG_DMA:  q_push[0] = 1'b1;
        ENG_GEMM: q_push[1] = 1'b1;
        ENG_VEC:  q_push[2] = 1'b1;
        ENG_ATTN: q_push[3] = 1'b1;
        default: ;
      endcase
    end
  end
  assign pf_pop = issue;

  for (genvar i = 0; i < 4; i++) begin : g_q
    sync_fifo #(.WIDTH(INSTR_W), .DEPTH(QDEPTH)) u_q (
      .clk, .rst_n, .clr(start), .push(q_push[i]), .wdata(pf_out), .pop(q_pop[i]),
      .rdata(q_out[i]), .full(q_full[i]), .empty(q_empty[i]), .count(q_count[i]));
  end

  assign dma_instr_valid  = !q_empty[0];
  assign gemm_instr_valid = !q_empty[1];
  assign vec_instr_valid  = !q_empty[2];
  assign attn_instr_valid = !q_empty[3];
  assign dma_instr  = instr_words_t'(q_out[0]);
  assign gemm_instr = instr_words_t'(q_out[1]);
  assign vec_instr  = instr_words_t'(q_out[2]);
  assign attn_instr = instr_words_t'(q_out[3]);
  assign q_pop[0] = dma_instr_valid  && dma_instr_ready;
  assign q_pop[1] = gemm_instr_valid && gemm_instr_ready;
  assign q_pop[2] = vec_instr_valid  && vec_instr_ready;
  assign q_pop[3] = attn_instr_valid && attn_instr_ready;
  assign dma_q_empty  = q_empty[0];
  assign gemm_q_empty = q_empty[1];
  assign vec_q_empty  = q_empty[2];
  assign attn_q_empty = q_empty[3];

  // ---- semaphore update ------------------------------------------------------------------------
  logic cp_signal;
  assign cp_signal = issue && (eng == ENG_CP) && (ssem != NO_SEM);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int unsigned s = 0; s < NSEM; s++) sem[s] <= '0;
    end else begin
      for (int unsigned s = 0; s < NSEM; s++) begin
        logic [2:0] inc;
        inc = {2'd0, dma_done_pulse  && (dma_done_sig_sem  == 8'(s))} +
              {2'd0, gemm_done_pulse && (gemm_done_sig_sem == 8'(s))} +
              {2'd0, vec_done_pulse  && (vec_done_sig_sem  == 8'(s))} +
              {2'd0, attn_done_pulse && (attn_done_sig_sem == 8'(s))} +
              {2'd0, cp_signal && (ssem == 8'(s))};
        if (start) sem[s] <= '0;
        else       sem[s] <= sem[s] + {29'd0, inc};
      end
    end
  end

  // ---- control -----------------------------------------------------------------------------------
  logic all_idle;
  assign all_idle = q_empty[0] && q_empty[1] && q_empty[2] && q_empty[3] &&
                    !dma_busy && !gemm_busy && !vec_busy && !attn_busy && !fetch_held && (outstanding == 3'd0);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= CP_IDLE;
      pc          <= '0;
      outstanding <= '0;
      discard     <= '0;
      halt_seen   <= 1'b0;
      fetch_held  <= 1'b0;
      done        <= 1'b0;
    end else if (start) begin
      state       <= CP_RUN;
      pc          <= pc_start;
      outstanding <= '0;
      discard     <= '0;
      halt_seen   <= 1'b0;
      fetch_held  <= 1'b0;
      done        <= 1'b0;
    end else begin
      // Track only an unaccepted presented request. Its PC is stable because
      // PC advances exclusively on acceptance.
      if (fetch_req_valid) fetch_held <= !fetch_req_ready;
      // fetch bookkeeping
      if (fetch_accept) pc <= pc + 32'd64;
      case ({fetch_accept, fetch_rsp_valid})
        2'b10:   outstanding <= outstanding + 3'd1;
        2'b01:   outstanding <= outstanding - 3'd1;
        default: ;
      endcase
      // A held request can be accepted after HALT has already been observed.
      // Count that response for discard too; simultaneous accept/return cancel.
      case ({halt_seen && fetch_accept, fetch_rsp_valid && discard != 3'd0})
        2'b10: discard <= discard + 3'd1;
        2'b01: discard <= discard - 3'd1;
        default: ;
      endcase
      if (rsp_keep && rsp_is_halt) begin
        halt_seen <= 1'b1;
        discard   <= outstanding - 3'd1 + {2'd0, fetch_accept};   // everything issued after the HALT
      end

      case (state)
        CP_RUN: begin
          if (issue && op == OP_HALT) state <= CP_HALT;
        end
        CP_HALT: begin
          if (all_idle) begin
            done  <= 1'b1;
            state <= CP_DONE;
          end
        end
        default: ;
      endcase
    end
  end

  assign running = (state == CP_RUN) || (state == CP_HALT);

  // ---- perf ------------------------------------------------------------------------------------------
  assign perf_instr_issued = issue;
  assign perf_stall_wait   = head_valid && !wait_ok;
  assign perf_stall_qfull  = head_valid && wait_ok && q_target_full;
  assign perf_stall_fetch  = (state == CP_RUN) && pf_empty && !halt_seen;

  logic unused_ok;
  always_comb begin
    unused_ok = pf_full | (|pf_count) | (|wsem[7:5]);
    for (int unsigned i = 0; i < 4; i++) unused_ok = unused_ok | (|q_count[i]);
  end
endmodule
