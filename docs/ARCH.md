# llaccel microarchitecture

Single clock, synchronous active-low reset. All parameters live in
`rtl/llaccel_pkg.sv` and mirror `include/llaccel/isa.h`.

| parameter | default | meaning |
|---|---|---|
| `SRAM_BYTES` | 1 MiB | on-chip scratchpad (KV cache included) |
| `NBANKS` | 16 | banks, each 16 B wide; line = 256 B; `bank = addr[7:4]` |
| `GEMM_TN / GEMM_TK` | 16 / 16 | MAC array 16 (n) × 16 (k) INT8 |
| `GEMM_TM` | 16 | max rows per GEMM instruction (accumulator rows) |
| `VEC_LANES` | 16 | i16 lanes |
| `ATTN_LANES` | 64 | i8 MACs; supports D ∈ {16,32,64} |
| `ATTN_TMAX` | 256 | score buffer entries (max keys per query) |
| `NSEM` | 32 | semaphores |
| `QDEPTH` | 8 | per-engine instruction queue depth |
| `DRAM_BEAT` | 64 B | DRAM data beat |
| `EPILOGUE_FUSION` | 0 (v1) / 1 (v2) | GEMM epilogue RESADD/SILU/MUL datapath present |
| `NUM_PERF` | 24 | performance counters (64-bit) |

## Module hierarchy

```
llaccel_top            (SRAM banks + core; used by Verilator)
 ├─ sram_bank ×16      (behavioral 1-port RAM, 16 B wide, byte enables)
 └─ llaccel_core       (synthesized boundary; SRAM as ports)
     ├─ cmd_proc       fetch (DRAM) → decode → issue; semaphores; HALT/done
     ├─ dma_engine     2-D DRAM↔SRAM copy, 64 B beats
     ├─ gemm_engine    W-tile regs (double-buffered) → 16×16 MAC → i32 acc[16][16] → epilogue
     ├─ vec_engine     16-lane i16 ALU, RMSNorm (isqrt+div), RoPE, SiLU LUT, MUL/ADD/QUANT
     ├─ attn_engine    QK dot, fixed-point softmax (exp LUTs + div), PV, KV write
     ├─ sram_xbar      per-bank fixed-priority arbiter, conflict counters
     ├─ dram_arb       cmd_proc fetch vs dma
     └─ perf_counters
```

## SRAM crossbar contract

Every requester port is a *line request*: `{valid, we, addr[23:0], nbytes ∈ {16,32,64,256}, wdata[2047:0], wstrb}` where `addr` is `nbytes`-aligned for 256, else 16-aligned, and the transfer never crosses a 256-B line. The xbar grants a request in a cycle iff every bank it touches is not taken by a higher-priority port that cycle; a granted read returns data in the next cycle (`rvalid`, `rdata`), a granted write completes that cycle. Ungranted requests must be held. Priority (high→low):

```
gemm_w (256 B rd) > gemm_a (rd) > gemm_wr > attn_rd > attn_wr > vec_rd0 > vec_rd1 > vec_wr > dma_wr > dma_rd
```
Each denied cycle increments the requester's `sram_stall` perf counter.

## Command processor

* Prefetch FIFO (4 instructions) from DRAM via `dram_arb` (lower priority than DMA).
* Decode: header → engine id; `wait_sem/wait_val` checked against the semaphore
  file; stall (counter `cp_stall_wait`) until satisfied; stall when the target
  queue is full (`cp_stall_qfull`).
* Semaphore increments come from engine completion pulses (`done_pulse`,
  `done_signal_sem`); increments and CP checks are ordered so a WAIT sees a
  SIGNAL from the previous cycle.
* `HALT`: stop fetching; assert `done` when all engines report `idle` and all queues are empty.
* `start` pulse resets PC, semaphores, perf counters, and queues.

## Engine skeleton (shared)

```
input  instr_valid, instr (llaccel_instr_t), output instr_ready
output busy, output done_pulse (one cycle when an instruction has fully retired)
sram port(s) per contract, perf increments
```
Each engine pops one instruction, runs it to completion (FSM), pulses `done`,
then pops the next. Retire order == queue order.

## GEMM engine timing

For `nt in N/16`: for `kt in K/16`: load tile (1 cycle, 256 B), then stream `M`
rows (`A[m][kt·16 +: 16]`, 16 B, 1 row/cycle) through the 16×16 multiplier
array and 16 adder trees into `acc[m][0..15]` (i32, 3-stage pipeline).
Then drain `M` rows: per cycle one row of 16 accumulators → requant (16 ×
(i32·i32→i64 >> S) ) → epilogue → write 32 B (i16) or 16 B (i8). Weight tile
registers are double-buffered so the next tile load overlaps the current
stream (the xbar still serializes the 256-B load against A reads, so per-tile
cost is `1 + M` cycles). `rq` (8 B/ch) and `bias` are fetched once per `nt`
(128 B and 64 B reads). MAC-busy cycles are counted (`gemm_mac_cycles`) for
utilization = mac_cycles / total cycles.

## Vector engine timing

16 lanes/cycle; operands read 32 B/cycle each; 3-stage pipeline (read →
multiply → shift/saturate/write). RMSNorm: pass 1 sum-of-squares (K/16 cycles),
then `isqrt` (24-cycle bit-serial) and `udiv` (32-cycle restoring), then pass 2
(K/16 cycles). RoPE: per row per head, reads `x1`,`x2`,`cos`,`sin` 16 lanes at a time
(D/2/16 steps, 4 reads per step). SiLU: LUT is 257 × u16 ROM, two reads per lane per cycle
(implemented as two ROM copies or a dual-read ROM).

## Attention engine timing

Per `(m, h)`: pass 1 streams `T = POS+m+1` K rows (D bytes/cycle, 64 MACs → 1
key/cycle for D ≤ 64), scores into the score buffer with running max; pass 2
walks the buffer computing `p[t]` (1/cycle) and the sum; `udiv` (32 cycles);
pass 3 streams V rows computing `pn[t]` on the fly and accumulating `o[d]`;
then requant + write D bytes. `KV_WRITE` copies `M × Hkv` rows.

## DMA engine

Descriptor: rows × row_bytes with strides. Issues 64-B DRAM read beats
(`≤ 16` outstanding) and writes 64 B (4 banks) per cycle into SRAM (`dma_wr`);
stores read 64 B from SRAM and issue 64-B DRAM writes with byte strobes.
Completion = last SRAM write done (load) or last DRAM write accepted (store).

## DRAM interface (to the C++ model)

```
req_valid, req_ready, req_we, req_addr[31:0], req_wdata[511:0], req_wstrb[63:0]
rsp_valid, rsp_rdata[511:0]     (read responses, in order, one per cycle max)
```
The model accepts one request per cycle when `req_ready` (bandwidth = 64
B/cycle), returns read data `LATENCY` cycles later (default 100), up to 32 outstanding.

## Performance counters (index → meaning)

```
0 cycles            1 instr_issued      2 cp_stall_wait    3 cp_stall_qfull   4 cp_stall_fetch
5 gemm_busy         6 gemm_mac_cycles   7 gemm_sram_stall  8 gemm_epilogue_cycles
9 vec_busy         10 vec_sram_stall   11 attn_busy       12 attn_sram_stall  13 attn_mac_cycles
14 dma_busy        15 dma_sram_stall   16 dma_dram_wait
17 sram_rd_bytes   18 sram_wr_bytes    19 dram_rd_bytes   20 dram_wr_bytes
21 gemm_idle_qempty 22 vec_idle_qempty 23 attn_idle_qempty
```
All counters are exposed as a `perf[NUM_PERF]` output of `llaccel_top` and read
by the host after `done`.

## Synthesized boundary

`llaccel_core` (everything except the SRAM bank storage). SRAM bank area is
estimated separately from the platform's fakeram macros and reported as such.
