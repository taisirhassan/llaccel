# llaccel ISA (v1 / v2)

Fixed 64-byte instructions (16 little-endian u32 words), fetched by the command
processor from DRAM starting at `pc_start`. In-order issue to four engine
queues (dma, gemm, vec, attn); engines execute their own queues in order and
concurrently with each other. Cross-engine dependencies are expressed with
counting semaphores.

## Header (words 0–1, all opcodes)

```
word0 = opcode[7:0] | flags[15:8] | wait_sem[23:16] | signal_sem[31:24]
word1 = wait_val
```
* `wait_sem != 0xFF`: the CP does not issue this instruction until
  `sem[wait_sem] >= wait_val`. Since issue is in-order, everything behind it also waits.
* `signal_sem != 0xFF`: when the engine *completes* the instruction (all
  results written to SRAM / DRAM), `sem[signal_sem] += 1`.
* 32 semaphores, 32-bit, all reset to 0 at `start`.
* `HALT` completes when every engine queue is empty and idle; then `done` rises.

## Device registers

| reg | set by | use |
|---|---|---|
| `POS` | host, per launch | absolute position of row 0 (RoPE table row, attention key count, KV write slot) |
| `pc_start` | host | first instruction address (DRAM) |

## Opcodes

Sizes: `M ≤ 16`; `N, K, count` multiples of 16; all SRAM addresses 16-byte
aligned; GEMM `w_addr` 256-byte aligned; DRAM addresses 64-byte aligned for
DMA rows of ≥ 64 bytes, else 16-byte aligned.

| op | code | engine | words 2.. |
|---|---|---|---|
| NOP | 0x00 | cp | — |
| HALT | 0x01 | cp | — |
| DMA_LOAD | 0x10 | dma | `sram_dst, dram_src, rows, row_bytes, src_stride, dst_stride` |
| DMA_STORE | 0x11 | dma | `sram_src, dram_dst, rows, row_bytes, src_stride, dst_stride` |
| GEMM | 0x20 | gemm | `a, w, out, rq, bias, aux, M, N, K, ep, silu_Mi, silu_Si_sh` |
| VEC_RMSNORM | 0x30 | vec | `src, gamma, dst, M, K, eps_t, C, sh_post` |
| VEC_ROPE | 0x31 | vec | `src, dst, M, H, D, table, table_stride` |
| VEC_SILU | 0x32 | vec | `src, dst, count, Mi, Si, sh_out` |
| VEC_MUL | 0x33 | vec | `a, b, dst, count, sh` |
| VEC_ADD | 0x34 | vec | `a, b, dst, count, sh_b` |
| VEC_QUANT | 0x35 | vec | `src, dst, count, M, S` |
| ATTN | 0x40 | attn | `q, out, kbase, vbase, M, H, Hkv, D, kv_stride, Ms, Ss, Mo, So` |
| KV_WRITE | 0x41 | attn | `src, base, M, Hkv, D, kv_stride` |

Word positions are exactly the order listed (word2 = first operand).

### GEMM details
* `flags` bit0 = `has_bias` (else `bias` ignored).
* `ep` = `mode[3:0] | out_i8[4] | aux_shift[15:8]`; `mode`: 0 NONE, 1 RESADD, 2 SILU, 3 MUL.
* `silu_Si_sh` = `Si[7:0] | sh_out[15:8]`.
* `A`: i8 `[M][K]`, row stride `K`. `out`/`aux`: `[M][N]` i16 (row stride `2N`) or i8 (`N`).
* `rq`: `N × {M_n i32, S_n i32}` = 8 bytes/channel. `bias`: `N × i32`.
* `W` tiled layout: tile `(nt, kt)` at `w + (nt · K/16 + kt) · 256`; byte
  `n·16 + k` of the tile is `W[nt·16 + n][kt·16 + k]`.
* Cycle model: per `(nt, kt)` tile, 1 cycle weight load (256 B, all banks) +
  `M` cycles streaming A rows (16 B each); per `nt`, `M` drain cycles through the
  epilogue (reads rq/bias once per `nt`, aux per row, writes one output row-slice
  of 16 per cycle).

### VEC details
* i16 element streams, 16 lanes per cycle, 32 B per operand per cycle.
* `VEC_RMSNORM`: rows `M`, row length `K` (multiple of 16), `gamma` `[K]`.
* `VEC_ROPE`: `x [M][H·D]`, table row `p` at `table + p · table_stride` (`[cos D/2][sin D/2]` i16); `p = POS + m`.

### ATTN / KV_WRITE details
* `q` i8 `[M][H·D]`; output i8 `[M][H·D]`.
* `k[kvh][t][d]` at `kbase + kvh·kv_stride + t·D`; `D ∈ {16, 32, 64}`; keys `t < POS + m + 1`.
* `KV_WRITE`: `src` i8 `[M][Hkv·D]` → `base + kvh·kv_stride + (POS+m)·D`.

## Binary container (`.llbin`)

```
magic "LLBN", u32 version=1, u32 n_sections, then sections:
  {u32 kind, u32 flags, u64 offset, u64 size}  kinds: 1 DRAM_IMAGE, 2 PROGRAM (flags=M rows), 3 META_JSON
```
`META_JSON` carries: model dims, `E_RES`, `E_LOGIT`, vocab size, embedding
table offset/stride, input buffer DRAM address + row stride, output (logits)
DRAM address + row stride, per-program entry PC and M, SRAM layout summary,
compiler statistics (instruction counts, SRAM utilization, expected DRAM bytes).
