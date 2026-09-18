# DRAM-backed KV architecture measurements

All 64 RTL runs matched every generated token, per-step argmax, and last-row logit against the independent quantized graph. Model: trained character decoder, dimension 128, FFN 384, 4 layers, 4 query heads / 2 KV heads, D = 32, vocabulary 65. Prompt: `ROMEO:` (6 tokens); 16 generated tokens. Reported counters are means over the 16 decode launches (positions 6–21); prefill is excluded. Host tokenization, image loading, embedding staging and greedy selection are outside these device-cycle measurements.

Sweep: DMA depths 16/32, v1/v2, inorder/overlap schedules, 8/32KiB weight chunks, DRAM latency parameters 0/25/100/300. A 4KiB chunk cannot represent this model’s minimum 16-column FFN tile (6144 bytes), so 8KiB is used. Latency 0 remains cycle-driven bus simulation. These are simulator cycle/traffic measurements, with no frequency or silicon-throughput assumption.

## Best configuration at each latency

| Latency | DMA depth | Target | Schedule | Chunk KiB | Decode cycles |
|---:|---:|---|---|---:|---:|
| 0 | 16 | v2 | overlap | 32 | 21,864.8 |
| 0 | 32 | v2 | overlap | 32 | 21,837.3 |
| 25 | 16 | v2 | overlap | 32 | 31,449.9 |
| 25 | 32 | v2 | overlap | 32 | 24,866.6 |
| 100 | 16 | v2 | overlap | 32 | 94,246.8 |
| 100 | 32 | v2 | overlap | 32 | 59,531.4 |
| 300 | 16 | v1 | overlap | 32 | 266,168.1 |
| 300 | 32 | v1 | overlap | 32 | 167,592.4 |

## Latency 100: all configurations

| Depth | Target | Schedule | Chunk KiB | Cycles | GEMM busy % | DMA busy % | Attention busy % | DMA wait | Attention wait |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | v1 | inorder | 8 | 115,343.0 | 7.5 | 82.7 | 7.6 | 81,489.0 | 5,536.0 |
| 32 | v1 | inorder | 8 | 83,832.0 | 10.4 | 76.1 | 10.5 | 49,978.0 | 5,536.0 |
| 16 | v1 | inorder | 32 | 107,405.0 | 7.6 | 82.2 | 8.2 | 74,703.0 | 5,536.0 |
| 32 | v1 | inorder | 32 | 69,751.0 | 11.7 | 72.6 | 12.6 | 37,049.0 | 5,536.0 |
| 16 | v1 | overlap | 8 | 100,254.8 | 8.7 | 95.5 | 9.5 | 81,926.2 | 6,262.9 |
| 32 | v1 | overlap | 8 | 71,930.2 | 12.1 | 89.4 | 14.1 | 50,648.6 | 6,885.9 |
| 16 | v1 | overlap | 32 | 94,338.9 | 8.7 | 94.5 | 10.2 | 74,858.1 | 6,383.3 |
| 32 | v1 | overlap | 32 | 59,572.2 | 13.7 | 86.5 | 16.8 | 37,009.7 | 6,737.6 |
| 16 | v2 | inorder | 8 | 115,080.0 | 7.6 | 82.8 | 7.6 | 81,489.0 | 5,536.0 |
| 32 | v2 | inorder | 8 | 83,569.0 | 10.4 | 76.4 | 10.5 | 49,978.0 | 5,536.0 |
| 16 | v2 | inorder | 32 | 107,167.0 | 7.6 | 82.4 | 8.2 | 74,703.0 | 5,536.0 |
| 32 | v2 | inorder | 32 | 69,513.0 | 11.8 | 72.8 | 12.7 | 37,049.0 | 5,536.0 |
| 16 | v2 | overlap | 8 | 101,650.6 | 8.6 | 94.1 | 9.2 | 81,771.9 | 6,091.7 |
| 32 | v2 | overlap | 8 | 72,434.5 | 12.0 | 88.7 | 13.4 | 50,514.9 | 6,459.8 |
| 16 | v2 | overlap | 32 | 94,246.8 | 8.7 | 94.6 | 10.2 | 74,844.7 | 6,358.6 |
| 32 | v2 | overlap | 32 | 59,531.4 | 13.7 | 86.5 | 16.8 | 37,026.1 | 6,723.8 |

Busy percentages can overlap and are occupancy, not useful-MAC utilization. Wait counters can overlap busy counters and one another.

## Best latency 100 configuration: counter attribution

| Counter | DMA16 | DMA32 |
|---|---:|---:|
| cycles | 94,246.8 | 59,531.4 |
| gemm_busy | 8,177.0 | 8,177.0 |
| gemm_mac_cycles | 3,112.0 | 3,112.0 |
| gemm_sram_stall | 2,776.0 | 2,776.0 |
| gemm_epilogue_cycles | 2,070.0 | 2,070.0 |
| vec_busy | 1,319.0 | 1,319.0 |
| vec_sram_stall | 39.8 | 39.8 |
| attn_busy | 9,613.8 | 9,972.4 |
| attn_mac_cycles | 928.0 | 928.0 |
| attn_sram_stall | 0.0 | 0.0 |
| attn_dram_wait | 6,358.6 | 6,723.8 |
| attn_dram_rd_bytes | 59,392.0 | 59,392.0 |
| attn_dram_wr_bytes | 1,024.0 | 1,024.0 |
| dma_busy | 89,112.8 | 51,484.5 |
| dma_sram_stall | 802.0 | 1,578.7 |
| dma_dram_wait | 74,844.7 | 37,026.1 |
| cp_stall_wait | 92,767.8 | 57,947.1 |
| cp_stall_qfull | 0.0 | 0.0 |
| cp_stall_fetch | 1,274.0 | 1,379.2 |
| dram_rd_bytes | 929,280.0 | 929,280.0 |
| dram_wr_bytes | 1,216.0 | 1,216.0 |
| sram_rd_bytes | 914,624.0 | 914,624.0 |
| sram_wr_bytes | 878,496.0 | 878,496.0 |

Attention reads K three times and V once to preserve global softmax normalization with bounded on-chip buffering. D = 32 currently transfers a containing 64-byte beat for each 32-byte row, so bus traffic includes overfetch. KV writes similarly use 64-byte bus beats with masks. DRAM read/write totals include attention traffic; attention counters provide its attribution.

Raw records in `results.json` contain all 27 performance counters, image and simulator hashes, and each exact-match outcome. Per-run JSON/log files preserve the full trace. `status.json` records completion and compiler hash.

## Fusion tradeoff on the measured workload

With DMA32, overlap scheduling and 32KiB chunks fixed, v2 reduces decode cycles
relative to v1 by 0.88% at latency0, 0.70% at latency25, and 0.069% at latency100;
at latency300 it uses 0.048% more cycles. These small differences do not support
a broad throughput claim for fusion. The checked synthesis areas are 1.825 mm²
for v1 and 2.047 mm² for v2, approximately 12.2% higher for v2; both exclude
external SRAM and DRAM infrastructure, and routed timing remains pending.
For this workload, v1 is a reasonable area-conscious choice while retaining v2
as an experimentally measured option. The larger measured performance gain
comes from DMA depth and scheduling, not fusion alone.

## Reproduce

```sh
uv run python scripts/benchmark_arch.py --compiler build/dma32/compiler/bin/llaccel-compile --baseline build/dram-kv-dma16/llaccel-sim --candidate build/dma32/llaccel-sim --export build/export --out build/architecture-study-dram-kv --tokens 16 --chunks 8192 32768 --latencies 0 25 100 300 --schedules inorder overlap
```

## Attention context bandwidth profile

Seven randomized cases matched the untiled fixed-point reference and preserved every SRAM byte outside the output. Each case uses one query row and one head, D = 64, Q0.15 probabilities, naturally aligned K/V rows, DRAM latency 100, a maximum of 32 outstanding reads, and no artificial SRAM grant denial. Cycles run from instruction acceptance to completion.

| Context | Cycles | DRAM read bytes | DRAM write bytes | Single K + V logical bytes |
|---:|---:|---:|---:|---:|
| 32 | 657 | 8,192 | 0 | 4,096 |
| 128 | 2,061 | 32,768 | 0 | 16,384 |
| 256 | 3,933 | 65,536 | 0 | 32,768 |
| 512 | 7,819 | 131,072 | 0 | 65,536 |
| 1024 | 15,591 | 262,144 | 0 | 131,072 |
| 2048 | 31,135 | 524,288 | 0 | 262,144 |
| 4096 | 62,223 | 1,048,576 | 0 | 524,288 |

Measured read traffic is exactly `4 × T × 64` bytes: three K passes and one V pass. This is twice the logical traffic of reading K and V once. The latter column is an analytical byte count, not a measured older implementation. All ATTN runs perform zero DRAM writes; KV_WRITE is a separate operation.

The same RTL binary handles every context with a fixed 256-entry × 32-bit score/probability buffer (1 KiB), plus fixed query, accumulator, and request buffers. This is logical storage capacity; the standard-cell synthesis flow maps internal RTL arrays, while the external scratchpad SRAM is a separate boundary. Context growth consumes DRAM capacity and linearly increasing bus traffic; it does not enlarge these attention buffer structures. This does not establish that every other model allocation, such as resident RoPE tables, is constant with context.

For contexts at least 256, measured cycles are `47 + 3886 × (T / 256)` at these bus settings. This describes the tested full-tile contexts, not a general performance guarantee. GQA cache sharing is supported, but cross-query-head reuse of fetched K/V rows is not implemented: each query head performs its own passes.

Reproduce:

```sh
make -C tb/attn build
build/tb/attn/attn/Vtb_attn_top +profile=build/architecture-study-dram-kv/attention-context.json +seed=1
```

Without `+profile`, the full existing 714-case regression suite still runs.

## Physical implementation

The final ISA2/DMA32 RTL snapshot is hashed in `build/synth/final-dram-kv/manifest.json`. Native Yosys runs the pinned ORFS synthesis scripts with optional full-adder extraction disabled after profiling identified pathological recursion. The mapped netlist is handed to the downloaded ORFS Docker image for physical implementation. Both final mapped netlists pass connection checks: v1 area is 1,824,750.158 µm² and v2 is 2,047,287.886 µm². V1 also completed clock-tree synthesis with 2,083,809 µm² cell area and −3.99 ns setup slack against the 3 ns constraint. Register-to-register hold slack is +0.08 ns; the worst asynchronous-reset removal check is −0.33 ns. Six capacitance violations remain, with zero slew/fanout violations. V1 global routing subsequently completed with zero overflow, 2,083,809 µm² core area and −4.13 ns setup slack. Detailed routing failed at track assignment from memory exhaustion with both eight and two threads; final routed timing and DRC are unavailable. These intermediate results do not establish timing closure. The source-attributed intermediate record is `results/dram-kv-2026-09-10/v1-cts-diagnostic.json`. External 1 MiB SRAM bank storage is excluded from the `llaccel_core` boundary.

## Historical measurements

Earlier on-chip-KV measurements are retained in `work/streaming-kv-baseline/RESULTS-before-dram-kv.md`. They predate the DRAM KV architecture and should not be used to describe its current performance.
