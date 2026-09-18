# KV capacity and bandwidth: pretrained boundary measurements

These measurements use the independently selected, frozen SmolLM2-135M and Qwen2.5-0.5B images, one decode launch per boundary, DRAM latency 100, DMA depth 32, v1 hardware, and the overlap schedule. Each launch matched every vocabulary logit and the entire KV state against the independent quantized reference. The state before the launch was initialized from a real tokenizer-produced WikiText prefix. This is a **preinitialized-state boundary test**, not a measurement of full RTL execution of that prefix. No clock frequency or token-throughput assumption is used.

## Capacity of the measured images

| Model | Layers L | Query heads H | KV heads Hkv | D | GQA group H/Hkv | Compiled context | Reserved DRAM KV | Compiler SRAM peak | Resident constants |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M | 30 | 9 | 3 | 64 | 3 | 1,024 | 11,796,480 B (11.25 MiB) | 592,416 B | 131,072 B |
| Qwen2.5-0.5B | 24 | 14 | 2 | 64 | 7 | 1,024 | 6,291,456 B (6 MiB) | 913,952 B | 131,072 B |

These values come from the model image metadata. Both images reserve zero KV bytes in the 1 MiB software-addressed SRAM and provide M=16 prefill and M=1 decode programs. Reserved KV capacity stays at the compiled maximum even when only T positions are populated. For int8 keys and values, the logical populated capacity is:

`C(T) = L × 2 × Hkv × T × D bytes`.

Smol consumes 11,520 bytes per additional token of KV capacity; Qwen consumes 6,144. At context 4,096 their analytical cache requirements would be 45 MiB and 24 MiB respectively. These particular pretrained images support 1,024 positions; the engine limit of 4,096 and separate tiny-model tests do not establish that these full pretrained models have been compiled and validated at 4,096. Resident RoPE tables and other SRAM allocations must also fit.

The attention engine has a fixed 256-entry × 32-bit score/probability buffer (1 KiB), plus fixed query, accumulator and request buffers. It does not retain the full context in SRAM. That internal storage is separate from the compiler’s software-addressed SRAM allocation figures above.

## Measured traffic and cycles

MiB means 1,048,576 bytes. “Populated KV” is the analytical C(T); reads and cycles are measured counters. Total reads include attention, weight/constant transfers, and instruction fetches. All values describe one complete model decode launch.

### SmolLM2-135M

| T | Populated KV MiB | Attention read MiB | Total read MiB | Attention share | Cycles |
|---:|---:|---:|---:|---:|---:|
| 255 | 2.802 | 16.809 | 147.073 | 11.43% | 8,563,332 |
| 256 | 2.812 | 16.875 | 147.139 | 11.47% | 8,565,910 |
| 257 | 2.823 | 16.941 | 147.205 | 11.51% | 8,665,599 |
| 511 | 5.614 | 33.684 | 163.948 | 20.55% | 9,606,003 |
| 512 | 5.625 | 33.750 | 164.014 | 20.58% | 9,607,958 |
| 513 | 5.636 | 33.816 | 164.080 | 20.61% | 9,716,412 |
| 1023 | 11.239 | 67.434 | 197.698 | 34.11% | 11,720,641 |
| 1024 | 11.250 | 67.500 | 197.764 | 34.13% | 11,722,437 |

### Qwen2.5-0.5B

| T | Populated KV MiB | Attention read MiB | Total read MiB | Attention share | Cycles |
|---:|---:|---:|---:|---:|---:|
| 255 | 1.494 | 20.918 | 496.775 | 4.21% | 29,271,527 |
| 256 | 1.500 | 21.000 | 496.857 | 4.23% | 29,273,716 |
| 257 | 1.506 | 21.082 | 496.939 | 4.24% | 29,401,777 |
| 511 | 2.994 | 41.918 | 517.775 | 8.10% | 30,586,158 |
| 512 | 3.000 | 42.000 | 517.857 | 8.11% | 30,588,229 |
| 513 | 3.006 | 42.082 | 517.939 | 8.12% | 30,717,433 |
| 1023 | 5.994 | 83.918 | 559.775 | 14.99% | 33,190,829 |
| 1024 | 6.000 | 84.000 | 559.857 | 15.00% | 33,192,882 |

Both models use naturally aligned D=64 rows, so each row transfers exactly one 64-byte beat. Measured attention reads equal `4 × L × H × T × D` at every listed context: three K passes plus one V pass for **each query head**. KV writes add a measured 11,520 bytes per Smol decode or 6,144 bytes per Qwen decode, independent of T. ATTN itself only reads DRAM; the write counters cover the associated KV_WRITE operations.

The jump when T crosses 256 or 512 includes setup for an additional tile in every head/layer. Traffic remains exactly linear: each new token adds 69,120 attention-read bytes for Smol or 86,016 for Qwen. Across this sweep, non-attention reads remain constant at 136,591,616 bytes for Smol and 498,971,840 for Qwen.

## Analytical alternatives, not measured implementations

| Traffic at T=1,024 | Smol MiB | Qwen MiB | Status |
|---|---:|---:|---|
| Current three K + one V, per query head | 67.5 | 84 | Measured |
| Same passes, ideal reuse across a GQA group | 22.5 | 12 | Analytical, current traffic divided by H/Hkv |
| Single K + V read, per query head | 33.75 | 42 | Analytical |
| Single K + V read, ideal GQA reuse | 11.25 | 6 | Analytical lower bound |

The last row equals reading the populated KV cache once. It would reduce attention traffic by factors of 6 and 14 relative to today’s implementation, but it is not a demonstrated algorithm or speedup. The three K passes currently find the global maximum, form the global exponential sum, then recompute probabilities after the reciprocal is known; the V pass applies those final quantized probabilities. Eliminating passes requires additional score storage or a different numerical algorithm. An online-softmax replacement must prove the required fixed-point semantics rather than assume equivalent rounding.

GQA shares cache capacity today, but fetched K/V rows are not reused across query heads in a group. Reusing each pass across the group could preserve the existing global normalization order while maintaining per-head maximum, sum, score-tile and output-accumulator state. This is a bounded-storage design option, not an implemented result. Arithmetic work remains, and neither byte savings nor shared reads imply an equal cycle speedup.

## Bottleneck implications

| T=1,024 measured counter | Smol | Qwen |
|---|---:|---:|
| DMA DRAM wait cycles | 4,697,898 (40.1% of launch) | 17,213,850 (51.9% of launch) |
| Attention DRAM wait cycles | 540,480 (4.6% of launch) | 709,061 (2.1% of launch) |
| Attention busy cycles | 4,362,865 (37.2% of launch) | 5,450,441 (16.4% of launch) |
| GEMM busy cycles | 1,123,784 (9.6% of launch) | 4,041,232 (12.2% of launch) |

Busy and wait counters overlap and must not be summed into an exclusive cycle breakdown. Busy is engine occupancy, not useful-MAC utilization. DMA wait includes its memory-service stalls; total non-attention read traffic also includes instruction fetches, so it should not all be labeled weights.

The most defensible first performance target is the non-attention DMA path: at T=1,024, it accompanies 65.87% of Smol reads and 85.00% of Qwen reads, while DMA waits occupy 40.1% and 51.9% of launch cycles. Profile transfer sizes and queue occupancy, then test request concurrency and weight/constant scheduling under an unchanged exact-result check. The separate trained-tiny architecture sweep measured benefits from increasing the DMA window, but that result must not be presented as a measured pretrained speedup.

For the next **KV-specific** optimization, grouping GQA heads to reuse each fetched row is better supported by these counters than changing softmax numerics: it directly targets repeated reads while retaining the three-pass normalization contract. An ideal same-pass implementation would remove 45 MiB of Smol reads or 72 MiB of Qwen reads per T=1,024 decode—22.75% or 12.86% of measured total reads. Re-run the exact boundary suite and measure actual cycles before claiming performance gains.

## Provenance

- [Smol boundary results](../build/boundary-smol-selected/run/results.json), [provenance](../build/boundary-smol-selected/run/provenance.json), [tokenizer and prefix source](../build/boundary-smol-selected/prompt-provenance.json).
- [Qwen boundary results](../build/boundary-qwen-selected/run/results.json), [provenance](../build/boundary-qwen-selected/run/provenance.json), [tokenizer and prefix source](../build/boundary-qwen-selected/prompt-provenance.json).
- [Trained-tiny architecture sweep](../build/architecture-study-dram-kv/summary.md); [attention-only context sweep](../build/architecture-study-dram-kv/attention-context.md).

The local build artifacts include every performance counter and SHA256 hashes of inputs and generated fixtures. Exact frozen identifiers:

- SmolLM2-135M image SHA256: `a6f3da37817dec3e7bf8622d899bc70c040f2ac8615fe5bbd78b0f05019cdfd3`.
- SmolLM2-135M boundary tool SHA256: `c3d6700fac07374b6b94fc1c42c62fc4c7834243b25f70d87c569659039ba83d`.
- Qwen2.5-0.5B image SHA256: `a326960d07af4f788fabc8ab49f9b5be463b5a81694a2f331b5dbb3745998467`.
- Qwen2.5-0.5B boundary tool SHA256: `c3d6700fac07374b6b94fc1c42c62fc4c7834243b25f70d87c569659039ba83d`.

Both selected images passed all eight boundaries on RTL and functional simulation (32 device checks), with identical input hashes before preparation and after RTL execution. The selected images retain the same SRAM/cache allocations and all 27 performance counters at every boundary as the earlier authored-image campaigns; the tables above were verified against the selected runs rather than inferred from the earlier results. Temporary binary snapshots were removed only after both backends matched and fixture hashes were rechecked; their hashes remain. Superseded images and removed FP32 export intermediates have separate hash-recorded cleanup provenance.

The cycle measurements cover the isolated device launch. Host tokenization, image/fixture loading, embedding staging and next-token selection are outside this counter interval; no end-to-end serving latency is inferred.
