# llaccel — implementation plan

End-to-end: PyTorch model → torch.export → MLIR (`llaccel` dialect) → custom
64-byte ISA → cycle-accurate SystemVerilog accelerator (Verilator) → bit-exact
token output, with a measured compiler↔microarchitecture co-design experiment
and an OpenROAD PPA evaluation.

## Scope (what "done" means)

```
uv run llaccel-export  ckpt.pt  -o build/model.mlir  --calib build/calib.json
./llaccel-compile build/model.mlir --target=llaccel-v1 --quantize=int8 -o build/v1.llbin
./llaccel-sim     build/v1.llbin --prompt "ROMEO:" --tokens 64 --backend rtl --verify build/golden.json
./llaccel-compile build/model.mlir --target=llaccel-v2 --enable-fusion -o build/v2.llbin
./llaccel-sim     build/v2.llbin ...            # same output, fewer cycles / bytes
make synth                                      # yosys generic + OpenROAD Nangate45, v1 and v2
```

Success criteria (all measured, none asserted):
1. Verilated RTL output == C++ functional ISA simulator == numpy golden model,
   bit-exact, for prefill + N decode tokens of a trained Llama-architecture model.
2. Quantized model vs fp32 model: reported top-1 agreement and logit cosine
   similarity (quantization quality is reported, not hidden).
3. v2 (epilogue fusion in hardware + fusion pass in compiler) vs v1: cycles/token,
   SRAM bytes, DRAM bytes, synthesized area — all four numbers measured.
4. Scheduler experiment: `--schedule=inorder` vs `--schedule=overlap` (DMA/compute
   overlap with double-buffered weight tiles) — cycles/token measured.
5. OpenROAD (Nangate45) results for the compute core: clock, area, power estimate.

## Architecture in one page

```
 host (llaccel-sim)                                   llaccel_top (SystemVerilog)
 ┌───────────────┐   64B instr fetch     ┌──────────────────────────────────────┐
 │ DRAM model    │◄──────────────────────┤ cmd_proc: fetch → decode → in-order   │
 │ (C++, latency/│   64B/cycle data      │ issue to engine FIFOs; WAIT/SIGNAL    │
 │  bandwidth)   │◄────────────┐         │ semaphores; perf counters             │
 └───────────────┘             │         ├──────────┬──────────┬─────────┬──────┤
                               └─────────┤ dma_eng  │ gemm_eng │ vec_eng │ attn │
                                         │ 2-D copy │ 16x16 i8 │ 16-lane │ 64-  │
                                         │          │ MAC, acc │ i16 ALU │ lane │
                                         │          │ epilogue │ rmsnorm │ QK/PV│
                                         │          │ (v2:fuse)│ rope    │ soft-│
                                         │          │          │ silu ...│ max  │
                                         ├──────────┴──────────┴─────────┴──────┤
                                         │ sram_xbar: 16 banks × 16 B, per-bank │
                                         │ fixed-priority arbiter, conflict cnt │
                                         │ (KV cache lives in SRAM region)      │
                                         └──────────────────────────────────────┘
```

Detailed specs: `docs/ARCH.md` (microarchitecture), `docs/ISA.md` (encoding and
semantics), `docs/NUMERICS.md` (bit-exact integer contract shared by golden,
func-sim and RTL).

## Numerics contract (summary)

* Weights: INT8, symmetric, per-output-channel scale.
* GEMM inputs: INT8 per-tensor scale; accumulators INT32; requantization
  `sat(round_half_up((acc + bias) * M[n] >> S[n]))` with per-channel (M,S).
* Residual stream and vector intermediates: INT16 with a per-tensor power-of-two
  exponent chosen by the compiler from calibration.
* KV cache: INT8. Softmax: fixed-point with 2-table exp LUT and one integer divide.
* RMSNorm: 48-bit sum of squares, integer isqrt, integer divide, per-lane
  multiply — no floating point anywhere in the device.
* Every rounding/saturation point is specified once (NUMERICS.md) and
  implemented three times (numpy, C++, SV) — the test suite proves they agree.

## Compiler pipeline (C++ / MLIR)

```
python: torch.export → FX graph → pattern-match ATen (linear, rms_norm/rsqrt-form,
        rope rotate_half form, sdpa+GQA, silu, mul, add) → model.mlir (llaccel dialect)
        + weights.bin (int8 per-channel) + calib.json (activation abs-max)
llaccel-compile (MLIR PassManager):
  1. llaccel-quantize      types f32→i8/i16, computes all integer params, emits
                           requant tables / LUT constants into the DRAM image
  2. llaccel-fuse          (target≥v2) linear+add → epilogue=resadd;
                           linear+silu → epilogue=silu; linear+mul → epilogue=mul
  3. llaccel-tile          split linears along N into weight chunks that fit the
                           SRAM weight budget (pads N,K to multiples of 16)
  4. llaccel-lower-to-isa  high-level ops → llaccel.isa.* ops on symbolic SRAM
                           buffers; weight-tile DMAs; KV-cache addressing by POS
  5. llaccel-alloc-sram    liveness-interval first-fit allocator; reserved regions
                           (KV cache, RoPE tables, constants); reports utilization
  6. llaccel-schedule      list scheduling across engines, semaphore insertion,
                           double-buffered weight prefetch (overlap) or in-order
  7. llaccel-emit          .llbin container: DRAM image + prefill/decode programs
                           + JSON metadata
```

Two programs per model: prefill chunk (M=16 rows, padded; safe because causal
attention never reads a later position and pad rows' KV entries are overwritten)
and decode (M=1). Position comes from a device register `POS` so programs are
position-independent.

## Verification stack

| Level | Tool | Checks |
|---|---|---|
| op | Python `tests/` (pytest) | numpy golden ops vs float reference (error bounds) |
| engine | Verilator C++ unit TBs (`tb/`) | randomized GEMM/VEC/ATTN/DMA vs C++ reference (`include/llaccel/numerics.h`), bit-exact |
| ISA | `llaccel-sim --backend func` | C++ ISA interpreter vs numpy golden (`golden.json`), bit-exact |
| system | `llaccel-sim --backend rtl` | Verilated `llaccel_top` vs func-sim vs golden, bit-exact tokens + perf counters |
| accuracy | `llaccel-verify` | int model vs fp32 model agreement (reported) |

## Physical design

* `synth/yosys`: `read_slang` → `synth` → generic cell/area sanity (local).
* `synth/orfs`: local Yosys writes a flattened Verilog-2005 netlist of
  `llaccel_core` (compute core; SRAM banks presented as ports, area of SRAM
  estimated separately from the platform's fakeram macros), then the ORFS
  Docker image (amd64 under Rosetta) runs synth → floorplan → place → CTS →
  route on Nangate45 with an SDC clock; reports timing/area/power for v1 and v2.

## Model

`TinyLlama` (Llama/Qwen2 architecture: pre-RMSNorm, RoPE rotate_half, GQA,
SwiGLU, optional QKV bias): dim 128, 4 layers, 4 heads (head_dim 32), 2 KV heads,
FFN 384, char-level vocab (padded to 80), context 256. Trained on Tiny
Shakespeare on MPS for a few minutes; used for every end-to-end number.
The importer is written against generic ATen patterns so an HF Llama/Qwen2
block exports the same way (stretch goal, not a claim).

## Estimated size

| Area | Files | LOC (est.) |
|---|---|---|
| RTL (SV) | 14 | 4,500 |
| Verilator unit TBs + DRAM model | 7 | 1,600 |
| MLIR dialect (TableGen) + passes + tools | 20 | 5,000 |
| Runtime (llbin, func-sim, RTL backend, host loop) | 8 | 2,000 |
| Python (model, train, calib, FX importer, golden, verify) | 9 | 2,500 |
| Build, synth flow, scripts, docs | 12 | 900 |
| **Total** | | **~16,500** |

## Build order (dependencies)

1. Specs + shared headers (`isa.h`, `numerics.h`, generated LUTs) — everything else codes against these.
2. Python model + training (runs in background), golden ops, calibration, FX importer.
3. C++ ISA func-sim + runtime + `.llbin` reader (ISA semantics executable).
4. MLIR compiler (dialect → passes → emit); check func-sim == golden.
5. RTL engines with unit TBs; top; Verilator backend; check RTL == func-sim == golden.
6. Perf counters, v1/v2 + schedule experiments, synthesis, RESULTS.md.
