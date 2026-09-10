# llaccel compiler interfaces

Three file contracts: (1) exporter → compiler (`model.mlir` + `weights.bin` +
`weights.json` + `calib.json`), (2) compiler → golden model (`qgraph.json` +
`qweights.bin`), (3) compiler → runtime (`.llbin`, see ISA.md).

## 1. Exported model (Python → `llaccel-compile`)

`model.mlir` uses the high-level `llaccel` dialect on `f32` tensors. `?` is the
row (token) dimension. Every op result carries `llaccel.name` (string) which is
the key into `calib.json` (per-tensor abs-max observed on calibration data).

```mlir
module attributes {llaccel.model = {dim = 128 : i64, n_layers = 4 : i64, n_heads = 4 : i64,
    n_kv_heads = 2 : i64, head_dim = 32 : i64, ffn = 384 : i64, vocab = 65 : i64,
    max_seq = 256 : i64, rope_base = 10000.0 : f64, rms_eps = 1.0e-5 : f64}} {
  llaccel.weight @embed : tensor<65x128xf32>
  llaccel.weight @l0_attn_norm : tensor<128xf32>
  llaccel.weight @l0_wq : tensor<128x128xf32>          // [N][K] like nn.Linear.weight
  llaccel.weight @l0_bq : tensor<128xf32>              // optional bias
  ...
  func.func @forward(%x: tensor<?x128xf32> {llaccel.name = "input"}) -> tensor<?x65xf32> {
    %h  = llaccel.rmsnorm %x, @l0_attn_norm {eps = 1.0e-5 : f64, llaccel.name = "l0.h"} : tensor<?x128xf32>
    %q  = llaccel.linear %h, @l0_wq, @l0_bq {llaccel.name = "l0.q"} : tensor<?x128xf32> -> tensor<?x128xf32>
    %k  = llaccel.linear %h, @l0_wk {llaccel.name = "l0.k"} : tensor<?x128xf32> -> tensor<?x64xf32>
    %v  = llaccel.linear %h, @l0_wv {llaccel.name = "l0.v"} : tensor<?x128xf32> -> tensor<?x64xf32>
    %qr = llaccel.rope %q {heads = 4 : i64, llaccel.name = "l0.qr"} : tensor<?x128xf32>
    %kr = llaccel.rope %k {heads = 2 : i64, llaccel.name = "l0.kr"} : tensor<?x64xf32>
    %a  = llaccel.attention %qr, %kr, %v {layer = 0 : i64, heads = 4 : i64, kv_heads = 2 : i64,
            head_dim = 32 : i64, llaccel.name = "l0.a"}
          : (tensor<?x128xf32>, tensor<?x64xf32>, tensor<?x64xf32>) -> tensor<?x128xf32>
    %o  = llaccel.linear %a, @l0_wo {llaccel.name = "l0.o"} : tensor<?x128xf32> -> tensor<?x128xf32>
    %x1 = llaccel.add %x, %o {llaccel.name = "l0.x1"} : tensor<?x128xf32>
    %h2 = llaccel.rmsnorm %x1, @l0_ffn_norm {eps = 1.0e-5 : f64, llaccel.name = "l0.h2"} : tensor<?x128xf32>
    %g  = llaccel.linear %h2, @l0_wg {llaccel.name = "l0.g"} : tensor<?x128xf32> -> tensor<?x384xf32>
    %u  = llaccel.linear %h2, @l0_wu {llaccel.name = "l0.u"} : tensor<?x128xf32> -> tensor<?x384xf32>
    %sg = llaccel.silu %g {llaccel.name = "l0.sg"} : tensor<?x384xf32>
    %f  = llaccel.mul %sg, %u {llaccel.name = "l0.f"} : tensor<?x384xf32>
    %d  = llaccel.linear %f, @l0_wd {llaccel.name = "l0.d"} : tensor<?x384xf32> -> tensor<?x128xf32>
    %x2 = llaccel.add %x1, %d {llaccel.name = "l0.x2"} : tensor<?x128xf32>
    ...
    %hn = llaccel.rmsnorm %xL, @norm {eps = 1.0e-5 : f64, llaccel.name = "hn"} : tensor<?x128xf32>
    %lg = llaccel.linear %hn, @lm_head {llaccel.name = "logits"} : tensor<?x128xf32> -> tensor<?x65xf32>
    return %lg : tensor<?x65xf32>
  }
}
```
Attention semantics: causal over the KV cache of `layer`; `%kr`/`%v` rows are
appended to the cache at positions `POS + m` before the query rows attend
(so a query at `POS+m` sees keys `0..POS+m`). GQA: query head `h` uses KV head
`h / (heads / kv_heads)`. Scale `1/sqrt(head_dim)`.

`weights.bin`: raw little-endian `f32`, one tensor after another;
`weights.json`: `[{"name": "l0_wq", "shape": [128, 128], "offset": 0}, ...]` (offsets in bytes).
`calib.json`: `{"input": 3.2, "l0.h": 4.1, ...}` — per-tensor abs-max (float).
`tokenizer.json`: `{"itos": ["\n", " ", "!", ...]}` (char-level) used by the runtime.

## 2. Quantized graph dump (`llaccel-compile --dump-qgraph DIR`)

Written after quantization + fusion + lowering decisions, before tiling.
The numpy golden model executes exactly this list (docs/NUMERICS.md) — so the
golden is the *specification-level* reference and the func-sim / RTL are the
*ISA-level* implementations of the same list.

`DIR/qgraph.json`:
```json
{
  "model": {"dim":128,"n_layers":4,"n_heads":4,"n_kv_heads":2,"head_dim":32,"ffn":384,
            "vocab":65,"vocab_padded":80,"max_seq":256,"E_RES":-8,"E_LOGIT":-6},
  "weights_file": "qweights.bin",
  "tensors": {
    "embed":      {"dtype":"i16","shape":[80,128],"offset":0,"exp":-8},
    "l0_attn_norm":{"dtype":"i16","shape":[128],"offset":..,"exp":-12},
    "l0_wq":      {"dtype":"i8","shape":[128,128],"offset":..},
    "l0_wq.rq":   {"dtype":"rq","shape":[128],"offset":..},      // int32 pairs {M,S} per channel
    "l0_bq":      {"dtype":"i32","shape":[128],"offset":..},
    "rope_cos":   {"dtype":"i16","shape":[256,16],"offset":..},  // [max_seq][head_dim/2] Q1.14
    "rope_sin":   {"dtype":"i16","shape":[256,16],"offset":..}
  },
  "ops": [
    {"op":"rmsnorm","in":"input","gamma":"l0_attn_norm","out":"l0.h","K":128,"eps_t":..,"C":..,"sh_post":..},
    {"op":"quant","in":"l0.h","out":"l0.h.q","M":..,"S":..},
    {"op":"linear","in":"l0.h.q","w":"l0_wq","rq":"l0_wq.rq","bias":"l0_bq","out":"l0.q","N":128,"K":128,
       "out_dtype":"i16","epilogue":"none","aux":null,"aux_shift":0,"silu":null},
    {"op":"linear", ... "out":"l0.v","out_dtype":"i8" ...},
    {"op":"rope","in":"l0.q","out":"l0.qr","H":4,"D":32},
    {"op":"quant","in":"l0.qr","out":"l0.qr.q","M":..,"S":..},
    {"op":"kv_write","layer":0,"k":"l0.kr.q","v":"l0.v","Hkv":2,"D":32},
    {"op":"attention","q":"l0.qr.q","layer":0,"out":"l0.a","H":4,"Hkv":2,"D":32,"Ms":..,"Ss":..,"Mo":..,"So":..},
    {"op":"linear","in":"l0.a","w":"l0_wo","rq":"l0_wo.rq","bias":null,"out":"l0.x1","N":128,"K":128,
       "out_dtype":"i16","epilogue":"resadd","aux":"input","aux_shift":0,"silu":null},        // v2 form
    {"op":"add","a":"input","b":"l0.o","out":"l0.x1","sh_b":0},                                // v1 form
    {"op":"silu","in":"l0.g","out":"l0.sg","Mi":..,"Si":..,"sh_out":..},
    {"op":"mul","a":"l0.sg","b":"l0.u","out":"l0.f","sh":..},
    ...
  ],
  "input": "input", "output": "logits",
  "exps": {"input":-8, "l0.h":-8, ...}, "scales": {"l0.h.q": 0.0322, ...}
}
```
`"silu"` on a fused linear is `{"Mi":..,"Si":..,"sh_out":..}`. Tensor names in
`ops` refer to activations; weights refer to `tensors`.

## 3. `.llbin` META_JSON (compiler → runtime)

```json
{
  "model": { same as qgraph.model },
  "dram": {
    "image_bytes": 1234567,
    "embedding": {"addr": 0, "row_bytes": 256, "rows": 80},           // i16 rows at E_RES
    "input":     {"addr": .., "row_bytes": 256, "rows": 16},          // host writes M rows (i16)
    "logits":    {"addr": .., "row_bytes": 160, "rows": 16}           // device writes M rows (i16 at E_LOGIT)
  },
  "programs": [{"M": 16, "pc": .., "n_instr": ..}, {"M": 1, "pc": .., "n_instr": ..}],
  "sram": {"bytes": 1048576, "peak_used": .., "kv_cache_bytes": .., "resident_const_bytes": ..},
  "stats": {"M16": {"instructions": .., "dma_load_bytes": .., "dma_store_bytes": .., "gemm_macs": ..},
            "M1": {...}},
  "target": "llaccel-v2", "fusion": true, "schedule": "overlap"
}
```
Programs live inside the DRAM image at `pc`. The runtime: write input rows →
set `POS` → run program `M` → read logits rows → argmax over `vocab`.

## 4. Clarifications (Python side: exporter, reference quantizer, golden)

Added 2026-09-10 while finishing `python/llaccel/*`; they pin down details the
sections above leave open. Where a rule below conflicts with a "Compiler:" line of
NUMERICS.md, the *device* formula of NUMERICS.md / `numerics.h` is authoritative and
the rule below is the parameter choice that makes it correct.

### 4.1 Exporter (`model.mlir`)
* Float attributes are printed with Python `repr` plus a mandatory `.` (`1.0e-05`,
  `10000.0`); MLIR parses them identically to the §1 spelling `1.0e-5`.
* Single-head layouts are accepted: `view(1, T, 1, D)` leaves no column dim for
  the head axis, and GQA replication of a single KV head (`repeat_interleave` /
  `expand`+`reshape` of a size-1 dim) is a zero-stride head dim; both denote the
  §1 semantics with `heads = 1` / `kv_heads = 1`.
* `calib.json` holds one entry per `llaccel.name` in `model.mlir` plus `input`
  (the embedding output). Abs-max is measured on the exported graph itself.

### 4.2 Reference quantizer (`llaccel.refquant`, twin of `llaccel-quantize`/`-fuse`)
* Rounding: every float→int decision uses C `llround` (round half away from
  zero), (M, S) pairs come from `frexp` normalisation with `M ∈ [2^30, 2^31)`
  (`S` pinned at 63 for tiny ratios), exactly as
  `compiler/include/llaccel/Support/QuantParams.h`.
* `E_RES = max` of the i16 exponents of `input`, every `l*.x1`, `l*.x2` **and the
  addends `l*.o`, `l*.d`**; all of them are emitted at `E_RES` (so `sh_b = 0`).
  `E_LOGIT` is the exponent of `logits`; `lm_head` and `embed` are padded with
  zero rows to `vocab_padded = ceil16(vocab)`.
* RoPE preserves the exponent: `e(q) = e(qr) = max(e_calib(q), e_calib(qr))`,
  same for `k`/`kr` (the rotation can raise the per-element abs-max by √2).
* i8 activations: `v` projections (`s = absmax(v)/127`, written straight into the
  KV cache), the attention output `a` (`s_out = absmax(a)/127`), and one `quant`
  op `name → name.q` (`s = absmax(name)/127`) in front of every i16-input linear
  (`h`, `h2`, `f`, `hn`) and in front of the KV write / attention (`qr`, `kr`).
* RMSNorm `R`: the largest `R ≤ 31` with `C = llround(2^R·√K) < 2^32` and the
  estimated `inv = floor(C / floor(√(K·rms_q² + eps_t)))` below `65535/4`, where
  `rms_q = (absmax_in / 4) / 2^e_x` (abs-max ≈ 4 standard deviations).
* SiLU: `Mi = 2^30`, `Si = 30 − (e_x + 12)`, `sh_out = 16 + e_y − e_x`.
* **MUL / ADD shift sign.** With `y = sat16(rshr(a·b, sh))` and `a·b` at exponent
  `e_a + e_b`, the shift is `sh = e_y − e_a − e_b` (e.g. −14 − (−13) − (−12) = 11).
  NUMERICS.md's "Compiler:" line writes `e_a + e_b − e_y`, which is the negation
  and is < 0 for every real exponent set. Likewise ADD needs `sh_b = e_a − e_b ≥ 0`
  (0 for residual adds at `E_RES`). The reference quantizer and
  `compiler/include/llaccel/Support/QuantParams.h` both implement the correct sign.
* Fusion (`--fusion`, v2): a fused linear's `rq` targets the exponent of the
  pre-epilogue tensor (`o`/`d` → `E_RES`, `g`, `u`), the op result carries the fused
  name/exponent (`x1`/`x2`, `sg`, `f`), `aux_shift` is the MUL shift (or 0 for
  RESADD), `silu` holds `{Mi, Si, sh_out}`. `qweights.bin` is byte-identical
  between the v1 and v2 dumps; only the op list changes. The qgraph carries an
  extra `"fusion": bool` and `exps` also lists the pre-epilogue names.
* `kv_write` has no output tensor; a linear without bias has `"bias": null`.

### 4.3 `golden.json` (golden → runtime `--verify`)
```json
{"prompt_tokens": [...], "generated": [...], "argmax_per_step": [...],
 "logits_last_rows": [[...i16 × vocab...], ...], "steps": [{"kind","pos","rows","valid_rows"}, ...]}
```
* One entry of `argmax_per_step` / `logits_last_rows` / `steps` per device
  launch: every M=16 prefill chunk (in order), then every M=1 decode launch.
  Mirrors `runtime/src/host.cpp Host::generate`: `n_tokens` decode launches, the
  first `generated` token is the last prefill chunk's argmax, the last decode
  launch's argmax is recorded in `argmax_per_step` but not appended to
  `generated`; a decode launch is skipped when `L + i + 1 ≥ max_seq`.
* A logits row holds the first `vocab` (not `vocab_padded`) i16 logits at
  `E_LOGIT` of the launch's relevant row: row `(L−1) mod 16` of the last prefill
  chunk, row 15 of earlier chunks, row 0 of a decode launch.
* Argmax is over those `vocab` entries, lowest index on ties.
* `--dump-acts DIR` writes `step{i:03d}_{prefill|decode}_pos{P}_{name}.npy`
  (int32, `[M][cols]`) for every named activation of every launch.
