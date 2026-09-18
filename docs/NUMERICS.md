# llaccel numerics contract

This document is the single definition of every integer operation the device
performs. Three implementations exist and are tested against each other
bit-for-bit: `python/llaccel/golden.py` (numpy), `include/llaccel/numerics.h`
(C++, used by the functional simulator and the Verilator testbenches), and the
SystemVerilog engines under `rtl/`. There is no floating point anywhere in the
device; every float→int decision is made once by the compiler. The optional
`GoldenModel(..., fast_matmul=True)` uses host FP64 BLAS only when a conservative
integer-dot-product bound guarantees exact representation, then restores integer
results. This is a reference acceleration, not device floating-point arithmetic.

## Primitives

```
rshr(v, s)   : round-half-up arithmetic right shift.
               s == 0 → v ; else (v >> s) + ((v >> (s-1)) & 1), s in [0,63]
               Equivalent to (v + 2^(s-1)) >> s over mathematical integers.
               Shift first to avoid overflowing i64 during rounding.
sat8(v)      : clamp to [-128, 127]         satu8(v)  : clamp to [0, 255]
sat16(v)     : clamp to [-32768, 32767]     satu16(v) : clamp to [0, 65535]
mulshift(v, M, S) = rshr(v * M, S)          v: i32/i48, M: i32 in [0, 2^31), S in [0, 63]; product in i64
isqrt(t)     : floor(sqrt(t)), t: u48 → u24
udiv(a, b)   : floor(a / b), b ≥ 1 (unsigned)
```

## Tensor formats

| name | storage | value | chosen by |
|---|---|---|---|
| `i8`  | int8, per-tensor scale `s` (float) | `q · s` | compiler (calibration abs-max / 127) |
| `i8pc`| int8 weights, per-output-channel scale `s[n]` | `q · s[n]` | compiler (max\|w[n,:]\| / 127) |
| `i16` | int16, per-tensor exponent `e` | `q · 2^e` | compiler (`e = max(ceil(log2(absmax / 32767)), -15)`) |
| `i32` | accumulator | — | — |

`E_RES` is the embedding/input exponent. Residual branches select exponents
per value with calibrated headroom for their immediate ADD result. ADD orders
the coarser operand first and right-shifts the finer operand. This preserves
early-layer precision when late layers have large outliers. Logits are `i16`
at `E_LOGIT`. Quantized graphs retain their explicit per-value exponents.

For a residual linear branch feeding ADD, the compiler chooses the maximum
of the calibrated branch exponent, calibrated sum exponent, calibrated other
operand exponent, and that operand's already assigned exponent. The sum uses
the coarser input exponent. Residual epilogue fusion is legal only when both
input exponents already match and `sh_b = 0`; otherwise ADD stays explicit.
Zero-range i16 tensors use exponent −15; zero-range i8 tensors use scale 1.
Compiler float-to-integer rounding uses nearest with ties away from zero,
distinct from the device's `rshr` rule.

Optional experimental calibration bounds `name.i8_absmax` and
`name.i16_absmax` override raw abs-max for the corresponding representation.
They must be finite, nonnegative, no larger than raw abs-max, and positive for
nonzero-range tensors. Missing bounds use raw abs-max. Raw maxima remain the
source for RMS reciprocal/range planning. The selected pretrained recipe leaves
these clipping overrides disabled: lower sampled reconstruction MSE did not
reliably improve end-to-end cross-entropy. Offline SmoothQuant is an exact float
reparameterization before quantization and introduces no additional device ops;
its calibration-dependent quality must be evaluated separately from integer
conformance.

## GEMM (engine: gemm)

Inputs: `A` i8 `[M][K]` row-major (row stride K bytes), `W` i8pc `[N][K]` in
tiled layout (see ISA.md), optional `bias` i32 `[N]`, requant table `RQ[N] =
{M_n: i32, S_n: i32}`, optional `AUX` i16 `[M][N]`.

```
acc[m][n] = Σ_k A[m][k] · W[n][k]                      i32 (K ≤ 65535: no overflow, |acc| ≤ 2^30)
if has_bias: acc[m][n] += bias[n]                       signed 33-bit (no wrap)
t[m][n]   = sat16( mulshift(acc[m][n], M_n, S_n) )      i16
epilogue (mode):
  NONE   : out = t                          (out dtype i16)   or   out = sat8( mulshift(acc, M_n, S_n) )  (out dtype i8)
  RESADD : out = sat16( t + AUX[m][n] )                              (i16; AUX at the same exponent, checked by the fusion pass)
  SILU   : out = silu16( t ; Mi, Si, sh_out )                        (i16; see SiLU below)
  MUL    : out = sat16( rshr( t · AUX[m][n], aux_shift ) )           (i16)
```
The bias addition is widened before requantization: RTL uses 33 signed bits
and the software references use i64. A full-range i32 bias must not wrap the
MAC accumulator. The subsequent product fits signed i64 for valid multipliers.

`i8` output is only valid with mode NONE. Modes other than NONE exist only on
target `llaccel-v2` (hardware parameter `EPILOGUE_FUSION = 1`).

Compiler: `M_n · 2^-S_n ≈ s_a · s_w[n] / 2^e_out` (i16 out) or `s_a · s_w[n] / s_out` (i8 out),
with `M_n` normalized to `[2^30, 2^31)`.

## RMSNorm (engine: vec)

Inputs: `x` i16 `[M][K]` (exponent `e_x`), `g` i16 `[K]` (exponent `e_g`),
params `eps_t: u32`, `C: u32`, `sh_post: u8`. Per row:

```
ss   = Σ_k x[k]²                       u48   (K · 2^30 ≤ 2^46)
tt   = ss + eps_t                      u48
r    = isqrt(tt)                       u24
inv  = min( 65535, udiv(C, max(r, 1)) )      u16
y[k] = sat16( rshr( (x[k] · g[k]) · inv , sh_post ) )     x·g: i32; ·inv: i48; rshr in i64
```
Compiler: `eps_t = round(eps · K · 2^(-2 e_x))` must fit u32.
Calibration may provide `<input_name>.rms_min`, the minimum RMS across observed
input rows. It must be finite and between zero and the input abs-max. When the
statistic is absent, the compiler retains the legacy estimate `absmax / 4`.

Let `rms_q = rms_min / 2^e_x` (or the fallback estimate in the same units), and
`r_est = max(1, floor(sqrt(K · rms_q² + eps_t)))`. Choose the largest `R` in
[0,31] for which `C = round(2^R · sqrt(K)) < 2^32` and
`floor(C / r_est) < 65535 / 4`. The fourfold margin reduces reciprocal
saturation on quieter rows; calibration does not guarantee behavior on every
unseen input. Finally, `sh_post = R - e_g + e_y` must be in [0,63].
The compiler and independent Python quantizer apply the same rule.

## RoPE (engine: vec) — HF `rotate_half` convention

Inputs: `x` i16 `[M][H·D]`, table rows `T[p] = [cos[0..D/2), sin[0..D/2)]` i16 in
Q1.14 (`1.0 = 16384`), row `p` at `table + p · table_stride`. For row `m`, the
position is `P = POS + m`. For each head `h` and `i < D/2`:

```
x1 = x[m][h·D + i]        x2 = x[m][h·D + i + D/2]
c  = cos[P][i]            s  = sin[P][i]
y[m][h·D + i]       = sat16( rshr( x1·c − x2·s , 14 ) )
y[m][h·D + i + D/2] = sat16( rshr( x2·c + x1·s , 14 ) )
```
Exponent preserved. `cos/sin[p][i] = round(2^14 · cos/sin(p · θ_i))`, `θ_i = base^(−2i/D)`.

## SiLU (engine: vec, and gemm epilogue on v2)

Input `x` i16 (exponent `e_x`); params `Mi: i32, Si: u8, sh_out: u8`.
`SIG[j] = round( σ(−8 + j/16) · 65536 )` for `j = 0..256`, stored as u16 (σ(8)·65536 = 65514).

```
u   = sat16( mulshift(x, Mi, Si) )            u is x in Q3.12  (real = u / 4096, range [−8, 8))
idx = (u >> 8) + 128                          arithmetic shift; idx ∈ [0, 255]
f   = u & 255
sg  = SIG[idx] + ( ( (SIG[idx+1] − SIG[idx]) · f ) >> 8 )        u16 (σ is monotone → diff ≥ 0)
y   = sat16( rshr( x · sg , sh_out ) )        x·sg: i32 (i16 · u16)
```
Compiler: `Mi · 2^-Si ≈ 2^(e_x + 12)`; `sh_out = 16 + e_y − e_x`.

## Elementwise (engine: vec)

```
MUL   : y = sat16( rshr( a · b , sh ) )                 a, b, y: i16 ;  compiler: sh = e_y − e_a − e_b (sh ∈ [0,63])
ADD   : y = sat16( a + rshr( b , sh_b ) )               sh_b = e_a − e_b ≥ 0 ; e_y = e_a
QUANT : y = sat8( mulshift( x , M , S ) )               i16 → i8 ; M·2^-S ≈ 2^e_x / s_y
```
Counts are multiples of 16 elements.

## Attention (engine: attn) — causal, GQA, INT8 KV cache

Inputs: `q` i8 `[M][H·D]` (scale `s_q`), K/V cache i8 (scales `s_k`, `s_v`):
`k[kvh][t][d]` at `kbase + kvh·kv_stride + t·D`, same for `v` at `vbase`.
K/V bases are full 32-bit DRAM addresses in binary format version 2; query/output
rows reside in SRAM. The RTL traverses fixed 256-key tiles: a global maximum
pass, a global exponential-sum pass, then a probability/value accumulation
pass. Maxima, sums and output accumulators span all tiles; normalization is
never performed independently per tile. The compiler currently caps context at
4096, independently of the fixed on-chip tile capacity.
Params `Ms, Ss, Mo, So`. Tables `EXPI[i] = round(e^{−i} · 65535)` for `i = 0..15`,
`EXPF[f] = round(e^{−f/256} · 65535)` for `f = 0..255` (u16).

For each row `m` (`P = POS + m`, `T = P + 1` keys) and head `h` (`kvh = h / (H / Hkv)`):
```
s[t]  = Σ_d q[m][h·D + d] · k[kvh][t][d]          i32, t < T
mx    = max_t s[t]
z[t]  = mulshift( mx − s[t] , Ms , Ss )            nonnegative i64 (mx − s ≥ 0)
p[t]  = z[t] ≥ 4096 ? 0 : ( EXPI[z[t] >> 8] · EXPF[z[t] & 255] + 2^15 ) >> 16      u16 (max 65534)
sum   = Σ_t p[t]                                   u32
inv   = udiv( 2^31 , sum )                         u16 (sum ≥ 65534 ⇒ inv ≤ 32769)
legacy (ATTN flags bit1 = 0):
  pn[t] = satu8( rshr(p[t] · inv, 23) )            u8, Q0.8
  o[d]  = Σ_t pn[t] · v[kvh][t][d]                 i32
wide (ATTN flags bit1 = 1; compiler default):
  pn[t] = min(32767, rshr(p[t] · inv, 16))         u15, Q0.15
  o[d]  = rshr(Σ_t pn[t] · v[kvh][t][d], 7)        i32, rescaled to Q0.8
out[m][h·D + d] = sat8( mulshift( o[d] , Mo , So ) )
```
Compiler: `Ms · 2^-Ss ≈ 256 · s_q · s_k / sqrt(D)`; `Mo · 2^-So ≈ s_v / (256 · s_out)`.

The maximum is subtracted from raw i32 dot products before scaling; scores
are not clipped to i16 first. Thus a representable common score offset leaves
the output unchanged. With `D ≤ 64`, the raw dot-product difference fits i32;
the multiply and rounded shift use i64 before comparison with the exponential
cutoff. Directed Python and RTL tests cover large positive/negative offsets,
nonzero score gaps, and maximal unsigned multipliers.

The wide mode preserves `Mo/So` scaling and ISA operands. The narrow
probability mode is selected by a clear flag within the current binary format;
version-1 images from the earlier SRAM-KV architecture are not compatible with
the DRAM-KV interface. New compiler qgraphs
record `prob_bits: 15`; absent fields mean legacy 8. Separately rounding many
probabilities to Q0.8 causes normalization bias: 171 equal keys with V=127
produce 85. Q0.15 produces 127. All context lengths 1..256 with constant signed
values are checked to within one output LSB. Probabilities need not sum to
exact unity; the remaining reciprocal and rounding errors are measured rather
than described as exact floating-point softmax.

## KV write (engine: attn)

`src` i8 `[M][Hkv·D]` → `base + kvh·kv_stride + (POS + m)·D` for each `m, kvh` (D bytes each).

## Host-side

* Embedding lookup: host copies row `tok` of the i16 embedding table (in the
  DRAM image) into the input buffer; pad rows of a prefill chunk are zeros.
* Argmax over the first `vocab_size` logits (padded channels are ignored), lowest
  index wins ties. Greedy decoding only.
