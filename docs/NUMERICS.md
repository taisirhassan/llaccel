# llaccel numerics contract

This document is the single definition of every integer operation the device
performs. Three implementations exist and are tested against each other
bit-for-bit: `python/llaccel/golden.py` (numpy), `include/llaccel/numerics.h`
(C++, used by the functional simulator and the Verilator testbenches), and the
SystemVerilog engines under `rtl/`. There is no floating point anywhere in the
device; every float→int decision is made once by the compiler.

## Primitives

```
rshr(v, s)   : round-half-up arithmetic right shift.
               s == 0 → v ; else (v + (1 << (s-1))) >> s   (arithmetic shift, v is i64)
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
| `i8pc`| int8 weights, per-output-channel scale `s[n]` | `q · s[n]` | exporter (max\|w[n,:]\| / 127) |
| `i16` | int16, per-tensor exponent `e` | `q · 2^e` | compiler (`e = ceil(log2(absmax / 32767))`) |
| `i32` | accumulator | — | — |

The residual stream uses a single exponent `E_RES` for the whole model
(max over all residual tensors) so residual adds need no shifts.
Embedding rows are `i16` at `E_RES`. Logits are `i16` at `E_LOGIT`.

## GEMM (engine: gemm)

Inputs: `A` i8 `[M][K]` row-major (row stride K bytes), `W` i8pc `[N][K]` in
tiled layout (see ISA.md), optional `bias` i32 `[N]`, requant table `RQ[N] =
{M_n: i32, S_n: i32}`, optional `AUX` i16 `[M][N]`.

```
acc[m][n] = Σ_k A[m][k] · W[n][k]                      i32 (K ≤ 65535: no overflow, |acc| ≤ 2^30)
if has_bias: acc[m][n] += bias[n]                       i32
t[m][n]   = sat16( mulshift(acc[m][n], M_n, S_n) )      i16
epilogue (mode):
  NONE   : out = t                          (out dtype i16)   or   out = sat8( mulshift(acc, M_n, S_n) )  (out dtype i8)
  RESADD : out = sat16( t + AUX[m][n] )                              (i16; AUX at the same exponent, guaranteed by E_RES)
  SILU   : out = silu16( t ; Mi, Si, sh_out )                        (i16; see SiLU below)
  MUL    : out = sat16( rshr( t · AUX[m][n], aux_shift ) )           (i16)
```
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
Compiler: `C = round(2^R · sqrt(K))` with `R` chosen so that `C < 2^32` and
`inv` stays below 65535 for calibrated inputs; `eps_t = round(eps · K · 2^(-2 e_x))`;
`sh_post = R - e_g + e_y` (must be in [0, 63]).

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
Params `Ms, Ss, Mo, So`. Tables `EXPI[i] = round(e^{−i} · 65535)` for `i = 0..15`,
`EXPF[f] = round(e^{−f/256} · 65535)` for `f = 0..255` (u16).

For each row `m` (`P = POS + m`, `T = P + 1` keys) and head `h` (`kvh = h / (H / Hkv)`):
```
s[t]  = Σ_d q[m][h·D + d] · k[kvh][t][d]          i32, t < T
mx    = max_t s[t]
z[t]  = mulshift( mx − s[t] , Ms , Ss )            u32 (mx − s ≥ 0)
p[t]  = z[t] ≥ 4096 ? 0 : ( EXPI[z[t] >> 8] · EXPF[z[t] & 255] + 2^15 ) >> 16      u16 (max 65534)
sum   = Σ_t p[t]                                   u32
inv   = udiv( 2^31 , sum )                         u16 (sum ≥ 65534 ⇒ inv ≤ 32769)
pn[t] = satu8( ( p[t] · inv + 2^22 ) >> 23 )       u8, Q0.8
o[d]  = Σ_t pn[t] · v[kvh][t][d]                   i32
out[m][h·D + d] = sat8( mulshift( o[d] , Mo , So ) )
```
Compiler: `Ms · 2^-Ss ≈ 256 · s_q · s_k / sqrt(D)`; `Mo · 2^-So ≈ s_v / (256 · s_out)`.

## KV write (engine: attn)

`src` i8 `[M][Hkv·D]` → `base + kvh·kv_stride + (POS + m)·D` for each `m, kvh` (D bytes each).

## Host-side

* Embedding lookup: host copies row `tok` of the i16 embedding table (in the
  DRAM image) into the input buffer; pad rows of a prefill chunk are zeros.
* Argmax over the first `vocab_size` logits (padded channels are ignored), lowest
  index wins ties. Greedy decoding only.
