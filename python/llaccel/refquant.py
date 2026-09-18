"""Reference quantizer: export dir -> qgraph.json + qweights.bin (docs/DIALECT.md section 2).

    uv run python -m llaccel.refquant build/export -o build/q-ref/ [--fusion]

What this is
------------
A Python re-implementation of the *decisions* of the C++ `llaccel-quantize` (+
`llaccel-fuse`) passes, producing the same `qgraph.json` / `qweights.bin` dump the
compiler writes with `--dump-qgraph`. It is not the compiler: it exists so that

* the numpy golden model, `llaccel.verify` and the end-to-end tests can run before /
  independently of the C++ compiler, and
* the C++ compiler can be diffed against it (same tensor names, same op list, same
  integer parameters) — the quantization recipe is written down once in executable form.

Every parameter formula is the "Compiler:" line of docs/NUMERICS.md, implemented in
section "QuantParams.h twins" below as a line-by-line twin of the C++ header
`compiler/include/llaccel/Support/QuantParams.h` (same rounding: C `llround`, i.e.
round-half-away-from-zero; same evaluation order of the double arithmetic; same
`frexp`-based (M, S) normalisation; same R-selection rule for RMSNorm).

Policies (docs/DIALECT.md section 2 and its clarifications, docs/PLAN.md):

* i16 activations: `e = max(ceil(log2(absmax / 32767)), -15)` from calib.json.
* `E_RES` is the input/embedding exponent. Each residual branch reserves its immediate
  ADD result headroom; ADD puts the coarser operand first and right-shifts the finer one.
  `E_LOGIT` = exponent of `logits`.
* RoPE preserves the exponent: `e(q) = e(qr) = max(e_calib(q), e_calib(qr))`, same for k/kr.
* `v` projections are emitted i8 (`s = absmax / 127`) straight into the KV cache;
  attention output `a` is i8 (`s_out = absmax(a) / 127`).
* A `quant` op (`name` -> `name.q`, `s = absmax(name) / 127`) is inserted in front of every
  linear whose input is i16 (`h`, `h2`, `f`, `hn`) and in front of the KV write / attention
  (`qr`, `kr`).
* Weights: int8 per-output-channel, `s[n] = max|W[n,:]| / 127` (all-zero rows -> 1).
  `lm_head` and `embed` are padded to `vocab_padded` (multiple of 16) rows.
* Bias: i32 `llround(b / (s_a * s_w[n]))`.
* `--fusion` (v2): `linear+add` -> epilogue `resadd`, `linear+silu` -> `silu`,
  `linear+mul` -> `mul`; the linear's requant targets the exponent of the pre-epilogue
  tensor (`o`/`d` at their own residual exponent, `g`, `u`), the op result carries the fused tensor's name and
  exponent (`x1`/`x2`, `sg`, `f`). Residual fusion requires equal exponents;
  shifted residual ADDs stay explicit. The arithmetic is identical to the unfused ops.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from .golden import GoldenModel

INT32_MAX, INT32_MIN = 2**31 - 1, -(2**31)


# --------------------------------------------------------------------------------------
# C rounding
# --------------------------------------------------------------------------------------
def llround(x: float) -> int:
    """C `llround`: nearest integer, halves away from zero (Python's round() is half-even)."""
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"llround of {x}")
    t = math.trunc(x)
    frac = x - t  # exact for |x| < 2^52
    if frac >= 0.5:
        return t + 1
    if frac <= -0.5:
        return t - 1
    return t


def llround_arr(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("llround_arr requires finite values")
    if np.any(x >= float(2**63)) or np.any(x < -float(2**63)):
        raise ValueError("llround_arr input exceeds int64 range")
    t = np.trunc(x)
    frac = x - t
    return np.where(frac >= 0.5, t + 1, np.where(frac <= -0.5, t - 1, t)).astype(np.int64)


# --------------------------------------------------------------------------------------
# QuantParams.h twins (one function per C++ function, same name in snake_case)
# --------------------------------------------------------------------------------------
def normalize_mulshift(r: float) -> tuple[int, int]:
    """(M, S) with M * 2^-S ~= r and M in [2^30, 2^31) (QuantParams.h normalizeMulShift)."""
    if not (r > 0.0) or not math.isfinite(r):
        raise ValueError("normalize_mulshift: ratio must be positive and finite")
    f, e = math.frexp(r)  # r = f * 2^e, f in [0.5, 1)
    M = llround(math.ldexp(f, 31))
    S = 31 - e
    if M >= 1 << 31:  # f rounded up to 1.0
        M = 1 << 30
        S -= 1
    if S < 0:
        raise ValueError(f"normalize_mulshift: ratio {r} too large (S < 0)")
    if S > 63:  # tiny ratio: keep S = 63 and accept a smaller M (may be 0)
        M = llround(math.ldexp(r, 63))
        S = 63
    return M, S


def i16_exponent(absmax: float) -> int:
    """e = max(ceil(log2(absmax / 32767)), -15)."""
    if not (absmax > 0.0):
        return -15
    return max(int(math.ceil(math.log2(absmax) - math.log2(32767.0))), -15)


def i8_scale(absmax: float) -> float:
    """absmax / 127 (absmax 0 -> 1: the tensor is all zero)."""
    return absmax / 127.0 if absmax > 0.0 else 1.0


def quant_i8(v: np.ndarray, scale) -> np.ndarray:
    v, scale = np.asarray(v, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("quant_i8 requires finite values and positive finite scales")
    with np.errstate(over="ignore"):
        scaled = np.clip(v / scale, -128, 127)
    return llround_arr(scaled).astype(np.int8)


def quant_i16(v: np.ndarray, e: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        raise ValueError("quant_i16 requires finite values")
    with np.errstate(over="ignore"):
        scaled = np.clip(np.ldexp(v, -int(e)), -32768, 32767)
    return llround_arr(scaled).astype(np.int16)


def weight_row_scales(w: np.ndarray) -> np.ndarray:
    """Per-output-channel scale max|W[n,:]| / 127, zero rows -> 1 (QuantParams.h weightRowScale)."""
    mx = np.abs(np.asarray(w, dtype=np.float64)).max(axis=1)
    return np.where(mx > 0.0, mx / 127.0, 1.0)


def gemm_rq(s_a: float, s_w: float, out_i8: bool, e_out: int, s_out: float) -> tuple[int, int]:
    """i16 out: M*2^-S ~= s_a*s_w / 2^e_out ; i8 out: s_a*s_w / s_out."""
    r = s_a * s_w / s_out if out_i8 else s_a * s_w / math.ldexp(1.0, int(e_out))
    return normalize_mulshift(r)


def gemm_bias(b: float, s_a: float, s_w: float) -> int:
    v = llround(b / (s_a * s_w))
    return max(INT32_MIN, min(INT32_MAX, v))


def quant_params(e_x: int, s_y: float) -> tuple[int, int]:
    """QUANT i16 -> i8: M*2^-S ~= 2^e_x / s_y."""
    return normalize_mulshift(math.ldexp(1.0, int(e_x)) / s_y)


def rmsnorm_params(K: int, eps: float, e_x: int, e_g: int, e_y: int, absmax_x: float, name: str, rms_min: float | None = None) -> dict:
    """eps_t = llround(eps K 2^(-2 e_x)); R = largest <= 31 with C = llround(2^R sqrt(K)) < 2^32 and the
    estimated inv below 65535/4 using measured rms_min, or absmax/4 when absent;
    sh_post = R - e_g + e_y."""
    if rms_min is not None and (not math.isfinite(rms_min) or not 0 <= rms_min <= absmax_x):
        raise ValueError(f"rmsnorm {name}: rms_min must be finite and in [0, absmax]")
    eps_t = llround(eps * float(K) * math.ldexp(1.0, int(-2 * e_x)))
    if not 0 <= eps_t < 2**32:
        raise ValueError(f"rmsnorm {name}: eps_t does not fit the ISA u32 operand")
    rms_q = (absmax_x / 4.0 if rms_min is None else rms_min) / math.ldexp(1.0, int(e_x))
    r_typ = math.floor(math.sqrt(float(K) * rms_q * rms_q + float(eps_t)))
    if r_typ < 1.0:
        r_typ = 1.0
    R, C = -1, 0
    for cand in range(31, -1, -1):
        Cc = llround(math.ldexp(math.sqrt(float(K)), cand))
        if Cc >= 4294967296:
            continue
        inv_typ = math.floor(Cc / r_typ)
        if inv_typ < 65535.0 / 4.0:
            R, C = cand, Cc
            break
    if R < 0:
        raise ValueError(f"rmsnorm {name}: no R in [0,31] keeps inv below 65535/4 (input abs-max {absmax_x} too small for e_x)")
    sh_post = R - e_g + e_y
    if not 0 <= sh_post <= 63:
        raise ValueError(f"rmsnorm {name}: sh_post = {sh_post} outside [0, 63]")
    return {"eps_t": eps_t, "C": C, "R": R, "sh_post": sh_post}


def silu_params(e_x: int, e_y: int, name: str) -> dict:
    """Mi = 2^30, Si = 30 - (e_x + 12) (Mi*2^-Si = 2^(e_x+12)); sh_out = 16 + e_y - e_x."""
    Mi, Si, sh_out = 1 << 30, 30 - (e_x + 12), 16 + e_y - e_x
    if not 0 <= Si <= 63:
        raise ValueError(f"silu {name}: Si = {Si} outside [0, 63]")
    if not 0 <= sh_out <= 63:
        raise ValueError(f"silu {name}: sh_out = {sh_out} outside [0, 63]")
    return {"Mi": Mi, "Si": Si, "sh_out": sh_out}


def mul_shift(e_a: int, e_b: int, e_y: int, name: str) -> int:
    """MUL: y = sat16(rshr(a*b, sh)); a*b has exponent e_a + e_b, so sh = (e_a + e_b) - e_y ... expressed as a
    *right* shift: sh = e_y - e_a - e_b.

    NOTE: NUMERICS.md's "Compiler:" line (and QuantParams.h `mulShift`) write `e_a + e_b - e_y`, which is the
    negative of the shift the device formula needs (it is < 0 for every real exponent set, e.g. -13-12+14).
    The device formula (numerics.h `vmul`) is authoritative; see the DIALECT.md clarifications."""
    sh = e_y - e_a - e_b
    if not 0 <= sh <= 63:
        raise ValueError(f"mul {name}: sh = {sh} outside [0, 63]")
    return sh


def add_shift(e_a: int, e_b: int, name: str) -> int:
    """ADD: y = sat16(a + rshr(b, sh_b)), e_y = e_a, so b must be brought from e_b to e_a: sh_b = e_a - e_b >= 0.
    The coarser operand is ordered first by the quantizer."""
    sh = e_a - e_b
    if not 0 <= sh <= 63:
        raise ValueError(f"add {name}: sh_b = {sh} outside [0, 63] (b must not have a larger exponent than a)")
    return sh


def attn_params(s_q: float, s_k: float, s_v: float, s_out: float, D: int) -> dict:
    Ms, Ss = normalize_mulshift(256.0 * s_q * s_k / math.sqrt(float(D)))
    Mo, So = normalize_mulshift(s_v / (256.0 * s_out))
    return {"Ms": Ms, "Ss": Ss, "Mo": Mo, "So": So}


def rope_tables(max_seq: int, D: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin[p][i] = llround(2^14 cos/sin(p * theta_i)), theta_i = base^(-2i/D); [max_seq][D/2] i16."""
    half = D // 2
    cos_t = np.zeros((max_seq, half), dtype=np.int16)
    sin_t = np.zeros((max_seq, half), dtype=np.int16)
    for i in range(half):
        theta = math.pow(base, -2.0 * float(i) / float(D))
        for p in range(max_seq):
            ang = float(p) * theta
            cos_t[p, i] = min(llround(16384.0 * math.cos(ang)), 32767)
            sin_t[p, i] = min(llround(16384.0 * math.sin(ang)), 32767)
    return cos_t, sin_t


# --------------------------------------------------------------------------------------
# qweights.bin writer
# --------------------------------------------------------------------------------------
class Blob:
    """Concatenates tensors (each padded to 64 B) and records the DIALECT.md `tensors` entries."""

    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.off = 0
        self.tensors: dict[str, dict] = {}

    def add(self, name: str, arr: np.ndarray, dtype: str, exp: int | None = None) -> None:
        if name in self.tensors:
            raise ValueError(f"duplicate tensor {name}")
        raw = {"i8": arr.astype("<i1"), "i16": arr.astype("<i2"), "i32": arr.astype("<i4"), "rq": arr.astype("<i4")}[dtype]
        b = raw.tobytes()
        spec = {"dtype": dtype, "shape": [int(s) for s in (arr.shape[:-1] if dtype == "rq" else arr.shape)], "offset": self.off}
        if exp is not None:
            spec["exp"] = int(exp)
        self.tensors[name] = spec
        self.parts.append(b)
        self.off += len(b)
        pad = (-len(b)) % 64
        if pad:
            self.parts.append(b"\0" * pad)
            self.off += pad

    def bytes(self) -> bytes:
        return b"".join(self.parts)


# --------------------------------------------------------------------------------------
# export dir reader
# --------------------------------------------------------------------------------------
def parse_model_attrs(mlir: str) -> dict:
    m = re.search(r"llaccel\.model\s*=\s*\{(.*?)\}\}", mlir, re.S)
    if not m:
        raise ValueError("model.mlir: llaccel.model attribute not found")
    out = {}
    for k, v, ty in re.findall(r"(\w+)\s*=\s*([-+0-9.eE]+)\s*:\s*(i64|f64)", m.group(1)):
        out[k] = int(v) if ty == "i64" else float(v)
    return out


def load_export(export_dir: Path) -> tuple[dict, dict[str, np.ndarray], dict[str, float]]:
    """(model attrs, {weight name: f64 array (exact f32 values)}, {llaccel.name: absmax})."""
    export_dir = Path(export_dir)
    cfg = parse_model_attrs((export_dir / "model.mlir").read_text())
    blob = (export_dir / "weights.bin").read_bytes()
    with open(export_dir / "weights.json") as f:
        idx = json.load(f)
    weights = {}
    for e in idx:
        n = int(np.prod(e["shape"]))
        weights[e["name"]] = np.frombuffer(blob, dtype="<f4", count=n, offset=e["offset"]).reshape(e["shape"]).astype(np.float64)
    with open(export_dir / "calib.json") as f:
        calib = json.load(f)
    return cfg, weights, calib


# --------------------------------------------------------------------------------------
# the quantizer
# --------------------------------------------------------------------------------------
class RefQuantizer:
    def __init__(self, cfg: dict, weights: dict[str, np.ndarray], calib: dict[str, float], fusion: bool) -> None:
        self.cfg, self.W, self.calib, self.fusion = cfg, weights, calib, fusion
        self.blob = Blob()
        self.ops: list[dict] = []
        self.exps: dict[str, int] = {}
        self.scales: dict[str, float] = {}

    # ---- activation formats ------------------------------------------------------------
    def absmax(self, name: str) -> float:
        if name not in self.calib:
            raise KeyError(f"calib.json has no entry for activation {name!r}")
        value = float(self.calib[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"calibration for {name!r} must be finite and nonnegative")
        return value

    def quant_absmax(self, name: str, bits: int) -> float:
        raw = self.absmax(name)
        key = name + f".i{bits}_absmax"
        bound = self.absmax(key) if key in self.calib else raw
        if bound > raw or (raw > 0 and bound == 0):
            raise ValueError(f"invalid selected quantization bound for {name!r}")
        return bound

    def e16(self, name: str) -> int:
        if name not in self.exps:
            self.exps[name] = i16_exponent(self.quant_absmax(name, 16))
        return self.exps[name]

    def s8(self, name: str, src: str | None = None) -> float:
        """Scale of i8 activation `name`, calibrated from `src` (default: `name` minus a `.q` suffix)."""
        if name not in self.scales:
            self.scales[name] = i8_scale(self.quant_absmax(src or name.removesuffix(".q"), 8))
        return self.scales[name]

    # ---- op emitters -----------------------------------------------------------------------
    def rmsnorm(self, x: str, gamma: str, y: str, heads: int = 1) -> None:
        e_x, e_y = self.e16(x), self.e16(y)
        g = self.W[gamma]
        e_g = i16_exponent(float(np.abs(g).max()))
        self.blob.add(gamma, quant_i16(g, e_g), "i16", e_g)
        K = int(g.shape[0])
        rms_min = self.absmax(x + ".rms_min") if x + ".rms_min" in self.calib else None
        p = rmsnorm_params(K, self.cfg["rms_eps"], e_x, e_g, e_y, self.absmax(x), y, rms_min)
        self.ops.append({"op": "rmsnorm", "in": x, "gamma": gamma, "out": y, "K": K, "eps_t": p["eps_t"], "C": p["C"],
                         "sh_post": p["sh_post"], **({"heads": heads} if heads != 1 else {})})

    def quant(self, x: str) -> str:
        y = x + ".q"
        M, S = quant_params(self.e16(x), self.s8(y, x))
        self.ops.append({"op": "quant", "in": x, "out": y, "M": M, "S": S})
        return y

    def linear(self, a: str, w: str, b: str | None, y: str, out_dtype: str, *, e_rq: int | None = None, N_pad: int | None = None,
               epilogue: str = "none", aux: str | None = None, aux_shift: int = 0, silu: dict | None = None) -> None:
        """`e_rq`: exponent the requant targets (i16 out; defaults to e(y)); the op's `out` is `y`."""
        w_f = self.W[w]
        if N_pad is not None and N_pad != w_f.shape[0]:
            w_f = np.concatenate([w_f, np.zeros((N_pad - w_f.shape[0], w_f.shape[1]))], axis=0)
        N, K = (int(s) for s in w_f.shape)
        s_w = weight_row_scales(w_f)
        self.blob.add(w, quant_i8(w_f, s_w[:, None]), "i8")
        s_a = self.s8(a)
        if out_dtype == "i16":
            e_out = self.e16(y) if e_rq is None else e_rq
            rq = [gemm_rq(s_a, float(s_w[n]), False, e_out, 0.0) for n in range(N)]
        elif out_dtype == "i8":
            if epilogue != "none":
                raise ValueError(f"linear {y}: i8 output is only valid with epilogue none")
            rq = [gemm_rq(s_a, float(s_w[n]), True, 0, self.s8(y)) for n in range(N)]
        else:
            raise ValueError(out_dtype)
        self.blob.add(w + ".rq", np.array(rq, dtype=np.int64), "rq")
        bias = None
        if b is not None and b in self.W:
            bq = np.zeros(N)
            bq[: self.W[b].shape[0]] = self.W[b]
            self.blob.add(b, np.array([gemm_bias(float(bq[n]), s_a, float(s_w[n])) for n in range(N)], dtype=np.int32), "i32")
            bias = b
        self.ops.append({"op": "linear", "in": a, "w": w, "rq": w + ".rq", "bias": bias, "out": y, "N": N, "K": K,
                         "out_dtype": out_dtype, "epilogue": epilogue, "aux": aux, "aux_shift": aux_shift, "silu": silu})

    def residual_linear(self, a: str, w: str, bias: str, branch: str, residual: str, out: str) -> None:
        """Reserve this ADD's headroom in the branch GEMM, then align its operand scales."""
        e_residual = self.e16(residual)
        e_branch = max(self.e16(branch), self.e16(out), e_residual,
                       i16_exponent(self.quant_absmax(residual, 16)))
        self.exps[branch] = self.exps[out] = e_branch
        if self.fusion and e_branch == e_residual:
            self.linear(a, w, bias, out, "i16", e_rq=e_branch, epilogue="resadd",
                        aux=residual, aux_shift=0)
        else:
            self.linear(a, w, bias, branch, "i16")
            first, second = (residual, branch) if e_residual == e_branch else (branch, residual)
            self.ops.append({"op": "add", "a": first, "b": second, "out": out,
                             "sh_b": add_shift(self.e16(first), self.e16(second), out)})

    # ---- the graph -----------------------------------------------------------------------------
    def run(self) -> dict:
        c = self.cfg
        dim, L, H, Hkv, D, ffn, vocab, max_seq = (int(c[k]) for k in ("dim", "n_layers", "n_heads", "n_kv_heads", "head_dim",
                                                                    "ffn", "vocab", "max_seq"))
        vocab_padded = (vocab + 15) // 16 * 16
        for what, val in (("dim", dim), ("ffn", ffn), ("n_kv_heads*head_dim", Hkv * D)):
            if val % 16:
                raise ValueError(f"{what} = {val} must be a multiple of 16")
        if D not in (16, 32, 64):
            raise ValueError(f"head_dim {D} not in (16, 32, 64) (ISA.md ATTN)")

        # E_RES describes embedding/input only. Later outliers must not reduce
        # the precision of every earlier residual.
        E_RES = self.e16("input")
        E_LOGIT = i16_exponent(self.quant_absmax("logits", 16))
        self.exps["logits"] = E_LOGIT

        # constants: embedding (padded rows), RoPE tables
        emb = np.zeros((vocab_padded, dim))
        emb[:vocab] = self.W["embed"]
        self.blob.add("embed", quant_i16(emb, E_RES), "i16", E_RES)
        if "rope_cos_input" in self.W or "rope_sin_input" in self.W:
            tables = []
            for name in ("rope_cos_input", "rope_sin_input"):
                table = self.W[name]
                if table.shape != (max_seq, D // 2) or not np.isfinite(table).all() or ((table < -2.0) | (table > 32767.0 / 16384.0)).any():
                    raise ValueError(f"invalid static RoPE table {name}")
                tables.append(llround_arr(table * 16384.0).astype(np.int16))
            cos_t, sin_t = tables
        else:
            cos_t, sin_t = rope_tables(max_seq, D, float(c["rope_base"]))
        self.blob.add("rope_cos", cos_t, "i16")
        self.blob.add("rope_sin", sin_t, "i16")

        residual = "input"
        for i in range(L):
            p = f"l{i}"
            self.rmsnorm(residual, f"{p}_attn_norm", f"{p}.h")
            hq = self.quant(f"{p}.h")
            qk_norm = f"{p}_q_norm" in self.W
            if qk_norm != (f"{p}_k_norm" in self.W):
                raise ValueError("Q/K normalization must be paired")
            q_src, k_src = (f"{p}.qn", f"{p}.kn") if qk_norm else (f"{p}.q", f"{p}.k")
            for a, b in ((q_src, f"{p}.qr"), (k_src, f"{p}.kr")):  # RoPE preserves the exponent
                self.exps[a] = self.exps[b] = max(i16_exponent(self.quant_absmax(a, 16)), i16_exponent(self.quant_absmax(b, 16)))
            self.linear(hq, f"{p}_wq", f"{p}_bq", f"{p}.q", "i16")
            self.linear(hq, f"{p}_wk", f"{p}_bk", f"{p}.k", "i16")
            self.linear(hq, f"{p}_wv", f"{p}_bv", f"{p}.v", "i8")
            if qk_norm:
                self.rmsnorm(f"{p}.q", f"{p}_q_norm", q_src, heads=H)
                self.rmsnorm(f"{p}.k", f"{p}_k_norm", k_src, heads=Hkv)
            self.ops.append({"op": "rope", "in": q_src, "out": f"{p}.qr", "H": H, "D": D})
            self.ops.append({"op": "rope", "in": k_src, "out": f"{p}.kr", "H": Hkv, "D": D})
            qq, kq = self.quant(f"{p}.qr"), self.quant(f"{p}.kr")
            self.ops.append({"op": "kv_write", "layer": i, "k": kq, "v": f"{p}.v", "Hkv": Hkv, "D": D})
            self.ops.append({"op": "attention", "q": qq, "layer": i, "out": f"{p}.a", "H": H, "Hkv": Hkv, "D": D, "prob_bits": 15,
                             **attn_params(self.s8(qq), self.s8(kq), self.s8(f"{p}.v"), self.s8(f"{p}.a"), D)})
            self.residual_linear(f"{p}.a", f"{p}_wo", f"{p}_bo", f"{p}.o", residual, f"{p}.x1")
            self.rmsnorm(f"{p}.x1", f"{p}_ffn_norm", f"{p}.h2")
            h2q = self.quant(f"{p}.h2")
            e_g, e_u, e_sg, e_f = (self.e16(f"{p}.{n}") for n in ("g", "u", "sg", "f"))
            silu = silu_params(e_g, e_sg, f"{p}.sg")
            sh_mul = mul_shift(e_sg, e_u, e_f, f"{p}.f")
            if self.fusion:
                self.linear(h2q, f"{p}_wg", f"{p}_bg", f"{p}.sg", "i16", e_rq=e_g, epilogue="silu", silu=silu)
                self.linear(h2q, f"{p}_wu", f"{p}_bu", f"{p}.f", "i16", e_rq=e_u, epilogue="mul", aux=f"{p}.sg", aux_shift=sh_mul)
            else:
                self.linear(h2q, f"{p}_wg", f"{p}_bg", f"{p}.g", "i16")
                self.linear(h2q, f"{p}_wu", f"{p}_bu", f"{p}.u", "i16")
                self.ops.append({"op": "silu", "in": f"{p}.g", "out": f"{p}.sg", **silu})
                self.ops.append({"op": "mul", "a": f"{p}.sg", "b": f"{p}.u", "out": f"{p}.f", "sh": sh_mul})
            fq = self.quant(f"{p}.f")
            self.residual_linear(fq, f"{p}_wd", f"{p}_bd", f"{p}.d", f"{p}.x1", f"{p}.x2")
            residual = f"{p}.x2"
        self.rmsnorm(residual, "norm", "hn")
        hnq = self.quant("hn")
        self.linear(hnq, "lm_head", "lm_head_bias", "logits", "i16", N_pad=vocab_padded)

        return {
            "model": {"dim": dim, "n_layers": L, "n_heads": H, "n_kv_heads": Hkv, "head_dim": D, "ffn": ffn, "vocab": vocab,
                      "vocab_padded": vocab_padded, "max_seq": max_seq, "E_RES": E_RES, "E_LOGIT": E_LOGIT},
            "weights_file": "qweights.bin",
            "tensors": self.blob.tensors,
            "ops": self.ops,
            "input": "input", "output": "logits",
            "exps": self.exps, "scales": self.scales,
            "fusion": self.fusion,
        }


def quantize_export(export_dir: Path, out_dir: Path, fusion: bool = False) -> dict:
    """Quantize `export_dir` (model.mlir + weights.* + calib.json) into `out_dir/{qgraph.json, qweights.bin}`."""
    export_dir, out_dir = Path(export_dir), Path(out_dir)
    cfg, W, calib = load_export(export_dir)
    q = RefQuantizer(cfg, W, calib, fusion)
    qgraph = q.run()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qweights.bin").write_bytes(q.blob.bytes())
    with open(out_dir / "qgraph.json", "w") as f:
        json.dump(qgraph, f, indent=1)
    tok = export_dir / "tokenizer.json"
    if tok.exists():
        (out_dir / "tokenizer.json").write_bytes(tok.read_bytes())
    return qgraph


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="reference quantizer (Python twin of llaccel-quantize / llaccel-fuse)")
    ap.add_argument("export_dir")
    ap.add_argument("-o", "--out", default="build/q-ref/")
    ap.add_argument("--fusion", "--fuse", dest="fusion", action="store_true", help="emit v2 epilogue-fused linears")
    args = ap.parse_args(argv)
    g = quantize_export(Path(args.export_dir), Path(args.out), fusion=args.fusion)
    counts: dict[str, int] = {}
    for op in g["ops"]:
        counts[op["op"]] = counts.get(op["op"], 0) + 1
    print(f"wrote {args.out}: {len(g['ops'])} ops {counts}, {len(g['tensors'])} tensors, E_RES={g['model']['E_RES']} "
          f"E_LOGIT={g['model']['E_LOGIT']} fusion={args.fusion}")
    GoldenModel(args.out)  # load check


if __name__ == "__main__":
    main()
