"""Reference quantizer: export dir -> qgraph.json + qweights.bin (DIALECT.md section 2).

    uv run python -m llaccel.refquant build/export/ -o build/q/ [--fuse]

This is *test infrastructure*, not the compiler: it mirrors the integer-parameter
decisions the `llaccel-quantize` pass has to make (the "Compiler:" lines of
docs/NUMERICS.md) so that the golden model, verify tool and the end-to-end tests
can run before / independently of the C++ compiler. The C++ compiler's qgraph
is the one the runtime is checked against; this one exists so that the Python
side is self-checking and so the quantization recipe is written down once in
executable form. `--fuse` emits the v2 (epilogue-fused) op forms.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from .golden import GoldenModel, isqrt48


# --------------------------------------------------------------------------------------
# fixed-point helpers (the compiler's side of NUMERICS.md)
# --------------------------------------------------------------------------------------
def exp_for(absmax: float) -> int:
    """i16 exponent: e = ceil(log2(absmax / 32767))."""
    if absmax <= 0:
        return -15
    return int(math.ceil(math.log2(absmax / 32767.0)))


def mulshift_params(ratio: float, s_max: int = 63) -> tuple[int, int]:
    """(M, S) with M * 2^-S ~= ratio, M normalized to [2^30, 2^31) when possible."""
    if ratio <= 0:
        return 0, 0
    e = math.floor(math.log2(ratio))
    S = 30 - e
    if S > s_max:
        S = s_max
    if S < 0:
        S = 0
    M = int(round(ratio * 2.0**S))
    if M >= 2**31:
        M //= 2
        S -= 1
        if S < 0:
            raise ValueError(f"ratio {ratio} too large for mulshift")
    return M, S


def to_i16(x: np.ndarray, e: int) -> np.ndarray:
    return np.clip(np.round(x / 2.0**e), -32768, 32767).astype(np.int16)


def to_i8_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.abs(w).max(axis=1) / 127.0
    s_safe = np.where(s > 0, s, 1.0)
    q = np.clip(np.round(w / s_safe[:, None]), -128, 127).astype(np.int8)
    return q, s


# --------------------------------------------------------------------------------------
class Blob:
    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.off = 0
        self.tensors: dict[str, dict] = {}

    def add(self, name: str, arr: np.ndarray, dtype: str, exp: int | None = None) -> None:
        raw = {"i8": arr.astype("<i1"), "i16": arr.astype("<i2"), "i32": arr.astype("<i4"), "rq": arr.astype("<i4")}[dtype]
        b = raw.tobytes()
        spec = {"dtype": dtype, "shape": [int(s) for s in (arr.shape[:-1] if dtype == "rq" else arr.shape)], "offset": self.off}
        if exp is not None:
            spec["exp"] = exp
        self.tensors[name] = spec
        self.parts.append(b)
        self.off += len(b)
        pad = (-len(b)) % 64
        if pad:
            self.parts.append(b"\0" * pad)
            self.off += pad


def parse_model_attrs(mlir: str) -> dict:
    m = re.search(r"llaccel\.model\s*=\s*\{(.*?)\}\}", mlir, re.S)
    if not m:
        raise ValueError("model.mlir: llaccel.model attribute not found")
    out = {}
    for k, v, ty in re.findall(r"(\w+)\s*=\s*([-+0-9.eE]+)\s*:\s*(i64|f64)", m.group(1)):
        out[k] = int(v) if ty == "i64" else float(v)
    return out


def load_export(export_dir: Path) -> tuple[dict, dict[str, np.ndarray], dict[str, float]]:
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
def quantize_export(export_dir: Path, out_dir: Path, fuse: bool = False, rms_ratio: float = 64.0) -> dict:
    cfg, W, calib = load_export(export_dir)
    dim, L, H, Hkv, D, ffn, vocab, max_seq = (cfg[k] for k in ("dim", "n_layers", "n_heads", "n_kv_heads", "head_dim",
                                                               "ffn", "vocab", "max_seq"))
    eps, base = cfg["rms_eps"], cfg["rope_base"]
    vocab_padded = (vocab + 15) // 16 * 16
    if dim % 16 or ffn % 16 or (Hkv * D) % 16:
        raise ValueError("dim/ffn/kv width must be multiples of 16")

    exps: dict[str, int] = {}
    scales: dict[str, float] = {}
    res_names = ["input"] + [f"l{i}.{n}" for i in range(L) for n in ("x1", "x2", "o", "d")]
    E_RES = max(exp_for(calib[n]) for n in res_names)
    E_LOGIT = exp_for(calib["logits"])
    for n in res_names:
        exps[n] = E_RES
    exps["logits"] = E_LOGIT

    blob = Blob()
    ops: list[dict] = []

    # embedding, padded to vocab_padded rows
    emb = np.zeros((vocab_padded, dim))
    emb[:vocab] = W["embed"]
    blob.add("embed", to_i16(emb, E_RES), "i16", E_RES)
    # rope tables Q1.14
    inv_freq = base ** (-np.arange(0, D, 2, dtype=np.float64) / D)
    ang = np.outer(np.arange(max_seq, dtype=np.float64), inv_freq)
    blob.add("rope_cos", np.clip(np.round(np.cos(ang) * 16384), -32768, 32767).astype(np.int16), "i16")
    blob.add("rope_sin", np.clip(np.round(np.sin(ang) * 16384), -32768, 32767).astype(np.int16), "i16")

    def i16_act(name: str) -> int:
        if name not in exps:
            exps[name] = exp_for(calib[name])
        return exps[name]

    def i8_act(name: str, src: str | None = None) -> float:
        """scale of an i8 activation `name`, calibrated from tensor `src` (default: name without .q)."""
        if name not in scales:
            scales[name] = calib[src or name.removesuffix(".q")] / 127.0
        return scales[name]

    def rmsnorm_op(x: str, gamma: str, y: str) -> None:
        e_x, e_y = i16_act(x), i16_act(y)
        g = W[gamma]
        e_g = exp_for(np.abs(g).max())
        blob.add(gamma, to_i16(g, e_g), "i16", e_g)
        K = g.shape[0]
        eps_t = int(round(eps * K * 2.0 ** (-2 * e_x)))
        absmax_int = calib[x] / 2.0**e_x
        R_max = int(math.floor(math.log2((2**32 - 1) / math.sqrt(K))))
        R = min(R_max, int(math.floor(math.log2(65535.0 * absmax_int / rms_ratio))))
        R = max(R, e_g - e_y)  # sh_post >= 0
        C = int(round(2.0**R * math.sqrt(K)))
        sh_post = R - e_g + e_y
        if not 0 <= sh_post <= 63 or C >= 2**32:
            raise ValueError(f"rmsnorm {y}: sh_post={sh_post} C={C} out of range")
        ops.append({"op": "rmsnorm", "in": x, "gamma": gamma, "out": y, "K": K, "eps_t": eps_t, "C": C, "sh_post": sh_post})

    def quant_op(x: str) -> str:
        y = x + ".q"
        e_x, s_y = i16_act(x), i8_act(y, x)
        M, S = mulshift_params(2.0**e_x / s_y)
        ops.append({"op": "quant", "in": x, "out": y, "M": M, "S": S})
        return y

    def linear_op(a: str, w: str, b: str | None, y: str, out_dtype: str, N_pad: int | None = None,
                  epilogue: str = "none", aux: str | None = None, aux_shift: int = 0, silu: dict | None = None,
                  e_out: int | None = None) -> None:
        w_f = W[w]
        if N_pad is not None and N_pad != w_f.shape[0]:
            w_f = np.concatenate([w_f, np.zeros((N_pad - w_f.shape[0], w_f.shape[1]))], axis=0)
        N, K = w_f.shape
        q, s_w = to_i8_per_channel(w_f)
        blob.add(w, q, "i8")
        s_a = i8_act(a)
        if out_dtype == "i16":
            if e_out is not None:
                exps[y] = e_out
            e_y = i16_act(y)
            denom = 2.0**e_y
        else:
            denom = i8_act(y)
        rq = np.array([mulshift_params(s_a * s_w[n] / denom) for n in range(N)], dtype=np.int64)
        blob.add(w + ".rq", rq, "rq")
        bias = None
        if b is not None and b in W:
            bq = np.zeros(N)
            bq[: W[b].shape[0]] = W[b]
            s_safe = np.where(s_w > 0, s_w, 1.0)
            blob.add(b, np.round(bq / (s_a * s_safe)).astype(np.int32), "i32")
            bias = b
        ops.append({"op": "linear", "in": a, "w": w, "rq": w + ".rq", "bias": bias, "out": y, "N": N, "K": K,
                    "out_dtype": out_dtype, "epilogue": epilogue, "aux": aux, "aux_shift": aux_shift, "silu": silu})

    def silu_params(x: str, y: str) -> dict:
        e_x, e_y = i16_act(x), i16_act(y)
        Mi, Si = mulshift_params(2.0 ** (e_x + 12))
        sh_out = 16 + e_y - e_x
        if not 0 <= sh_out <= 63:
            raise ValueError(f"silu {y}: sh_out={sh_out}")
        return {"Mi": Mi, "Si": Si, "sh_out": sh_out}

    residual = "input"
    for i in range(L):
        p = f"l{i}"
        rmsnorm_op(residual, f"{p}_attn_norm", f"{p}.h")
        hq = quant_op(f"{p}.h")
        linear_op(hq, f"{p}_wq", f"{p}_bq", f"{p}.q", "i16")
        linear_op(hq, f"{p}_wk", f"{p}_bk", f"{p}.k", "i16")
        linear_op(hq, f"{p}_wv", f"{p}_bv", f"{p}.v", "i8")
        exps[f"{p}.qr"] = exps[f"{p}.q"]  # rope preserves the exponent
        exps[f"{p}.kr"] = exps[f"{p}.k"]
        ops.append({"op": "rope", "in": f"{p}.q", "out": f"{p}.qr", "H": H, "D": D})
        ops.append({"op": "rope", "in": f"{p}.k", "out": f"{p}.kr", "H": Hkv, "D": D})
        qq, kq = quant_op(f"{p}.qr"), quant_op(f"{p}.kr")
        ops.append({"op": "kv_write", "layer": i, "k": kq, "v": f"{p}.v", "Hkv": Hkv, "D": D})
        s_q, s_k, s_v, s_o = scales[qq], scales[kq], scales[f"{p}.v"], i8_act(f"{p}.a")
        Ms, Ss = mulshift_params(256.0 * s_q * s_k / math.sqrt(D))
        Mo, So = mulshift_params(s_v / (256.0 * s_o))
        ops.append({"op": "attention", "q": qq, "layer": i, "out": f"{p}.a", "H": H, "Hkv": Hkv, "D": D,
                    "Ms": Ms, "Ss": Ss, "Mo": Mo, "So": So})
        if fuse:
            linear_op(f"{p}.a", f"{p}_wo", f"{p}_bo", f"{p}.x1", "i16", epilogue="resadd", aux=residual, e_out=E_RES)
        else:
            linear_op(f"{p}.a", f"{p}_wo", f"{p}_bo", f"{p}.o", "i16", e_out=E_RES)
            ops.append({"op": "add", "a": residual, "b": f"{p}.o", "out": f"{p}.x1", "sh_b": 0})
        rmsnorm_op(f"{p}.x1", f"{p}_ffn_norm", f"{p}.h2")
        h2q = quant_op(f"{p}.h2")
        e_g, e_u, e_sg, e_f = (i16_act(f"{p}.{n}") for n in ("g", "u", "sg", "f"))
        # vmul is a *right* shift of a*b (exponent e_sg + e_u) to exponent e_f: sh = e_f - e_sg - e_u.
        # (NUMERICS.md's "Compiler:" line writes this with the opposite sign; numerics.h is authoritative.)
        sh_mul = e_f - e_sg - e_u
        if not 0 <= sh_mul <= 63:
            raise ValueError(f"mul {p}.f: sh={sh_mul}")
        if fuse:
            # fused form: the linear's rq targets the pre-epilogue exponent e(g) / e(u) while the op's
            # `out` tensor lives at e(sg) / e(f); `e_out` sets the former, the exps map keeps the latter.
            linear_op(h2q, f"{p}_wg", f"{p}_bg", f"{p}.sg", "i16", epilogue="silu", silu=silu_params(f"{p}.g", f"{p}.sg"),
                      e_out=e_g)
            exps[f"{p}.sg"] = e_sg
            linear_op(h2q, f"{p}_wu", f"{p}_bu", f"{p}.f", "i16", epilogue="mul", aux=f"{p}.sg", aux_shift=sh_mul, e_out=e_u)
            exps[f"{p}.f"] = e_f
        else:
            linear_op(h2q, f"{p}_wg", f"{p}_bg", f"{p}.g", "i16")
            linear_op(h2q, f"{p}_wu", f"{p}_bu", f"{p}.u", "i16")
            ops.append({"op": "silu", "in": f"{p}.g", "out": f"{p}.sg", **silu_params(f"{p}.g", f"{p}.sg")})
            ops.append({"op": "mul", "a": f"{p}.sg", "b": f"{p}.u", "out": f"{p}.f", "sh": sh_mul})
        fq = quant_op(f"{p}.f")
        if fuse:
            linear_op(fq, f"{p}_wd", f"{p}_bd", f"{p}.x2", "i16", epilogue="resadd", aux=f"{p}.x1", e_out=E_RES)
        else:
            linear_op(fq, f"{p}_wd", f"{p}_bd", f"{p}.d", "i16", e_out=E_RES)
            ops.append({"op": "add", "a": f"{p}.x1", "b": f"{p}.d", "out": f"{p}.x2", "sh_b": 0})
        residual = f"{p}.x2"
    rmsnorm_op(residual, "norm", "hn")
    hnq = quant_op("hn")
    linear_op(hnq, "lm_head", "lm_head_bias", "logits", "i16", N_pad=vocab_padded, e_out=E_LOGIT)

    qgraph = {
        "model": {"dim": dim, "n_layers": L, "n_heads": H, "n_kv_heads": Hkv, "head_dim": D, "ffn": ffn, "vocab": vocab,
                  "vocab_padded": vocab_padded, "max_seq": max_seq, "E_RES": E_RES, "E_LOGIT": E_LOGIT},
        "weights_file": "qweights.bin",
        "tensors": blob.tensors,
        "ops": ops,
        "input": "input", "output": "logits",
        "exps": exps, "scales": scales,
        "fusion": fuse,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qweights.bin").write_bytes(b"".join(blob.parts))
    with open(out_dir / "qgraph.json", "w") as f:
        json.dump(qgraph, f, indent=1)
    tok = export_dir / "tokenizer.json"
    if tok.exists():
        (out_dir / "tokenizer.json").write_bytes(tok.read_bytes())
    return qgraph


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("-o", "--out", default="build/q/")
    ap.add_argument("--fuse", action="store_true")
    args = ap.parse_args(argv)
    g = quantize_export(Path(args.export_dir), Path(args.out), fuse=args.fuse)
    print(f"wrote {args.out}: {len(g['ops'])} ops, {len(g['tensors'])} tensors, E_RES={g['model']['E_RES']} "
          f"E_LOGIT={g['model']['E_LOGIT']} fusion={args.fuse}")
    GoldenModel(args.out)  # load check


if __name__ == "__main__":
    main()
