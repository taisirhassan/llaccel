"""numpy golden model: executes a quantized graph (docs/DIALECT.md section 2) with the
exact integer semantics of docs/NUMERICS.md / include/llaccel/numerics.h.

    uv run python -m llaccel.golden --qgraph build/q/ --prompt "ROMEO:" --tokens 64 -o build/golden.json
    uv run python -m llaccel.golden --qgraph build/q/ --prompt "ROMEO:" --tokens 64 --dump-acts build/acts/

Every primitive below is a line-by-line twin of the C++ function of the same
name in numerics.h. All arithmetic is int64 numpy (or Python int); there is no
floating point on the data path. The device's execution model is mirrored too:
prefill runs in configurable chunks (default M=16) of rows (zero-padded, positions POS+m, pad rows'
KV entries written and later overwritten) and decode runs M=1, so the sequence
of ops and positions is identical to what the ISA programs perform.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .luts import exp_frac_lut, exp_int_lut, sigmoid_lut

I64 = np.int64
SIG = sigmoid_lut().astype(I64)
EXPI = exp_int_lut().astype(I64)
EXPF = exp_frac_lut().astype(I64)
PREFILL_M = 16


# --------------------------------------------------------------------------------------
# Primitives (numerics.h)
# --------------------------------------------------------------------------------------
def rshr(v, s):
    """round-half-up arithmetic right shift; `s` may be a scalar or a broadcastable int64 array."""
    v = np.asarray(v, dtype=I64)
    s = np.asarray(s, dtype=I64)
    if s.ndim == 0:
        if int(s) == 0:
            return v.copy()
        return (v >> s) + ((v >> (s - I64(1))) & I64(1))
    s_safe = np.where(s == 0, I64(1), s)
    rounded = (v >> s_safe) + ((v >> (s_safe - I64(1))) & I64(1))
    return np.where(s == 0, v, rounded)


def sat8(v):
    return np.clip(np.asarray(v, dtype=I64), -128, 127)


def satu8(v):
    return np.clip(np.asarray(v, dtype=I64), 0, 255)


def sat16(v):
    return np.clip(np.asarray(v, dtype=I64), -32768, 32767)


def satu16(v):
    return np.clip(np.asarray(v, dtype=I64), 0, 65535)


def mulshift(v, M, S):
    return rshr(np.asarray(v, dtype=I64) * np.asarray(M, dtype=I64), S)


def isqrt48(t: int) -> int:
    """floor(sqrt(t)) for t < 2^48, bit-serial exactly as the RTL / numerics.h."""
    t = int(t)
    if not 0 <= t < (1 << 48):
        raise ValueError(f"isqrt48 input out of range: {t}")
    rem = root = 0
    for i in range(23, -1, -1):
        rem = (rem << 2) | ((t >> (2 * i)) & 3)
        root <<= 1
        trial = (root << 1) | 1
        if trial <= rem:
            rem -= trial
            root |= 1
    return root


def udiv(a: int, b: int) -> int:
    a, b = int(a), int(b)
    if a < 0 or b < 0:
        raise ValueError("udiv operands must be unsigned")
    return (a // (b if b else 1)) & 0xFFFFFFFF


# ---- GEMM epilogue ------------------------------------------------------------------
def requant16(acc, M, S):
    return sat16(mulshift(acc, M, S))


def requant8(acc, M, S):
    return sat8(mulshift(acc, M, S))


# ---- SiLU --------------------------------------------------------------------------------
def silu16(x, Mi: int, Si: int, sh_out: int):
    x = np.asarray(x, dtype=I64)
    u = sat16(mulshift(x, Mi, Si))
    idx = (u >> 8) + 128  # arithmetic shift: idx in [0, 255]
    f = u & 255
    sg = SIG[idx] + (((SIG[idx + 1] - SIG[idx]) * f) >> 8)
    return sat16(rshr(x * sg, sh_out))


# ---- elementwise ---------------------------------------------------------------------------
def vmul(a, b, sh: int):
    return sat16(rshr(np.asarray(a, dtype=I64) * np.asarray(b, dtype=I64), sh))


def vadd(a, b, sh_b: int):
    return sat16(np.asarray(a, dtype=I64) + rshr(b, sh_b))


def vquant(x, M: int, S: int):
    return sat8(mulshift(x, M, S))


# ---- RMSNorm -------------------------------------------------------------------------------
def rmsnorm_row(x, g, eps_t: int, C: int, sh_post: int):
    x = np.asarray(x, dtype=I64)
    g = np.asarray(g, dtype=I64)
    ss = int(np.sum(x * x))
    tt = ss + int(eps_t)
    r = isqrt48(tt)
    inv = min(65535, udiv(C, max(r, 1)))
    xg = x * g
    return sat16(rshr(xg * I64(inv), sh_post))


def rmsnorm(x, g, eps_t: int, C: int, sh_post: int):
    x = np.asarray(x, dtype=I64)
    return np.stack([rmsnorm_row(row, g, eps_t, C, sh_post) for row in x])


# ---- RoPE (rotate_half) --------------------------------------------------------------------
def rope_row(x, H: int, D: int, cosv, sinv):
    x = np.asarray(x, dtype=I64)
    c = np.asarray(cosv, dtype=I64)
    s = np.asarray(sinv, dtype=I64)
    y = np.empty_like(x)
    half = D // 2
    for h in range(H):
        x1 = x[h * D : h * D + half]
        x2 = x[h * D + half : (h + 1) * D]
        y[h * D : h * D + half] = sat16(rshr(x1 * c - x2 * s, 14))
        y[h * D + half : (h + 1) * D] = sat16(rshr(x2 * c + x1 * s, 14))
    return y


def rope(x, H: int, D: int, cos_tab, sin_tab, pos: int):
    """x [M][H*D]; row m uses table row pos+m."""
    x = np.asarray(x, dtype=I64)
    return np.stack([rope_row(x[m], H, D, cos_tab[pos + m], sin_tab[pos + m]) for m in range(x.shape[0])])


# ---- Attention (one query row, one head) ---------------------------------------------------
def attention_head(q, keys, vals, Ms: int, Ss: int, Mo: int, So: int, prob_bits: int = 8):
    """q [D] i8; keys/vals [T][D] i8 -> out [D] i8 (numerics.h attention_head)."""
    q = np.asarray(q, dtype=I64)
    keys = np.asarray(keys, dtype=I64)
    vals = np.asarray(vals, dtype=I64)
    scores = keys @ q  # i32 per key
    mx = int(scores.max())
    z = mulshift(I64(mx) - scores, Ms, Ss)  # >= 0
    zi = np.minimum(z, 4095)
    p = np.where(z < 4096, (EXPI[zi >> 8] * EXPF[zi & 255] + (1 << 15)) >> 16, I64(0))
    total = int(p.sum())
    inv = udiv(1 << 31, total)
    if prob_bits not in (8, 15):
        raise ValueError("attention prob_bits must be 8 or 15")
    pn = np.minimum(rshr(p * I64(inv), 31 - prob_bits), (1 << prob_bits) - 1)
    o = pn @ vals  # i32 per d
    if prob_bits == 15:
        o = rshr(o, 7)  # preserve the ISA's existing Q0.8 output multiplier
    return sat8(mulshift(o, Mo, So))


# --------------------------------------------------------------------------------------
# Quantized graph loading
# --------------------------------------------------------------------------------------
_DT = {"i8": (np.int8, 1), "i16": (np.dtype("<i2"), 2), "i32": (np.dtype("<i4"), 4), "rq": (np.dtype("<i4"), 8)}


class QGraph:
    def __init__(self, qdir: Path | str) -> None:
        self.dir = Path(qdir)
        with open(self.dir / "qgraph.json") as f:
            self.g = json.load(f)
        self.model = self.g["model"]
        self.ops = self.g["ops"]
        self.blob = np.memmap(self.dir / self.g["weights_file"], dtype=np.uint8, mode="r")
        self.tensors: dict[str, np.ndarray] = {}
        self.tensor_exp: dict[str, int] = {}
        for name, spec in self.g["tensors"].items():
            dt, size = _DT[spec["dtype"]]
            count = int(np.prod(spec["shape"]))
            if spec["dtype"] == "rq":
                arr = np.frombuffer(self.blob, dtype=dt, count=2 * count, offset=spec["offset"])
                arr = arr.reshape(count, 2)
            else:
                arr = np.frombuffer(self.blob, dtype=dt, count=count, offset=spec["offset"])
                arr = arr.reshape(spec["shape"])
            self.tensors[name] = arr
            if "exp" in spec:
                self.tensor_exp[name] = spec["exp"]
        self.exps: dict[str, int] = dict(self.g.get("exps", {}))
        self.scales: dict[str, float] = dict(self.g.get("scales", {}))
        self.input_name = self.g.get("input", "input")
        self.output_name = self.g.get("output", "logits")


def exact_blas_matmul(a: np.ndarray, w: np.ndarray, chunk_rows: int = 1024) -> np.ndarray:
    """Compute integer A @ W.T with exact FP64 BLAS accumulation.

    The signed dot-product absolute bound must fit 53 integer bits; therefore
    every product and every possible partial sum is exactly representable.
    Chunked weight conversion bounds temporary storage independently of vocab.
    """
    if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[1]:
        raise ValueError("invalid matrix dimensions")
    if a.dtype.kind not in "iu" or w.dtype.kind not in "iu" or chunk_rows < 1:
        raise ValueError("exact BLAS requires integer inputs and positive chunks")
    # Actual activations are small, but use the weight dtype bound to avoid
    # scanning the entire checkpoint on each token batch.
    amax = max(abs(int(a.min(initial=0))), abs(int(a.max(initial=0))))
    wi = np.iinfo(w.dtype)
    wmax = max(abs(int(wi.min)), abs(int(wi.max)))
    if a.shape[1] * amax * wmax > (1 << 53):
        raise ValueError("integer dot product exceeds exact FP64 accumulation bound")
    af = a.astype(np.float64)
    out = np.empty((a.shape[0], w.shape[0]), dtype=I64)
    for start in range(0, w.shape[0], chunk_rows):
        stop = min(start + chunk_rows, w.shape[0])
        out[:, start:stop] = (af @ w[start:stop].astype(np.float64).T).astype(I64)
    return out


# --------------------------------------------------------------------------------------
# Golden model
# --------------------------------------------------------------------------------------
class GoldenModel:
    def __init__(self, qgraph_dir: Path | str, *, fast_matmul: bool = False) -> None:
        self.fast_matmul = fast_matmul
        self.q = QGraph(qgraph_dir)
        m = self.q.model
        self.dim = int(m["dim"])
        self.n_layers = int(m["n_layers"])
        self.vocab = int(m["vocab"])
        self.vocab_padded = int(m.get("vocab_padded", self.vocab))
        self.max_seq = int(m["max_seq"])
        self.prefill_m = int(m.get("prefill_m", PREFILL_M))
        if not 1 <= self.prefill_m <= 16 or self.max_seq % self.prefill_m:
            raise ValueError("max_seq must be divisible by prefill_m in [1,16]")
        self.E_RES = int(m["E_RES"])
        self.E_LOGIT = int(m["E_LOGIT"])
        self.embed = self.q.tensors["embed"]
        self.rope_cos = self.q.tensors["rope_cos"]
        self.rope_sin = self.q.tensors["rope_sin"]
        self.reset()

    def reset(self) -> None:
        """Clear persistent KV state between independent sequences."""
        self.kv: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for op in self.q.ops:
            if op["op"] == "kv_write":
                L, Hkv, D = int(op["layer"]), int(op["Hkv"]), int(op["D"])
                self.kv[L] = (np.zeros((Hkv, self.max_seq, D), np.int8), np.zeros((Hkv, self.max_seq, D), np.int8))

    # ---- one device launch --------------------------------------------------------------
    def run_chunk(self, rows: np.ndarray, pos: int, acts: dict | None = None) -> np.ndarray:
        """Executes the op list for `rows` [M][dim] (i16 at E_RES) at positions pos..pos+M-1.
        Returns logits [M][vocab_padded] (i16 at E_LOGIT). `acts` collects every named activation."""
        rows = np.asarray(rows, dtype=I64)
        M = rows.shape[0]
        if pos < 0 or pos + M > self.max_seq:
            raise ValueError(f"positions {pos}..{pos + M - 1} exceed max_seq={self.max_seq}")
        act: dict[str, np.ndarray] = {self.q.input_name: rows}
        tens = self.q.tensors
        for op in self.q.ops:
            k = op["op"]
            if k == "rmsnorm":
                x = act[op["in"]]
                gamma = tens[op["gamma"]]
                heads = int(op.get("heads", 1))
                if heads < 1 or x.shape[1] != heads * len(gamma):
                    raise ValueError("RMSNorm head geometry mismatch")
                act[op["out"]] = rmsnorm(x.reshape(-1, len(gamma)), gamma,
                    op["eps_t"], op["C"], op["sh_post"]).reshape(x.shape)
            elif k == "quant":
                act[op["out"]] = vquant(act[op["in"]], op["M"], op["S"])
            elif k == "linear":
                act[op["out"]] = self.linear(op, act)
            elif k == "rope":
                act[op["out"]] = rope(act[op["in"]], op["H"], op["D"], self.rope_cos, self.rope_sin, pos)
            elif k == "kv_write":
                self.kv_write(op, act, pos)
            elif k == "attention":
                act[op["out"]] = self.attention(op, act, pos)
            elif k == "add":
                act[op["out"]] = vadd(act[op["a"]], act[op["b"]], op["sh_b"])
            elif k == "mul":
                act[op["out"]] = vmul(act[op["a"]], act[op["b"]], op["sh"])
            elif k == "silu":
                act[op["out"]] = silu16(act[op["in"]], op["Mi"], op["Si"], op["sh_out"])
            else:
                raise ValueError(f"unknown qgraph op {k!r}")
        if acts is not None:
            acts.update(act)
        return act[self.q.output_name]

    def linear(self, op: dict, act: dict) -> np.ndarray:
        a = act[op["in"]]
        w = self.q.tensors[op["w"]]
        rq = self.q.tensors[op["rq"]]
        N, K = int(op["N"]), int(op["K"])
        if w.shape != (N, K) or a.shape[1] != K:
            raise ValueError(f"linear {op['out']}: shapes A{a.shape} W{w.shape} vs N={N} K={K}")
        acc = exact_blas_matmul(a, w) if self.fast_matmul else a @ w.T
        if op.get("bias"):
            acc = acc + self.q.tensors[op["bias"]][None, :]
        Mn, Sn = rq[:, 0][None, :], rq[:, 1][None, :]
        if op.get("out_dtype", "i16") == "i8":
            if op.get("epilogue", "none") != "none":
                raise ValueError("i8 output is only valid with epilogue none")
            return requant8(acc, Mn, Sn)
        t = requant16(acc, Mn, Sn)
        ep = op.get("epilogue", "none")
        if ep == "none":
            return t
        if ep == "resadd":
            return sat16(t + act[op["aux"]])
        if ep == "silu":
            s = op["silu"]
            return silu16(t, s["Mi"], s["Si"], s["sh_out"])
        if ep == "mul":
            return sat16(rshr(t * act[op["aux"]], op["aux_shift"]))
        raise ValueError(f"unknown epilogue {ep!r}")

    def kv_write(self, op: dict, act: dict, pos: int) -> None:
        kc, vc = self.kv[int(op["layer"])]
        Hkv, D = int(op["Hkv"]), int(op["D"])
        k, v = act[op["k"]], act[op["v"]]
        for m in range(k.shape[0]):
            for h in range(Hkv):
                kc[h, pos + m] = k[m, h * D : (h + 1) * D]
                vc[h, pos + m] = v[m, h * D : (h + 1) * D]

    def attention(self, op: dict, act: dict, pos: int) -> np.ndarray:
        q = act[op["q"]]
        kc, vc = self.kv[int(op["layer"])]
        H, Hkv, D = int(op["H"]), int(op["Hkv"]), int(op["D"])
        rep = H // Hkv
        M = q.shape[0]
        out = np.zeros((M, H * D), I64)
        for m in range(M):
            T = pos + m + 1
            for h in range(H):
                kvh = h // rep
                out[m, h * D : (h + 1) * D] = attention_head(q[m, h * D : (h + 1) * D], kc[kvh, :T], vc[kvh, :T],
                                                             op["Ms"], op["Ss"], op["Mo"], op["So"], op.get("prob_bits", 8))
        return out

    # ---- host side --------------------------------------------------------------------------
    def embed_rows(self, tokens, M: int) -> np.ndarray:
        rows = np.zeros((M, self.dim), I64)
        for i, t in enumerate(tokens):
            rows[i] = self.embed[int(t)]
        return rows

    @staticmethod
    def argmax(logits_row: np.ndarray, vocab: int) -> int:
        return int(np.argmax(logits_row[:vocab]))  # lowest index wins ties

    def prefill(self, tokens, acts: list | None = None) -> list[np.ndarray]:
        """Chunked prefill using model.prefill_m (default 16). Returns the logits row for every prompt token."""
        tokens = [int(t) for t in tokens]
        rows_out = []
        for start in range(0, len(tokens), self.prefill_m):
            chunk = tokens[start : start + self.prefill_m]
            a = {} if acts is not None else None
            logits = self.run_chunk(self.embed_rows(chunk, self.prefill_m), start, a)
            if acts is not None:
                acts.append({"kind": "prefill", "pos": start, "rows": self.prefill_m, "valid_rows": len(chunk), "acts": a})
            rows_out.extend(logits[: len(chunk)])
        return rows_out

    def decode(self, token: int, pos: int, acts: list | None = None) -> np.ndarray:
        a = {} if acts is not None else None
        logits = self.run_chunk(self.embed_rows([token], 1), pos, a)
        if acts is not None:
            acts.append({"kind": "decode", "pos": pos, "rows": 1, "valid_rows": 1, "acts": a})
        return logits[0]

    def generate(self, prompt_tokens, n_tokens: int, acts: list | None = None) -> dict:
        """Greedy generation with exactly the launch sequence of the C++ runtime (runtime/src/host.cpp
        `Host::generate`): one launch per configured prefill chunk, then one M=1 decode launch per generated
        token (the last decode launch's argmax is recorded but not appended to `generated`, like the
        runtime). A decode launch is skipped when the next position would not fit in `max_seq`.

        Returns the golden.json record: `argmax_per_step` / `logits_last_rows` have one entry per
        launch; a logits row is the `vocab` (not `vocab_padded`) i16 logits of the launch's relevant row
        (row (L-1)%16 of the last prefill chunk, row 15 of earlier chunks, row 0 of a decode)."""
        if not isinstance(n_tokens, int) or n_tokens < 0:
            raise ValueError("n_tokens must be a nonnegative integer")
        prompt_tokens = [int(t) for t in prompt_tokens]
        if not prompt_tokens:
            raise ValueError("empty prompt")
        if len(prompt_tokens) + n_tokens > self.max_seq:
            raise ValueError(f"prompt + tokens = {len(prompt_tokens) + n_tokens} exceeds max_seq={self.max_seq}")
        self.reset()
        steps, argmax_per_step, logits_rows = [], [], []

        def record(kind: str, pos: int, valid: int, row: np.ndarray) -> int:
            steps.append({"kind": kind, "pos": pos, "rows": self.prefill_m if kind == "prefill" else 1, "valid_rows": valid})
            argmax_per_step.append(self.argmax(row, self.vocab))
            logits_rows.append([int(v) for v in row[: self.vocab]])
            return argmax_per_step[-1]

        rows = self.prefill(prompt_tokens, acts)
        L = len(prompt_tokens)
        n_chunks = math.ceil(L / self.prefill_m)
        nxt = -1
        for c in range(n_chunks):
            last = min(L, (c + 1) * self.prefill_m) - 1
            nxt = record("prefill", c * self.prefill_m, last + 1 - c * self.prefill_m, rows[last])
        generated: list[int] = []
        for i in range(n_tokens):
            generated.append(nxt)
            if L + i + 1 >= self.max_seq:
                break  # no room for another position (runtime stops launching too)
            nxt = record("decode", L + i, 1, self.decode(nxt, L + i, acts))
        return {"prompt_tokens": prompt_tokens, "generated": generated, "argmax_per_step": argmax_per_step,
                "logits_last_rows": logits_rows, "steps": steps,
                "model": {"vocab": self.vocab, "vocab_padded": self.vocab_padded, "E_LOGIT": self.E_LOGIT,
                          "E_RES": self.E_RES, "max_seq": self.max_seq}}


def dump_acts(acts: list, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, step in enumerate(acts):
        for name, arr in step["acts"].items():
            np.save(out_dir / f"step{i:03d}_{step['kind']}_pos{step['pos']}_{name}.npy", arr.astype(np.int32))
            n += 1
    return n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="numpy golden model")
    ap.add_argument("--qgraph", required=True)
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json (default: <qgraph>/tokenizer.json)")
    ap.add_argument("-o", "--out", default="build/golden.json")
    ap.add_argument("--dump-acts", default=None)
    args = ap.parse_args(argv)
    from .data import CharTokenizer
    tok_path = Path(args.tokenizer) if args.tokenizer else Path(args.qgraph) / "tokenizer.json"
    tok = CharTokenizer.from_json(tok_path)
    g = GoldenModel(args.qgraph)
    acts = [] if args.dump_acts else None
    rec = g.generate(tok.encode(args.prompt), args.tokens, acts)
    rec["prompt"] = args.prompt
    rec["text"] = tok.decode(rec["generated"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f)
    print(f"wrote {args.out}: {len(rec['steps'])} launches, {len(rec['generated'])} tokens")
    print(args.prompt + rec["text"])
    if acts is not None:
        n = dump_acts(acts, Path(args.dump_acts))
        print(f"dumped {n} activation arrays to {args.dump_acts}")


if __name__ == "__main__":
    main()
