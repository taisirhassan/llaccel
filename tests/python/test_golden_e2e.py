"""(d) end-to-end golden on a hand-built 1-layer dim-32 qgraph with random int weights (no compiler),
plus the full Python pipeline (export -> calibrate -> reference quantizer -> golden -> verify) on a tiny model."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from llaccel import golden as G
from llaccel.refquant import Blob, normalize_mulshift as mulshift_params

DIM, H, HKV, D, FFN, VOCAB, MAXSEQ = 32, 2, 1, 16, 64, 16, 64
OPS_PER_LAUNCH_V1 = 25  # op outputs (24; kv_write has none) + input
E_RES, E_H, E_QK, E_G, E_SG, E_F, E_LOGIT = -10, -12, -12, -12, -13, -14, -8
S_H, S_QK, S_V, S_A, S_H2, S_F, S_HN = 3 / 127, 2 / 127, 1.5 / 127, 1.2 / 127, 3 / 127, 1 / 127, 3 / 127


def build_fixture(out: Path, fuse: bool, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    blob = Blob()
    ops: list[dict] = []
    blob.add("embed", rng.integers(-6000, 6000, (VOCAB, DIM)), "i16", E_RES)
    inv_freq = 10000.0 ** (-np.arange(0, D, 2) / D)
    ang = np.outer(np.arange(MAXSEQ), inv_freq)
    blob.add("rope_cos", np.round(np.cos(ang) * 16384).astype(np.int16), "i16")
    blob.add("rope_sin", np.round(np.sin(ang) * 16384).astype(np.int16), "i16")

    def rms(x, gamma, y, e_x, e_y):
        e_g = -14
        blob.add(gamma, rng.integers(12000, 20000, DIM), "i16", e_g)  # gamma ~ 0.7..1.2
        R = 24
        ops.append({"op": "rmsnorm", "in": x, "gamma": gamma, "out": y, "K": DIM,
                    "eps_t": int(round(1e-5 * DIM * 2.0 ** (-2 * e_x))), "C": int(round(2.0**R * math.sqrt(DIM))),
                    "sh_post": R - e_g + e_y})

    def quant(x, e_x, s_y):
        M, S = mulshift_params(2.0**e_x / s_y)
        ops.append({"op": "quant", "in": x, "out": x + ".q", "M": M, "S": S})
        return x + ".q"

    def lin(a, s_a, w, y, N, K, out_dtype, denom, bias=False, **ep):
        blob.add(w, rng.integers(-127, 128, (N, K)), "i8")
        s_w = rng.uniform(0.01, 0.03, N)
        blob.add(w + ".rq", np.array([mulshift_params(s_a * s_w[n] / denom) for n in range(N)]), "rq")
        b = None
        if bias:
            blob.add(w + ".b", rng.integers(-2000, 2000, N), "i32")
            b = w + ".b"
        ops.append({"op": "linear", "in": a, "w": w, "rq": w + ".rq", "bias": b, "out": y, "N": N, "K": K,
                    "out_dtype": out_dtype, "epilogue": ep.get("epilogue", "none"), "aux": ep.get("aux"),
                    "aux_shift": ep.get("aux_shift", 0), "silu": ep.get("silu")})

    rms("input", "l0_attn_norm", "l0.h", E_RES, E_H)
    hq = quant("l0.h", E_H, S_H)
    lin(hq, S_H, "l0_wq", "l0.q", DIM, DIM, "i16", 2.0**E_QK, bias=True)
    lin(hq, S_H, "l0_wk", "l0.k", HKV * D, DIM, "i16", 2.0**E_QK, bias=True)
    lin(hq, S_H, "l0_wv", "l0.v", HKV * D, DIM, "i8", S_V, bias=True)
    ops.append({"op": "rope", "in": "l0.q", "out": "l0.qr", "H": H, "D": D})
    ops.append({"op": "rope", "in": "l0.k", "out": "l0.kr", "H": HKV, "D": D})
    qq, kq = quant("l0.qr", E_QK, S_QK), quant("l0.kr", E_QK, S_QK)
    ops.append({"op": "kv_write", "layer": 0, "k": kq, "v": "l0.v", "Hkv": HKV, "D": D})
    Ms, Ss = mulshift_params(256.0 * S_QK * S_QK / math.sqrt(D))
    Mo, So = mulshift_params(S_V / (256.0 * S_A))
    ops.append({"op": "attention", "q": qq, "layer": 0, "out": "l0.a", "H": H, "Hkv": HKV, "D": D,
                "Ms": Ms, "Ss": Ss, "Mo": Mo, "So": So})
    if fuse:
        lin("l0.a", S_A, "l0_wo", "l0.x1", DIM, DIM, "i16", 2.0**E_RES, epilogue="resadd", aux="input")
    else:
        lin("l0.a", S_A, "l0_wo", "l0.o", DIM, DIM, "i16", 2.0**E_RES)
        ops.append({"op": "add", "a": "input", "b": "l0.o", "out": "l0.x1", "sh_b": 0})
    rms("l0.x1", "l0_ffn_norm", "l0.h2", E_RES, E_H)
    h2q = quant("l0.h2", E_H, S_H2)
    Mi, Si = mulshift_params(2.0 ** (E_G + 12))
    silu = {"Mi": Mi, "Si": Si, "sh_out": 16 + E_SG - E_G}
    sh_mul = E_F - E_SG - E_G  # vmul right-shifts a*b (exponent E_SG + E_G) to E_F
    if fuse:
        lin(h2q, S_H2, "l0_wg", "l0.sg", FFN, DIM, "i16", 2.0**E_G, epilogue="silu", silu=silu)
        lin(h2q, S_H2, "l0_wu", "l0.f", FFN, DIM, "i16", 2.0**E_G, epilogue="mul", aux="l0.sg", aux_shift=sh_mul)
    else:
        lin(h2q, S_H2, "l0_wg", "l0.g", FFN, DIM, "i16", 2.0**E_G)
        lin(h2q, S_H2, "l0_wu", "l0.u", FFN, DIM, "i16", 2.0**E_G)
        ops.append({"op": "silu", "in": "l0.g", "out": "l0.sg", **silu})
        ops.append({"op": "mul", "a": "l0.sg", "b": "l0.u", "out": "l0.f", "sh": sh_mul})
    fq = quant("l0.f", E_F, S_F)
    if fuse:
        lin(fq, S_F, "l0_wd", "l0.x2", DIM, FFN, "i16", 2.0**E_RES, epilogue="resadd", aux="l0.x1")
    else:
        lin(fq, S_F, "l0_wd", "l0.d", DIM, FFN, "i16", 2.0**E_RES)
        ops.append({"op": "add", "a": "l0.x1", "b": "l0.d", "out": "l0.x2", "sh_b": 0})
    rms("l0.x2", "norm", "hn", E_RES, E_H)
    hnq = quant("hn", E_H, S_HN)
    lin(hnq, S_HN, "lm_head", "logits", VOCAB, DIM, "i16", 2.0**E_LOGIT)
    qgraph = {"model": {"dim": DIM, "n_layers": 1, "n_heads": H, "n_kv_heads": HKV, "head_dim": D, "ffn": FFN,
                        "vocab": VOCAB, "vocab_padded": VOCAB, "max_seq": MAXSEQ, "E_RES": E_RES, "E_LOGIT": E_LOGIT},
              "weights_file": "qweights.bin", "tensors": blob.tensors, "ops": ops, "input": "input", "output": "logits"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "qweights.bin").write_bytes(b"".join(blob.parts))
    (out / "qgraph.json").write_text(json.dumps(qgraph))
    (out / "tokenizer.json").write_text(json.dumps({"itos": [chr(97 + i) for i in range(VOCAB)]}))
    return out


@pytest.fixture(scope="module")
def fixture_dirs(tmp_path_factory):
    base = tmp_path_factory.mktemp("qfix")
    return build_fixture(base / "v1", fuse=False), build_fixture(base / "v2", fuse=True)


def test_prefill_decode_runs_and_is_deterministic(fixture_dirs):
    v1, _ = fixture_dirs
    rng = np.random.default_rng(0)
    prompt = rng.integers(0, VOCAB, 20).tolist()  # 2 prefill chunks (16 + 4 valid rows)
    recs = []
    for _ in range(2):
        g = G.GoldenModel(v1)
        recs.append(g.generate(prompt, 12))
    assert recs[0] == recs[1]
    r = recs[0]
    # runtime launch sequence: 2 prefill chunks + one decode launch per generated token
    assert len(r["generated"]) == 12 and len(r["steps"]) == 2 + 12 and len(r["logits_last_rows"]) == 14
    assert len(r["argmax_per_step"]) == 14 and r["generated"] == r["argmax_per_step"][1:13]
    assert r["steps"][0] == {"kind": "prefill", "pos": 0, "rows": 16, "valid_rows": 16}
    assert r["steps"][1] == {"kind": "prefill", "pos": 16, "rows": 16, "valid_rows": 4}
    assert r["steps"][2] == {"kind": "decode", "pos": 20, "rows": 1, "valid_rows": 1}
    assert r["steps"][-1] == {"kind": "decode", "pos": 31, "rows": 1, "valid_rows": 1}
    assert all(0 <= t < VOCAB for t in r["generated"])
    assert all(len(row) == VOCAB for row in r["logits_last_rows"])  # `vocab` entries, not vocab_padded
    assert all(-32768 <= v <= 32767 for row in r["logits_last_rows"] for v in row)
    # the recorded argmax is the lowest index among ties over the first `vocab` logits
    for row, am in zip(r["logits_last_rows"], r["argmax_per_step"]):
        assert row[am] == max(row) and am == row.index(max(row))
    # activations are not degenerate
    acts = []
    G.GoldenModel(v1).prefill(prompt, acts)
    a = acts[0]["acts"]
    assert set(a) >= {"input", "l0.h", "l0.h.q", "l0.q", "l0.qr", "l0.kr.q", "l0.a", "l0.x1", "l0.sg", "l0.f", "hn", "logits"}
    for name in ("l0.h", "l0.a", "l0.x2", "logits"):
        assert np.abs(a[name][:16]).max() > 0, name


def test_chunked_prefill_equals_sequential_decode(fixture_dirs):
    """Real rows of a padded M=16 chunk must be bit-identical to decoding those tokens one at a time."""
    v1, _ = fixture_dirs
    rng = np.random.default_rng(3)
    toks = rng.integers(0, VOCAB, 21).tolist()
    g = G.GoldenModel(v1)
    rows_prefill = g.prefill(toks)
    g2 = G.GoldenModel(v1)
    rows_decode = [g2.decode(t, i) for i, t in enumerate(toks)]
    assert len(rows_prefill) == len(rows_decode) == 21
    for a, b in zip(rows_prefill, rows_decode):
        assert np.array_equal(a, b)


def test_fused_v2_matches_unfused_v1(fixture_dirs):
    """RESADD/SILU/MUL epilogues are defined to be bit-identical to the separate vec ops."""
    v1, v2 = fixture_dirs
    prompt = list(range(10))
    assert G.GoldenModel(v1).generate(prompt, 16) == G.GoldenModel(v2).generate(prompt, 16)


def test_golden_cli_and_dump_acts(fixture_dirs, tmp_path):
    v1, _ = fixture_dirs
    out = tmp_path / "golden.json"
    acts = tmp_path / "acts"
    G.main(["--qgraph", str(v1), "--prompt", "abcde", "--tokens", "5", "-o", str(out), "--dump-acts", str(acts)])
    rec = json.loads(out.read_text())
    assert rec["prompt_tokens"] == [0, 1, 2, 3, 4] and len(rec["generated"]) == 5 and len(rec["argmax_per_step"]) == 6
    assert rec["argmax_per_step"][:5] == rec["generated"]  # last decode launch's argmax is not a generated token
    files = sorted(acts.glob("*.npy"))
    assert len(files) == 6 * OPS_PER_LAUNCH_V1
    assert np.load(acts / "step000_prefill_pos0_logits.npy").shape == (16, VOCAB)
    assert np.load(acts / "step001_decode_pos5_l0.a.npy").shape == (1, H * D)
    assert np.load(acts / "step005_decode_pos9_l0.kr.q.npy").shape == (1, HKV * D)
    # the golden.json logits row is the relevant row of the dumped logits activation
    assert np.load(acts / "step000_prefill_pos0_logits.npy")[4, :VOCAB].tolist() == rec["logits_last_rows"][0]
    assert np.load(acts / "step003_decode_pos7_logits.npy")[0, :VOCAB].tolist() == rec["logits_last_rows"][3]
    from llaccel.verify import compare_sim
    ok, msg = compare_sim(out, out)
    assert ok and msg.startswith("MATCH")
    bad = dict(rec)
    bad["generated"] = list(rec["generated"])
    bad["generated"][2] = (bad["generated"][2] + 1) % VOCAB
    (tmp_path / "sim.json").write_text(json.dumps(bad))
    ok, msg = compare_sim(tmp_path / "sim.json", out)
    assert not ok and "step 2" in msg


def test_position_overflow_is_an_error(fixture_dirs):
    v1, _ = fixture_dirs
    with pytest.raises(ValueError, match="max_seq"):
        G.GoldenModel(v1).generate(list(range(10)), MAXSEQ)


def test_last_position_skips_the_decode_launch_like_the_runtime(fixture_dirs):
    """host.cpp stops launching when L + i + 1 >= max_seq; the token is still emitted."""
    v1, _ = fixture_dirs
    r = G.GoldenModel(v1).generate(list(range(10)), MAXSEQ - 10)
    assert len(r["generated"]) == MAXSEQ - 10
    assert len(r["steps"]) == 1 + (MAXSEQ - 10 - 1) and r["steps"][-1]["pos"] == MAXSEQ - 2
    assert len(r["argmax_per_step"]) == len(r["logits_last_rows"]) == len(r["steps"])


def test_prefill_pad_rows_do_not_affect_valid_rows(fixture_dirs):
    """Causality: rows 0..11 of a chunk with 12 valid + 4 zero pad rows == rows 0..11 of the full 16-row chunk."""
    v1, _ = fixture_dirs
    toks = [(3 + i) % VOCAB for i in range(20)]
    a = G.GoldenModel(v1).prefill(toks)
    b = G.GoldenModel(v1).prefill(toks[:12])
    assert len(b) == 12
    for x, y in zip(a[:12], b):
        assert np.array_equal(x, y)


def test_full_python_pipeline(tiny_export):
    """export -> calibrate (synthetic corpus) -> reference quantizer (v1 and v2) -> golden -> verify."""
    from llaccel.data import CharTokenizer
    from llaccel.refquant import quantize_export
    from llaccel.verify import compare_models

    ex, model, cfg, itos = tiny_export
    assert {p.name for p in ex.iterdir()} == {"model.mlir", "weights.bin", "weights.json", "calib.json", "tokenizer.json"}
    calib = json.loads((ex / "calib.json").read_text())
    assert "input" in calib and "l1.x2" in calib and "logits" in calib and len(calib) == 2 + 16 * 2 + 1
    q1 = quantize_export(ex, ex.parent / "q1", fusion=False)
    q2 = quantize_export(ex, ex.parent / "q2", fusion=True)
    assert len(q1["ops"]) == 2 * 22 + 3 and len(q2["ops"]) == 2 * 18 + 3  # v2 drops add, silu, mul, add per layer
    g1, g2 = G.GoldenModel(ex.parent / "q1"), G.GoldenModel(ex.parent / "q2")
    assert g1.generate([0, 1, 2], 20) == g2.generate([0, 1, 2], 20)
    r = compare_models(model, CharTokenizer(itos), ex.parent / "q1", "abc", 10)
    assert r["n_steps"] == 11 and 0.0 <= r["top1_agreement"] <= 1.0 and -1.0 <= r["mean_cosine"] <= 1.0
    assert len(r["golden_text"]) == 10 and len(r["fp32_text"]) == 10


