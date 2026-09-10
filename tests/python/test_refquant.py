"""(e) reference quantizer: QuantParams.h twins (rounding, normalisation, R rule), parameter/float
consistency through the golden primitives, and the qgraph.json contract on a real tiny export."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from llaccel import golden as G
from llaccel import refquant as Q

RNG = np.random.default_rng(99)


# ---- rounding ------------------------------------------------------------------------------
def test_llround_is_half_away_from_zero():
    cases = {2.5: 3, -2.5: -3, 0.5: 1, -0.5: -1, 1.4999: 1, -1.4999: -1, 3.0: 3, 0.0: 0,
             0.49999999999999994: 0, -0.49999999999999994: 0, 1e15 + 0.5: 1000000000000001}
    for x, want in cases.items():
        assert Q.llround(x) == want, x
    assert round(2.5) == 2 and Q.llround(2.5) == 3  # Python round() is half-even; C llround is not
    xs = np.array(list(cases))
    assert Q.llround_arr(xs).tolist() == list(cases.values())
    with pytest.raises(ValueError):
        Q.llround(float("nan"))


def test_quant_i8_i16_round_half_away_and_saturate():
    assert Q.quant_i8(np.array([2.5, -2.5, 300.0, -300.0, 0.4]), 1.0).tolist() == [3, -3, 127, -128, 0]
    assert Q.quant_i16(np.array([1.5, -1.5, 1e9]), -1).tolist() == [3, -3, 32767]  # 1.5 / 2^-1 = 3.0
    assert Q.quant_i16(np.array([2.5]), 0).tolist() == [3]


# ---- (M, S) normalisation ------------------------------------------------------------------
def test_normalize_mulshift_matches_frexp_definition():
    for r in [1e-9, 1e-6, 0.001, 0.37, 0.5, 1.0, 1.0 - 2.0**-40, 3.5, 1000.0, 2.0**30, 2.0**30.999]:
        M, S = Q.normalize_mulshift(r)
        f, e = math.frexp(r)
        assert 0 <= S <= 63
        if S < 63:
            assert 2**30 <= M < 2**31
            assert abs(M * 2.0**-S / r - 1) <= 2.0**-30
        else:
            assert M == Q.llround(math.ldexp(r, 63))
    # f rounds up to 1.0 at 31 bits -> M = 2^30 with S one smaller (still exact)
    M, S = Q.normalize_mulshift(1.0 - 2.0**-40)
    assert (M, S) == (2**30, 30)
    # tiny ratio: S pinned at 63, M may be 0
    assert Q.normalize_mulshift(2.0**-70) == (0, 63)
    assert Q.normalize_mulshift(2.0**-40) == (2**23, 63)
    with pytest.raises(ValueError):
        Q.normalize_mulshift(2.0**31)
    with pytest.raises(ValueError):
        Q.normalize_mulshift(0.0)


def test_i16_exponent_and_i8_scale():
    assert Q.i16_exponent(32767.0) == 0 and Q.i16_exponent(32767.5) == 1 and Q.i16_exponent(1.0) == -14
    assert Q.i16_exponent(0.5) == -15 and Q.i16_exponent(1e-9) == -15 and Q.i16_exponent(0.0) == -15
    assert Q.i8_scale(127.0) == 1.0 and Q.i8_scale(0.0) == 1.0 and Q.i8_scale(2.54) == 2.54 / 127.0


def test_weight_row_scales_and_bias():
    w = np.array([[1.0, -2.0, 0.5], [0.0, 0.0, 0.0], [-127.0, 3.0, 3.0]])
    assert Q.weight_row_scales(w).tolist() == [2.0 / 127.0, 1.0, 1.0]
    q = Q.quant_i8(w, Q.weight_row_scales(w)[:, None])
    assert q.tolist() == [[64, -127, 32], [0, 0, 0], [-127, 3, 3]]  # 63.5 -> 64 (away from zero), 31.75 -> 32
    assert Q.gemm_bias(1.0, 0.5, 0.5) == 4 and Q.gemm_bias(1e30, 1.0, 1.0) == 2**31 - 1 and Q.gemm_bias(-1e30, 1.0, 1.0) == -(2**31)


# ---- parameter formulas realised through the golden primitives -----------------------------
def test_gemm_rq_realises_scale():
    """requant16(acc) * 2^e_out ~= acc * s_a * s_w, requant8(acc) * s_out ~= acc * s_a * s_w."""
    s_a, s_w, e_out, s_out = 0.031, 0.0047, -9, 0.02
    acc = RNG.integers(-2**20, 2**20, 1000)
    M, S = Q.gemm_rq(s_a, s_w, False, e_out, 0.0)
    y = G.requant16(acc, M, S)
    ref = acc * s_a * s_w / 2.0**e_out
    ok = np.abs(ref) < 32767
    assert np.all(np.abs(y[ok] - ref[ok]) <= 0.5 + 1e-6 * np.abs(ref[ok]))
    M, S = Q.gemm_rq(s_a, s_w, True, 0, s_out)
    y = G.requant8(acc[:50] // 4096, M, S)
    ref = (acc[:50] // 4096) * s_a * s_w / s_out
    assert np.all(np.abs(y - np.clip(ref, -128, 127)) <= 0.5 + 1e-6)


def test_quant_params_realises_scale():
    e_x, s_y = -11, 0.013
    M, S = Q.quant_params(e_x, s_y)
    x = RNG.integers(-32768, 32768, 2000)
    y = G.vquant(x, M, S)
    ref = np.clip(x * 2.0**e_x / s_y, -128, 127)
    assert np.all(np.abs(y - ref) <= 0.5 + 1e-6 * np.abs(ref))


def test_rmsnorm_r_rule_is_largest_feasible_and_output_scales():
    K, eps, e_x, e_g, e_y = 128, 1e-5, -13, -14, -12
    absmax = 3.9
    p = Q.rmsnorm_params(K, eps, e_x, e_g, e_y, absmax, "t")
    assert p["eps_t"] == Q.llround(eps * K * 2.0 ** (-2 * e_x)) and p["sh_post"] == p["R"] - e_g + e_y

    def feasible(R):
        C = Q.llround(math.ldexp(math.sqrt(K), R))
        rms_q = (absmax / 4.0) / 2.0**e_x
        r_typ = max(1.0, math.floor(math.sqrt(K * rms_q * rms_q + p["eps_t"])))
        return C < 2**32 and math.floor(C / r_typ) < 65535 / 4

    assert feasible(p["R"]) and not any(feasible(R) for R in range(p["R"] + 1, 32))
    # a row with rms = absmax/4 gets inv ~ 65535/4 (no clamp), and the output matches the float formula
    g = RNG.uniform(0.5, 1.5, K)
    gq = Q.quant_i16(g, e_g).astype(np.int64)
    xr = RNG.normal(0, absmax / 4, K)
    xr *= (absmax / 4) / np.sqrt(np.mean(xr**2))
    x = Q.quant_i16(xr, e_x).astype(np.int64)
    y = G.rmsnorm(x[None], gq, p["eps_t"], p["C"], p["sh_post"])[0]
    xq = x * 2.0**e_x
    ref = xq / np.sqrt(np.mean(xq**2) + eps) * (gq * 2.0**e_g)
    assert np.abs(y * 2.0**e_y - ref).max() / np.abs(ref).max() < 2e-3
    # the realised inv of that row is below the 65535/4 budget the R rule targets
    r = G.isqrt48(int(np.sum(x * x)) + p["eps_t"])
    assert G.udiv(p["C"], r) < 65535 / 4
    with pytest.raises(ValueError, match="sh_post"):
        Q.rmsnorm_params(K, eps, e_x, 20, -15, absmax, "t")  # R - 20 + (-15) < 0


def test_silu_mul_add_shifts_realise_exponents():
    e_g, e_sg, e_u, e_f = -12, -13, -12, -14
    sp = Q.silu_params(e_g, e_sg, "t")
    assert sp == {"Mi": 2**30, "Si": 30 - (e_g + 12), "sh_out": 16 + e_sg - e_g}
    x = RNG.integers(-20000, 16000, 500)  # silu(x) at e_sg = e_g - 1 saturates above ~4.0
    y = G.silu16(x, sp["Mi"], sp["Si"], sp["sh_out"])
    xr = x * 2.0**e_g
    assert np.abs(y * 2.0**e_sg - xr / (1 + np.exp(-xr))).max() < 0.01
    sh = Q.mul_shift(e_sg, e_u, e_f, "t")
    assert sh == e_f - e_sg - e_u == 11  # a right shift: NUMERICS.md's e_a + e_b - e_y would be -11
    a, b = RNG.integers(-2000, 2000, 500), RNG.integers(-2000, 2000, 500)
    prod = G.vmul(a, b, sh)
    assert np.abs(prod * 2.0**e_f - (a * 2.0**e_sg) * (b * 2.0**e_u)).max() <= 0.5 * 2.0**e_f
    assert Q.add_shift(-8, -8, "t") == 0 and Q.add_shift(-8, -10, "t") == 2
    with pytest.raises(ValueError):
        Q.add_shift(-10, -8, "t")
    with pytest.raises(ValueError):
        Q.mul_shift(-2, -2, -14, "t")
    with pytest.raises(ValueError):
        Q.silu_params(20, 20, "t")


def test_attn_params_and_rope_tables():
    p = Q.attn_params(0.04, 0.05, 0.01, 0.008, 32)
    assert p["Ms"] * 2.0 ** -p["Ss"] == pytest.approx(256 * 0.04 * 0.05 / math.sqrt(32), rel=2**-30)
    assert p["Mo"] * 2.0 ** -p["So"] == pytest.approx(0.01 / (256 * 0.008), rel=2**-30)
    cos_t, sin_t = Q.rope_tables(40, 16, 10000.0)
    assert cos_t.shape == sin_t.shape == (40, 8) and cos_t.dtype == np.int16
    inv = 10000.0 ** (-np.arange(0, 16, 2) / 16)
    ang = np.outer(np.arange(40), inv)
    assert np.abs(cos_t - 16384 * np.cos(ang)).max() <= 0.5 + 1e-9 and np.abs(sin_t - 16384 * np.sin(ang)).max() <= 0.5 + 1e-9
    assert cos_t[0].tolist() == [16384] * 8 and sin_t[0].tolist() == [0] * 8


# ---- the qgraph contract on a real export --------------------------------------------------
LINEAR_KEYS = {"op", "in", "w", "rq", "bias", "out", "N", "K", "out_dtype", "epilogue", "aux", "aux_shift", "silu"}


@pytest.fixture(scope="module")
def qgraphs(tiny_export, tmp_path_factory):
    ex = tiny_export[0]
    base = tmp_path_factory.mktemp("qref")
    return Q.quantize_export(ex, base / "v1", fusion=False), Q.quantize_export(ex, base / "v2", fusion=True), base


def test_qgraph_schema_and_policies(qgraphs, tiny_export):
    q1, q2, base = qgraphs
    ex, model, cfg, itos = tiny_export
    calib = json.loads((ex / "calib.json").read_text())
    m = q1["model"]
    assert m["vocab"] == 16 and m["vocab_padded"] == 16 and m["dim"] == 32 and q1["input"] == "input" and q1["output"] == "logits"
    assert set(q1["tensors"]) == set(q2["tensors"])
    assert (base / "v1" / "qweights.bin").read_bytes() == (base / "v2" / "qweights.bin").read_bytes()  # fusion: same data
    assert (base / "v1" / "tokenizer.json").exists()
    t = q1["tensors"]
    assert t["embed"] == {"dtype": "i16", "shape": [16, 32], "offset": 0, "exp": m["E_RES"]}
    assert t["rope_cos"]["shape"] == [64, 8] and t["l0_wk"]["shape"] == [16, 32] and t["l0_wk.rq"] == {"dtype": "rq", "shape": [16], "offset": t["l0_wk.rq"]["offset"]}
    assert t["l0_bq"]["dtype"] == "i32" and t["l0_attn_norm"]["dtype"] == "i16" and "exp" in t["l0_attn_norm"]
    assert all(v["offset"] % 64 == 0 for v in t.values())
    # E_RES is the max over residual tensors and their addends; every one of them is at E_RES
    res = ["input"] + [f"l{i}.{n}" for i in range(2) for n in ("x1", "x2", "o", "d")]
    assert m["E_RES"] == max(Q.i16_exponent(calib[n]) for n in res) and all(q1["exps"][n] == m["E_RES"] for n in res)
    assert m["E_LOGIT"] == Q.i16_exponent(calib["logits"]) == q1["exps"]["logits"]
    # rope preserves the exponent; quant inserted before every i16-input linear and before the kv write
    for i in range(2):
        assert q1["exps"][f"l{i}.q"] == q1["exps"][f"l{i}.qr"] >= Q.i16_exponent(calib[f"l{i}.qr"])
        assert q1["exps"][f"l{i}.k"] == q1["exps"][f"l{i}.kr"]
        assert q1["scales"][f"l{i}.h.q"] == calib[f"l{i}.h"] / 127 and q1["scales"][f"l{i}.v"] == calib[f"l{i}.v"] / 127
        assert q1["scales"][f"l{i}.a"] == calib[f"l{i}.a"] / 127
    ops = q1["ops"]
    kinds = [o["op"] for o in ops]
    layer = ["rmsnorm", "quant", "linear", "linear", "linear", "rope", "rope", "quant", "quant", "kv_write", "attention",
             "linear", "add", "rmsnorm", "quant", "linear", "linear", "silu", "mul", "quant", "linear", "add"]
    assert kinds == layer * 2 + ["rmsnorm", "quant", "linear"]
    for o in ops:
        if o["op"] == "linear":
            assert set(o) == LINEAR_KEYS and o["epilogue"] == "none" and o["aux"] is None and o["silu"] is None
            assert o["out_dtype"] == ("i8" if o["out"].endswith(".v") else "i16")
            assert o["in"].endswith(".q") or o["in"].endswith(".a")  # every linear input is i8
            assert (o["bias"] is not None) == (o["out"].split(".")[-1] in ("q", "k", "v"))
        elif o["op"] == "quant":
            assert o["out"] == o["in"] + ".q" and 2**30 <= o["M"] < 2**31
        elif o["op"] == "rmsnorm":
            assert set(o) == {"op", "in", "gamma", "out", "K", "eps_t", "C", "sh_post"} and o["C"] < 2**32
        elif o["op"] == "attention":
            assert set(o) == {"op", "q", "layer", "out", "H", "Hkv", "D", "Ms", "Ss", "Mo", "So"} and o["q"].endswith(".qr.q")
        elif o["op"] == "kv_write":
            assert set(o) == {"op", "layer", "k", "v", "Hkv", "D"} and o["k"].endswith(".kr.q") and o["v"].endswith(".v")
        elif o["op"] == "add":
            assert o["sh_b"] == 0
    # v2: fused forms replace add/silu/mul; the requant of the fused linear targets the pre-epilogue exponent
    k2 = [o["op"] for o in q2["ops"]]
    assert "add" not in k2 and "silu" not in k2 and "mul" not in k2 and k2.count("linear") == kinds.count("linear")
    fused = {o["out"]: o for o in q2["ops"] if o["op"] == "linear" and o["epilogue"] != "none"}
    assert set(fused) == {f"l{i}.{n}" for i in range(2) for n in ("x1", "x2", "sg", "f")}
    assert fused["l0.x1"]["aux"] == "input" and fused["l1.x1"]["aux"] == "l0.x2" and fused["l0.x2"]["aux"] == "l0.x1"
    s1 = next(o for o in ops if o["op"] == "silu" and o["out"] == "l0.sg")
    assert fused["l0.sg"]["silu"] == {"Mi": s1["Mi"], "Si": s1["Si"], "sh_out": s1["sh_out"]}
    m1 = next(o for o in ops if o["op"] == "mul" and o["out"] == "l0.f")
    assert fused["l0.f"]["aux"] == "l0.sg" and fused["l0.f"]["aux_shift"] == m1["sh"] and fused["l0.f"]["rq"] == "l0_wu.rq"
    assert q2["fusion"] is True and q1["fusion"] is False


def test_refquant_cli(tiny_export, tmp_path, capsys):
    ex = tiny_export[0]
    Q.main([str(ex), "-o", str(tmp_path / "q"), "--fusion"])
    out = capsys.readouterr().out
    assert "fusion=True" in out and (tmp_path / "q" / "qgraph.json").exists()
    assert json.loads((tmp_path / "q" / "qgraph.json").read_text())["fusion"] is True
