"""(b) golden primitives: exact properties and float-reference tolerances."""
from __future__ import annotations

import math

import numpy as np
import pytest

from llaccel import golden as G
from llaccel.refquant import i16_exponent, normalize_mulshift

RNG = np.random.default_rng(1234)


# ---- exact properties ----------------------------------------------------------------------
def test_rshr_round_half_up():
    assert G.rshr(np.array([5]), 1)[0] == 3       # 2.5 -> 3
    assert G.rshr(np.array([-5]), 1)[0] == -2     # -2.5 -> -2 (half-up == towards +inf)
    assert G.rshr(np.array([-6]), 1)[0] == -3
    assert G.rshr(np.array([7]), 2)[0] == 2       # 1.75 -> 2
    assert G.rshr(np.array([6]), 2)[0] == 2       # 1.5 -> 2
    assert G.rshr(np.array([-7]), 0)[0] == -7
    v = RNG.integers(-2**40, 2**40, 1000)
    s = RNG.integers(0, 40, 1000)
    ref = np.array([x if k == 0 else math.floor((x + 2 ** (k - 1)) / 2**k) for x, k in zip(v.tolist(), s.tolist())])
    assert np.array_equal(G.rshr(v, s), ref)
    # scalar-shift path agrees with array-shift path
    assert np.array_equal(G.rshr(v, 7), G.rshr(v, np.full(1000, 7)))


def test_saturation():
    v = np.array([-10**9, -32769, -32768, -129, -128, 0, 127, 128, 255, 256, 32767, 32768, 65535, 65536, 10**9])
    assert G.sat8(v).tolist() == [-128, -128, -128, -128, -128, 0, 127, 127, 127, 127, 127, 127, 127, 127, 127]
    assert G.satu8(v).tolist() == [0, 0, 0, 0, 0, 0, 127, 128, 255, 255, 255, 255, 255, 255, 255]
    assert G.sat16(v).tolist() == [-32768, -32768, -32768, -129, -128, 0, 127, 128, 255, 256, 32767, 32767, 32767, 32767, 32767]
    assert G.satu16(v).tolist() == [0, 0, 0, 0, 0, 0, 127, 128, 255, 256, 32767, 32768, 65535, 65535, 65535]


def test_isqrt48_loop_equals_math_isqrt():
    vals = RNG.integers(0, 2**48, 10_000).tolist() + [0, 1, 2, 3, 4, 2**48 - 1, 2**47, 2**46 - 1, 65535**2, 65536**2]
    for t in vals:
        assert G.isqrt48(t) == math.isqrt(t)


def test_udiv():
    assert G.udiv(10, 3) == 3
    assert G.udiv(10, 0) == 10          # b == 0 -> divide by 1 (numerics.h)
    assert G.udiv(2**31, 65534) == 32769
    assert G.udiv(2**40, 1) == (2**40) & 0xFFFFFFFF  # truncated to u32 like uint32_t(a / b)


def test_mulshift_params_and_exp():
    for r in [1e-6, 0.001, 0.37, 1.0, 3.5, 1000.0]:
        M, S = normalize_mulshift(r)
        assert 2**30 <= M < 2**31 and 0 <= S <= 63
        assert abs(M * 2.0**-S / r - 1) < 1e-8
    assert i16_exponent(32767.0) == 0 and i16_exponent(1.0) == -14 and i16_exponent(0.5) == -15


def test_mulshift_realises_ratio_within_rounding():
    """mulshift(v, M, S) == round(v * r) up to the 2^-31 relative error of M, for i32 v."""
    for r in [0.5, 0.37, 1.0 / 3.0, 3.5, 2.0**-10, 1000.0]:
        M, S = normalize_mulshift(r)
        v = RNG.integers(-(2**30), 2**30, 2000)
        got = G.mulshift(v, M, S)
        exact = v.astype(np.float64) * r
        assert np.all(np.abs(got - exact) <= 0.5 + np.abs(exact) * 2.0**-30)


# ---- float references ----------------------------------------------------------------------
def test_silu16_vs_float():
    e_x, e_y = -12, -12
    x = RNG.integers(-32768, 32768, 4096)
    Mi, Si = normalize_mulshift(2.0 ** (e_x + 12))
    y = G.silu16(x, Mi, Si, 16 + e_y - e_x)
    xr = x * 2.0**e_x
    ref = xr / (1 + np.exp(-xr))
    err = np.abs(y * 2.0**e_y - ref)
    assert err.max() < 0.02 * 8 and np.mean(err) < 0.004  # |x| < 8: LUT + linear interpolation, 1/16 step


def test_rmsnorm_vs_float():
    K, e_x, e_y = 128, -10, -12
    g = RNG.uniform(0.5, 2.0, K)
    e_g = i16_exponent(np.abs(g).max())
    gq = np.round(g / 2.0**e_g).astype(np.int64)
    R = 24
    C = int(round(2.0**R * math.sqrt(K)))
    sh_post = R - e_g + e_y
    eps = 1e-5
    eps_t = int(round(eps * K * 2.0 ** (-2 * e_x)))
    for _ in range(20):
        xr = RNG.normal(0, RNG.uniform(0.3, 3.0), K)
        x = np.clip(np.round(xr / 2.0**e_x), -32768, 32767).astype(np.int64)
        y = G.rmsnorm(x[None, :], gq, eps_t, C, sh_post)[0]
        xq = x * 2.0**e_x
        ref = xq / np.sqrt(np.mean(xq**2) + eps) * g
        rel = np.abs(y * 2.0**e_y - ref) / (np.abs(ref).max())
        assert rel.max() < 2e-3


def test_rope_vs_float():
    H, D, base, e = 4, 32, 10000.0, -12
    inv_freq = base ** (-np.arange(0, D, 2) / D)
    ang = np.outer(np.arange(64), inv_freq)
    cos_t = np.round(np.cos(ang) * 16384).astype(np.int64)
    sin_t = np.round(np.sin(ang) * 16384).astype(np.int64)
    pos = 17
    x = RNG.integers(-20000, 20000, (3, H * D))
    y = G.rope(x, H, D, cos_t, sin_t, pos)
    xr = (x * 2.0**e).reshape(3, H, D)
    c = np.concatenate([np.cos(ang), np.cos(ang)], -1)[pos : pos + 3][:, None, :]
    s = np.concatenate([np.sin(ang), np.sin(ang)], -1)[pos : pos + 3][:, None, :]
    rot = np.concatenate([-xr[..., D // 2 :], xr[..., : D // 2]], -1)
    ref = (xr * c + rot * s).reshape(3, H * D)
    assert np.abs(y * 2.0**e - ref).max() < 3 * 2.0**e


def test_attention_head_vs_float_softmax():
    D = 32
    s_q = s_k = 0.02
    s_v = 0.03
    s_out = 0.03  # a softmax-weighted average of v never exceeds max|v|, so s_out = s_v cannot saturate
    Ms, Ss = normalize_mulshift(256.0 * s_q * s_k / math.sqrt(D))
    Mo, So = normalize_mulshift(s_v / (256.0 * s_out))
    for T in (1, 5, 40, 200):
        q = RNG.integers(-127, 128, D)
        k = RNG.integers(-127, 128, (T, D))
        v = RNG.integers(-127, 128, (T, D))
        out = G.attention_head(q, k, v, Ms, Ss, Mo, So)
        sc = (k * s_k) @ (q * s_q) / math.sqrt(D)
        p = np.exp(sc - sc.max())
        p /= p.sum()
        ref = p @ (v * s_v)
        err = np.abs(out * s_out - ref)
        # u8 probabilities (1/256 each, T of them) + i8 output (s_out/2) + 65535-scaled exp LUTs
        assert err.max() < 0.15, (T, err.max())
        assert np.corrcoef(out * s_out, ref)[0, 1] > 0.99


def test_attention_single_key_is_identity_up_to_pn_rounding():
    """T = 1: p = 65534, inv = 32769, pn = satu8((65534*32769 + 2^22) >> 23) = 255, so o = 255 * v."""
    D = 16
    q = RNG.integers(-127, 128, D)
    v = RNG.integers(-127, 128, (1, D))
    Mo, So = normalize_mulshift(1.0 / 255.0)  # exactly undo the 255 so out == v
    out = G.attention_head(q, RNG.integers(-127, 128, (1, D)), v, 2**30, 40, Mo, So)
    assert np.array_equal(out, v[0])


def test_attention_probabilities_sum_property():
    """The max-score key gets p = 65534 exactly (NUMERICS: 'max 65534')."""
    z = np.array([0])
    p = (G.EXPI[z >> 8] * G.EXPF[z & 255] + (1 << 15)) >> 16
    assert p[0] == 65534


def test_vmul_vadd_vquant():
    a = np.array([1000, -1000, 32767])
    b = np.array([1000, 1000, 32767])
    assert G.vmul(a, b, 10).tolist() == [977, -977, 32767]
    assert G.vadd(a, b, 1).tolist() == [1500, -500, 32767]
    assert G.vquant(np.array([32767, -32768, 100]), 2**30, 38).tolist() == [127, -128, 0]


def test_silu16_exact_lut_points():
    """At u = 256*j (f = 0) the interpolation is exact: y = rshr(x * SIG[j+128], sh_out)."""
    Mi, Si, sh = 2**30, 30, 16  # e_x = -12: u = x (Q3.12 already)
    for x in (-32768, -4096, -256, 0, 256, 4096, 32512):
        idx = (x >> 8) + 128
        assert G.silu16(np.array([x]), Mi, Si, sh)[0] == G.sat16(G.rshr(np.array([x * int(G.SIG[idx])]), sh))[0]
    assert G.silu16(np.array([0]), Mi, Si, sh)[0] == 0
    # at a LUT point the only errors are the two roundings (table entry, output shift): <= 1 unit
    for x in (32000, -32000, 256 * 100):
        ref = x / (1.0 + math.exp(-x / 4096.0))
        assert abs(int(G.silu16(np.array([x]), Mi, Si, sh)[0]) - ref) <= 1.0


def test_rmsnorm_zero_row_and_inv_clamp():
    """All-zero row (prefill padding) stays zero; tiny rows hit the inv = 65535 clamp without overflow."""
    K = 32
    g = np.full(K, 16384)
    C = int(round(2.0**24 * math.sqrt(K)))
    assert np.all(G.rmsnorm(np.zeros((1, K), np.int64), g, 3, C, 20) == 0)
    y = G.rmsnorm(np.ones((1, K), np.int64), g, 0, C, 20)[0]
    assert np.all(y == G.sat16(G.rshr(np.array([16384 * 65535]), 20))[0])


def test_rope_position_zero_is_identity_and_exponent_preserved():
    D, H = 16, 2
    x = RNG.integers(-32768, 32768, (2, H * D))
    cos_t = np.full((4, D // 2), 16384)  # cos(0) = 1.0 in Q1.14
    sin_t = np.zeros((4, D // 2), np.int64)
    assert np.array_equal(G.rope(x, H, D, cos_t, sin_t, 0), x)


def test_attention_additive_score_offset_invariance():
    for prob_bits in (8, 15):
        for dim in (16, 32, 64):
            q = np.zeros(dim, dtype=np.int64)
            q[0] = 127
            vals = np.zeros((2, dim), dtype=np.int64)
            vals[:, 0] = [-127, 127]
            outputs = []
            for offset in (0, 126, -128):
                keys = np.zeros((2, dim), dtype=np.int64)
                keys[:, 0] = [offset, offset + 1]
                outputs.append(G.attention_head(q, keys, vals, 1 << 30, 24, 1 << 30, 38, prob_bits))
            assert all(np.array_equal(out, outputs[0]) for out in outputs)
            assert outputs[0][0] == 127  # gap exceeds exp cutoff, independent oracle
            # Largest valid unsigned multiplier and dot gap must not wrap.
            q.fill(127)
            keys = np.array([[-128] * dim, [127] * dim], dtype=np.int64)
            extreme = G.attention_head(q, keys, vals, 2**32 - 1, 0, 1 << 30, 38, prob_bits)
            assert extreme[0] == 127


def test_attention_common_offset_preserves_nontrivial_softmax():
    q = np.zeros(16, dtype=np.int64)
    q[0], q[1] = 1, 127
    keys = np.zeros((2, 16), dtype=np.int64)
    keys[:, 0] = [0, 1]
    vals = np.zeros((2, 16), dtype=np.int64)
    vals[:, 0] = [-127, 127]
    for prob_bits in (8, 15):
        baseline = G.attention_head(q, keys, vals, 1 << 30, 24, 1 << 30, 38, prob_bits)
        shifted = keys.copy()
        shifted[:, 1] = 126  # common scaled logit >1million, delta remains64
        out = G.attention_head(q, shifted, vals, 1 << 30, 24, 1 << 30, 38, prob_bits)
        assert np.array_equal(out, baseline)
        assert abs(int(out[0]) - 127 * math.tanh(0.125)) <= 1
