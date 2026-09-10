"""(b) golden primitives: exact properties and float-reference tolerances."""
from __future__ import annotations

import math

import numpy as np
import pytest

from llaccel import golden as G
from llaccel.refquant import exp_for, mulshift_params

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
        M, S = mulshift_params(r)
        assert 2**30 <= M < 2**31 and 0 <= S <= 63
        assert abs(M * 2.0**-S / r - 1) < 1e-8
    assert exp_for(32767.0) == 0 and exp_for(1.0) == -14 and exp_for(0.5) == -15


# ---- float references ----------------------------------------------------------------------
def test_silu16_vs_float():
    e_x, e_y = -12, -12
    x = RNG.integers(-32768, 32768, 4096)
    Mi, Si = mulshift_params(2.0 ** (e_x + 12))
    y = G.silu16(x, Mi, Si, 16 + e_y - e_x)
    xr = x * 2.0**e_x
    ref = xr / (1 + np.exp(-xr))
    err = np.abs(y * 2.0**e_y - ref)
    assert err.max() < 0.02 * 8 and np.mean(err) < 0.004  # |x| < 8: LUT + linear interpolation, 1/16 step


def test_rmsnorm_vs_float():
    K, e_x, e_y = 128, -10, -12
    g = RNG.uniform(0.5, 2.0, K)
    e_g = exp_for(np.abs(g).max())
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
    s_out = 0.02
    Ms, Ss = mulshift_params(256.0 * s_q * s_k / math.sqrt(D))
    Mo, So = mulshift_params(s_v / (256.0 * s_out))
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
        assert err.max() < 0.15, (T, err.max())  # u8 probabilities + i8 output
        assert np.corrcoef(out * s_out, ref)[0, 1] > 0.99


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
