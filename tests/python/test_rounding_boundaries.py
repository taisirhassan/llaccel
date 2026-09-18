import numpy as np
from llaccel.golden import rshr


def test_rounding_full_i64_domain():
    rng = np.random.default_rng(17)
    values = np.concatenate((np.array([-(1 << 63), (1 << 63) - 1, -1, 0, 1], dtype=np.int64),
                             rng.integers(-(1 << 63), (1 << 63) - 1, size=256, dtype=np.int64)))
    for shift in range(64):
        expected = np.array([int(v) if shift == 0 else (int(v) + (1 << (shift - 1))) >> shift
                             for v in values], dtype=np.int64)
        np.testing.assert_array_equal(rshr(values, shift), expected)
    shifts = np.arange(64, dtype=np.int64)
    value = np.int64((1 << 63) - 1)
    expected = [int(value) if int(s) == 0 else (int(value) + (1 << (int(s) - 1))) >> int(s) for s in shifts]
    np.testing.assert_array_equal(rshr(value, shifts), expected)


def test_wide_attention_preserves_uniform_constant_values():
    from llaccel.golden import attention_head
    for length in range(1, 257):
        q = np.zeros(16, dtype=np.int8)
        keys = np.zeros((length, 16), dtype=np.int8)
        for value in (-128, -1, 0, 1, 127):
            vals = np.full((length, 16), value, dtype=np.int8)
            got = attention_head(q, keys, vals, 1, 0, 1 << 30, 38, prob_bits=15)
            assert np.max(np.abs(got - value)) <= 1
    legacy = attention_head(q, np.zeros((171,16)), np.full((171,16),127), 1,0,1<<30,38)
    assert int(legacy[0]) == 85  # preserve old instruction semantics
