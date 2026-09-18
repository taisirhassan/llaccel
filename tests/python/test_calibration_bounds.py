import numpy as np
import pytest
from llaccel.calibration_bounds import QuantizationSamples
from llaccel.refquant import RefQuantizer


def test_selected_bound_never_increases_sampled_mse():
    rng = np.random.default_rng(28)
    sampler = QuantizationSamples(1024)
    maximum = 0
    for _ in range(8):
        values = rng.normal(size=4096).astype(np.float32)
        values[0] = 9.0
        maximum = max(maximum, float(abs(values).max()))
        sampler.add(values)
    for bits in (8, 16):
        result = sampler.select(maximum, bits)
        assert 0 < result['absmax'] <= maximum
        assert result['mse'] <= result['unclipped_mse']
        assert result['sampled_values'] <= 8 * 1025


def test_constant_zero_and_extreme_samples():
    sampler = QuantizationSamples(4)
    sampler.add(np.zeros(64, dtype=np.float32))
    assert sampler.select(0, 8)['absmax'] == 0
    with pytest.raises(ValueError):
        sampler.add([np.inf])


def test_optional_bounds_preserve_raw_maximum():
    q = RefQuantizer({}, {}, {'x': 8, 'x.i8_absmax': 4, 'x.i16_absmax': 4}, False)
    assert q.absmax('x') == 8
    assert q.s8('x.q') == pytest.approx(4 / 127)
    assert q.e16('x') == -12
    bad = RefQuantizer({}, {}, {'x': 8, 'x.i8_absmax': 9}, False)
    with pytest.raises(ValueError, match='invalid selected'):
        bad.s8('x.q')
