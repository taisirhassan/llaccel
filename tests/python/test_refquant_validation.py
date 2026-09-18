import numpy as np
import pytest
from llaccel import refquant as Q


def test_quant_saturates_before_int64_conversion():
    values = np.array([-1e300, 1e300])
    assert Q.quant_i8(values, 1.0).tolist() == [-128, 127]
    assert Q.quant_i16(values, -15).tolist() == [-32768, 32767]
    assert Q.quant_i8(np.array([1.0]), 1e-300).tolist() == [127]


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_refquant_rejects_nonfinite_values(value):
    with pytest.raises(ValueError): Q.llround_arr(np.array([value]))
    with pytest.raises(ValueError): Q.quant_i8(np.array([value]), 1.0)
    with pytest.raises(ValueError): Q.quant_i16(np.array([value]), 0)


def test_refquant_rejects_invalid_scales():
    for scale in [0.0, -1.0, float('nan'), float('inf')]:
        with pytest.raises(ValueError): Q.quant_i8(np.array([1.0]), scale)
    with pytest.raises(ValueError): Q.llround_arr(np.array([1e300]))


@pytest.mark.parametrize("eps", [-1.0, 2**32, 2**40])
def test_rmsnorm_rejects_epsilon_outside_isa_operand(eps):
    with pytest.raises(ValueError, match="ISA u32"):
        Q.rmsnorm_params(1, eps, 0, 0, 0, 1.0, "test")
