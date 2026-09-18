"""Representation diagnostics distinguish endpoint occupancy from clipping."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('profile_quantization',Path(__file__).resolve().parents[2]/'scripts/profile_quantization.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_endpoint_is_not_clipping():
    # Exactly representable endpoints occupy the limit without exceeding it.
    d = profile.range_metrics(np.array([-128,0,127]),np.array([-64.,0.,63.5]),8,.5)
    assert d['integer_endpoint_occupancy'] == pytest.approx(2/3)
    assert d['local_float_outside_endpoint_range'] == 0
    d = profile.range_metrics(np.array([-128,0,127]),np.array([-65.,0.,64.]),8,.5)
    assert d['local_float_outside_endpoint_range'] == pytest.approx(2/3)


def test_metric_reference_direction_and_zero():
    d = profile.metrics([2.,4.],[1.,2.])
    assert d['cosine'] == pytest.approx(1)
    assert d['relative_l2_error'] == pytest.approx(1)
    assert profile.metrics([0.],[0.])['cosine'] == 1
    assert profile.metrics([1.],[0.])['cosine'] == 0
    with pytest.raises(ValueError): profile.metrics([float('nan')],[0.])


@pytest.mark.parametrize('ep,expected',[('none',[6.,0.]),('resadd',[7.,2.]),('mul',[6.,0.])])
def test_fused_linear_and_zero_padded_tail(ep,expected):
    import torch
    op={'op':'linear','in':'x','w':'w','N':2,'epilogue':ep,'aux':'aux'}
    actual=profile.float_operation(op,{'x':torch.tensor([[2.,3.]]),'aux':torch.tensor([[1.,2.]])},{},{},{'w':torch.tensor([[3.,0.]])},{})
    torch.testing.assert_close(actual,torch.tensor([expected]))


@pytest.mark.parametrize('tokens,threads',[(0,2),(17,2),(8,0)])
def test_invalid_scope_rejected_before_loading(tokens,threads):
    with pytest.raises(ValueError): profile.profile(Path('missing'),Path('missing'),Path('missing'),tokens,threads)


def test_rope_is_i16_until_separate_quant():
    assert profile.output_bits({'op':'rope'}) == 16
    assert profile.output_bits({'op':'quant'}) == 8
    assert profile.output_bits({'op':'attention'}) == 8
    assert profile.output_bits({'op':'linear','out_dtype':'i8'}) == 8


def test_grouped_norm_does_not_mix_head_statistics():
    import torch
    x = torch.tensor([[1., 1., 10., 10.]])
    actual = profile.float_operation({'op':'rmsnorm','in':'x','gamma':'g','heads':2},
        {'x':x},{},{},{'g':torch.ones(2)},{'rms_eps':1e-6})
    torch.testing.assert_close(actual, torch.ones_like(x), atol=1e-6, rtol=1e-6)


def test_profiler_uses_exported_scaled_rope_tables():
    import torch
    x = torch.tensor([[1., 2., 3., 4.]])
    actual = profile.float_operation({'op':'rope','in':'x','H':1,'D':4},
        {'x':x},{},{},{'rope_cos_input':torch.zeros(1,2), 'rope_sin_input':torch.full((1,2),1.25)},
        {'rope_base':10000.})
    torch.testing.assert_close(actual,torch.tensor([[-3.75,-5.,1.25,2.5]]))
