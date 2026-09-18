import numpy as np
import pytest
import torch
from llaccel.model import TinyLlama, ModelConfig
from llaccel.export import import_model
from llaccel.calibrate import calibrate_tokens

@pytest.fixture(scope='module')
def graph():
    torch.manual_seed(84)
    model=TinyLlama(ModelConfig(dim=16,n_layers=1,n_heads=1,n_kv_heads=1,head_dim=16,ffn=32,vocab=32,max_seq=16))
    return import_model(model,model.cfg)

def test_tokenizer_ids_calibrate_exported_operations(graph):
    result=calibrate_tokens(graph,[[31,28,7,2],[1,30,4]])
    assert {k for k in result if not k.endswith(".rms_min")}==set(graph.node_names.values())
    assert result["input.rms_min"] > 0
    assert result["input.rms_min"] <= result["input"]
    assert all(np.isfinite(v) and v>=0 for v in result.values())
    assert result['input']>0

@pytest.mark.parametrize('sequences',[[],[[1]],[[0]*17],[[1,32]],[[1,1.0]],[[True,1]]])
def test_invalid_token_calibration(graph,sequences):
    with pytest.raises(ValueError):calibrate_tokens(graph,sequences)

def test_selected_quantization_bounds_are_explicit_and_raw_max_is_preserved(graph):
    sequences=[[31,28,7,2],[1,30,4]]
    baseline=calibrate_tokens(graph,sequences)
    selected=calibrate_tokens(graph,sequences,optimize_quantization=True,sample_budget=256)
    for name,maximum in baseline.items():
        assert selected[name] == maximum
        if name.endswith('.rms_min'):continue
        for bits in (8,16):
            assert 0 <= selected[name+f'.i{bits}_absmax'] <= maximum
