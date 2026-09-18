"""Regressions for input validation and verification false positives."""
import json
import numpy as np
import pytest
from llaccel.data import CharTokenizer, sample_batch
from llaccel.model import ModelConfig
from llaccel.verify import compare_sim


def test_batch_can_use_only_valid_window():
    x, y = sample_batch(np.arange(5), 2, 4, np.random.default_rng(0))
    assert x.tolist() == [[0, 1, 2, 3]] * 2
    assert y.tolist() == [[1, 2, 3, 4]] * 2


def test_batch_includes_last_valid_window():
    x, _ = sample_batch(np.arange(6), 100, 4, np.random.default_rng(0))
    assert set(x[:, 0]) == {0, 1}
    with pytest.raises(ValueError, match='context'):
        sample_batch(np.arange(4), 1, 4, np.random.default_rng(0))


@pytest.mark.parametrize('entries', [[], ['a', 'a'], ['ab'], [1]])
def test_tokenizer_rejects_invalid_entries(entries):
    with pytest.raises(ValueError): CharTokenizer(entries)


@pytest.mark.parametrize('field,value', [('n_kv_heads', 0), ('n_heads', 0), ('dim', -1),
    ('vocab', 0), ('max_seq', 0), ('rms_eps', float('nan')), ('rope_base', float('inf'))])
def test_model_config_rejects_invalid_dimensions(field, value):
    with pytest.raises(ValueError): ModelConfig(**{field: value})


def test_compare_sim_requires_complete_integer_logits(tmp_path):
    reference = {'generated': [1], 'argmax_per_step': [1, 0], 'logits_last_rows': [[2, 3], [4, 1]]}
    golden, sim = tmp_path / 'gold.json', tmp_path / 'sim.json'
    golden.write_text(json.dumps(reference))
    sim.write_text(json.dumps(reference))
    assert compare_sim(sim, golden)[0]
    invalid = [
        {k: v for k, v in reference.items() if k != 'logits_last_rows'},
        dict(reference, logits_last_rows=[]),
        dict(reference, logits_last_rows=[[2, 3]]),
        dict(reference, logits_last_rows=[[2], [4, 1]]),
        dict(reference, logits_last_rows=[[2.1, 3], [4, 1]]),
        dict(reference, generated=[1.1]),
    ]
    for record in invalid:
        sim.write_text(json.dumps(record))
        assert not compare_sim(sim, golden)[0], record


def test_calibration_rejects_invalid_parameters_before_loading_data():
    from types import SimpleNamespace
    from llaccel.calibrate import calibrate
    model = SimpleNamespace(cfg={'max_seq': 32})
    for n, length in [(0, 16), (1, 0), (1, 33)]:
        with pytest.raises(ValueError): calibrate(model, n_seqs=n, seq_len=length)


def test_calibration_rejects_nonfinite_activations():
    import torch
    from llaccel.calibrate import AbsMaxRecorder
    class Bad(torch.nn.Module):
        def forward(self, x): return x * float('nan')
    gm = torch.fx.symbolic_trace(Bad())
    node = next(n for n in gm.graph.nodes if n.op == 'call_function')
    recorder = AbsMaxRecorder(gm, {node: 'bad'})
    with pytest.raises(ValueError, match='nonfinite'):
        recorder.run(torch.ones(1))
