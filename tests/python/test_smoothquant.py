import copy

import pytest
import torch

from llaccel.model import ModelConfig, TinyLlama
from llaccel.smoothquant import smooth_model


def fixture(bias=False, tied=False):
    torch.manual_seed(72)
    model = TinyLlama(ModelConfig(dim=32, n_layers=2, n_heads=2, n_kv_heads=1,
        head_dim=16, ffn=48, vocab=80, max_seq=32, qkv_bias=bias)).eval()
    if bias:
        with torch.no_grad():
            for layer in model.layers:
                for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                    projection.bias.normal_(0, 0.1)
    if tied:
        model.lm_head.weight = model.embed_tokens.weight
    return model


@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('tied', [False, True])
@pytest.mark.parametrize('alpha', [0.0, 0.5, 1.0])
def test_float_equivalence(bias, tied, alpha):
    model = fixture(bias, tied)
    reference = copy.deepcopy(model)
    embedding = model.embed_tokens.weight.detach().clone()
    metadata = smooth_model(model, [list(range(16)), list(range(16, 32))], alpha, row_chunk=7)
    assert len(metadata['groups']) == 9
    assert metadata['untied_lm_head'] == tied
    torch.testing.assert_close(embedding, model.embed_tokens.weight, atol=0, rtol=0)
    with torch.no_grad():
        for length in (1, 7, 32):
            tokens = torch.arange(length).unsqueeze(0)
            torch.testing.assert_close(model(tokens), reference(tokens), atol=3e-6, rtol=3e-5)


def test_without_final_head_preserves_tie():
    model = fixture(tied=True)
    metadata = smooth_model(model, [[1, 2, 3]], smooth_lm_head=False)
    assert model.lm_head.weight is model.embed_tokens.weight
    assert metadata['untied_lm_head'] is False
    assert len(metadata['groups']) == 8


def test_activation_outlier_balanced():
    model = fixture()
    with torch.no_grad():
        model.layers[0].input_layernorm.weight[0] = 1000
    reference = copy.deepcopy(model)
    sequences = [list(range(16)), list(range(16, 32))]
    metadata = smooth_model(model, sequences, alpha=0.5)
    group = metadata['groups'][0]
    # Offline balancing moves most of the activation range to weight columns.
    assert group['activation_max_after'] < group['activation_max_before'] / 50
    assert group['weight_max_after'] > group['weight_max_before']
    assert group['activation_max_after'] == pytest.approx(group['weight_max_after'], rel=1e-5)
    with torch.no_grad():
        tokens = torch.tensor([sequences[0]])
        torch.testing.assert_close(model(tokens), reference(tokens), atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize('alpha', [-1, 1.1, float('nan'), float('inf'), True])
def test_bad_alpha(alpha):
    with pytest.raises(ValueError, match='alpha'):
        smooth_model(fixture(), [[1, 2]], alpha)


@pytest.mark.parametrize('sequences', [[], [[]], [[1] * 33], [[-1]], [[80]], [[True]], [[1.5]]])
def test_bad_tokens(sequences):
    with pytest.raises(ValueError):
        smooth_model(fixture(), sequences)


def test_no_mutation_on_nonfinite_and_hooks_removed():
    model = fixture()
    with torch.no_grad():
        model.lm_head.weight[0, 0] = float('nan')
    gamma = model.layers[0].input_layernorm.weight.detach().clone()
    model.train()
    with pytest.raises(ValueError, match='nonfinite weight'):
        smooth_model(model, [[1, 2, 3]])
    torch.testing.assert_close(gamma, model.layers[0].input_layernorm.weight, atol=0, rtol=0)
    assert model.training
    assert all(not module._forward_hooks for module in model.modules())


def test_zero_channels_remain_finite():
    model = fixture()
    with torch.no_grad():
        model.layers[0].input_layernorm.weight.zero_()
        for linear in (model.layers[0].self_attn.q_proj, model.layers[0].self_attn.k_proj,
                       model.layers[0].self_attn.v_proj):
            linear.weight.zero_()
    smooth_model(model, [[1, 2, 3]])
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_export_preserves_operation_graph():
    from llaccel.export import import_model
    model = fixture(bias=True, tied=True)
    original = import_model(model, model.cfg)
    smooth_model(model, [[1, 2, 3, 4]])
    transformed = import_model(model, model.cfg)
    assert [op.kind for op in transformed.ops] == [op.kind for op in original.ops]
    assert [op.out.name for op in transformed.ops] == [op.out.name for op in original.ops]


@pytest.mark.parametrize('values,ffn', [(True, False), (False, True), (True, True)])
def test_gqa_multiple_kv_groups_and_optional_transfers(values, ffn):
    torch.manual_seed(29)
    model = TinyLlama(ModelConfig(dim=64, n_layers=1, n_heads=4, n_kv_heads=2,
        head_dim=16, ffn=96, vocab=80, max_seq=32, qkv_bias=True)).eval()
    with torch.no_grad():
        model.layers[0].self_attn.v_proj.bias.normal_(0, 0.5)
    reference = copy.deepcopy(model)
    metadata = smooth_model(model, [list(range(16))], smooth_values=values, smooth_ffn=ffn)
    assert len(metadata['groups']) == 3 + int(values) + int(ffn)
    with torch.no_grad():
        for length in (1, 16, 32):
            tokens = torch.arange(length).unsqueeze(0)
            torch.testing.assert_close(model(tokens), reference(tokens), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('group', ['value_output', 'mlp_product_down'])
def test_transfer_outlier_range_and_actual_activations(group):
    model = fixture(bias=True)
    with torch.no_grad():
        if group == 'value_output':
            model.layers[0].self_attn.v_proj.bias[0] = 1000
        else:
            model.layers[0].mlp.up_proj.weight[0].mul_(100000)
    reference = copy.deepcopy(model)
    sequences = [list(range(16))]
    metadata = smooth_model(model, sequences)
    stats = next(g for g in metadata['groups'] if g['group'] == 'layers.0.' + group)
    assert stats['activation_max_after'] < stats['activation_max_before'] / 20
    assert stats['weight_max_after'] > stats['weight_max_before']
    measured = []
    if group == 'value_output':
        hook = model.layers[0].self_attn.v_proj.register_forward_hook(
            lambda _m, _i, out: measured.append(out.detach().abs().max().item()))
    else:
        hook = model.layers[0].mlp.down_proj.register_forward_pre_hook(
            lambda _m, inputs: measured.append(inputs[0].detach().abs().max().item()))
    with torch.no_grad():
        tokens = torch.tensor(sequences)
        actual = model(tokens)
        expected = reference(tokens)
    hook.remove()
    assert max(measured) == pytest.approx(stats['activation_max_after'], rel=2e-5)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)


def test_calibration_selected_alpha_minimizes_recorded_objective():
    model = fixture(bias=True, tied=True)
    original = copy.deepcopy(model)
    metadata = smooth_model(model, [list(range(16)), list(range(16, 32))],
                            auto_alpha=True, sample_rows=4, row_chunk=13)
    assert metadata['auto_alpha'] is True
    for group in metadata['groups']:
        selected = next(x for x in group['alpha_objective'] if x['alpha'] == group['selected_alpha'])
        assert selected['weighted_error'] == min(x['weighted_error'] for x in group['alpha_objective'])
    with torch.no_grad():
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        torch.testing.assert_close(model(tokens), original(tokens), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('qk_norm', [False, True])
def test_float_equivalence_independent_head_width_and_mlp_bias(qk_norm):
    torch.manual_seed(724)
    model = TinyLlama(ModelConfig(dim=32, n_layers=1, n_heads=4, n_kv_heads=2,
        head_dim=16, ffn=48, vocab=80, max_seq=32, qkv_bias=True,
        o_bias=True, mlp_bias=True, qk_norm=qk_norm)).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('.bias'):
                parameter.uniform_(-0.2, 0.2)
    reference = copy.deepcopy(model)
    smooth_model(model, [list(range(16)), list(range(16, 32))], auto_alpha=True, row_chunk=7)
    assert not torch.equal(model.layers[0].mlp.up_proj.bias, reference.layers[0].mlp.up_proj.bias)
    with torch.no_grad():
        for length in (1, 7, 32):
            tokens = torch.arange(length).unsqueeze(0)
            torch.testing.assert_close(model(tokens), reference(tokens), atol=3e-6, rtol=3e-5)
