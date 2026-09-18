"""Dense-family reference extensions: grouped normalization and static RoPE."""
import numpy as np
import pytest
import torch

from llaccel.calibrate import AbsMaxRecorder, calibrate_tokens
from llaccel.export import import_model
from llaccel.model import ModelConfig, TinyLlama
from llaccel.refquant import RefQuantizer, llround_arr


def test_grouped_rms_min_is_minimum_head_not_entire_projection():
    class Pass(torch.nn.Module):
        def forward(self, x):
            return x + 0
    gm = torch.fx.symbolic_trace(Pass())
    node = next(n for n in gm.graph.nodes if n.op == 'call_function')
    recorder = AbsMaxRecorder(gm, {node: 'q'}, {'q': 2})
    recorder.run(torch.tensor([[[1., 1., 100., 100.]]]))
    assert recorder.rms_min['q'] == 1.0


@pytest.mark.parametrize('scaled', [False, 'linear', 'yarn'])
def test_qk_reference_norm_groups_and_rope_exponents(scaled):
    from llaccel.rope import rope_config_from_hf
    rope = rope_config_from_hf({'rope_scaling': {'rope_type': scaled, 'factor': 4.}, 'max_position_embeddings': 32}, 'llama') if scaled else None
    cfg = ModelConfig(dim=32, n_layers=1, n_heads=4, n_kv_heads=2,
                      head_dim=16, ffn=48, vocab=32, max_seq=32, qk_norm=True,
                      rope_scaling=rope)
    torch.manual_seed(46)
    model = TinyLlama(cfg).eval()
    exported = import_model(model, cfg)
    calib = calibrate_tokens(exported, [list(range(16)), list(range(16, 32))])
    weights = {name: value.detach().numpy().astype(np.float64) for name, value in exported.weights}
    quantizer = RefQuantizer(exported.cfg, weights, calib, False)
    graph = quantizer.run()
    qnorm = next(op for op in graph['ops'] if op['out'] == 'l0.qn')
    knorm = next(op for op in graph['ops'] if op['out'] == 'l0.kn')
    assert (qnorm['heads'], knorm['heads'], qnorm['K'], knorm['K']) == (4, 2, 16, 16)
    assert graph['exps']['l0.qn'] == graph['exps']['l0.qr']
    assert graph['exps']['l0.kn'] == graph['exps']['l0.kr']
    # The calibration must observe individual head norms in raw projection values.
    captured = []
    hook = model.layers[0].self_attn.q_proj.register_forward_hook(lambda _, args, out: captured.append(out.detach()))
    with torch.no_grad():
        model(torch.tensor([list(range(16))]))
        model(torch.tensor([list(range(16, 32))]))
    hook.remove()
    expected = min(x.double().reshape(-1, 16).square().mean(-1).sqrt().min().item() for x in captured)
    assert calib['l0.q.rms_min'] == pytest.approx(expected, rel=1e-12)
    if scaled:
        for name in ('rope_cos', 'rope_sin'):
            entry = graph['tensors'][name]
            actual = np.frombuffer(quantizer.blob.bytes(), '<i2', count=32 * 8, offset=entry['offset']).reshape(32, 8)
            expected = llround_arr(weights[name + '_input'] * 16384).astype(np.int16)
            np.testing.assert_array_equal(actual, expected)
