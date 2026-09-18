"""Checkpoint contract tests, including independent HF float implementations."""
import json

import pytest
import torch
from safetensors.torch import save_file

from llaccel.hf_import import config_from_hf, load_hf_checkpoint
from llaccel.model import TinyLlama


def config(kind="llama"):
    return dict(model_type=kind, hidden_size=32, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1,
                intermediate_size=64, vocab_size=37, max_position_embeddings=256,
                rms_norm_eps=1e-6, rope_theta=10000.0, hidden_act="silu")


def checkpoint(tmp_path, kind="llama", sharded=False, tied=False):
    raw = config(kind)
    raw["tie_word_embeddings"] = tied
    model = TinyLlama(config_from_hf(raw, 32)).eval()
    if tied:
        model.lm_head.weight = model.embed_tokens.weight
    weights = {("lm_head.weight" if k == "lm_head.weight" else "model." + k): v.detach().clone()
               for k, v in model.state_dict().items() if not (tied and k == "lm_head.weight")}
    write(tmp_path, raw, weights, sharded)
    return model, raw, weights


def write(path, raw, weights, sharded=False):
    (path / "config.json").write_text(json.dumps(raw))
    if sharded:
        names = list(weights)
        split = len(names) // 2
        weight_map = {}
        for i, keys in enumerate((names[:split], names[split:])):
            name = f"model-{i}.safetensors"
            save_file({k: weights[k] for k in keys}, path / name)
            weight_map.update({k: name for k in keys})
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_file(weights, path / "model.safetensors")


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen3"])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_exact_mapping(tmp_path, kind, sharded, tied):
    original, _, _ = checkpoint(tmp_path, kind, sharded, tied)
    imported, metadata = load_hf_checkpoint(tmp_path, 32)
    tokens = torch.tensor([[0, 3, 6, 10, 36]])
    with torch.no_grad():
        torch.testing.assert_close(imported(tokens), original(tokens), rtol=0, atol=0)
    assert imported.training is False
    assert metadata["model_type"] == kind
    assert len(metadata["checkpoint_files"]) == (2 if sharded else 1)
    if tied:
        assert imported.lm_head.weight is imported.embed_tokens.weight


@pytest.mark.parametrize("change", [
    {"model_type": "qwen3_moe"}, {"rope_scaling": {"rope_type": "dynamic", "factor": 2}},
    {"use_sliding_window": True}, {"hidden_act": "gelu"},
    {"head_dim": 15}, {"quantization_config": {}},
    {"pretraining_tp": 2}, {"partial_rotary_factor": 0.5}, {"tie_word_embeddings": "false"},
    {"layer_types": ["full_attention", "sliding_attention"]},
])
def test_reject_unsupported_config(change):
    with pytest.raises(ValueError):
        config_from_hf(config() | change, 32)


@pytest.mark.parametrize("context", [0, 257, True, 1.5])
def test_reject_context(context):
    with pytest.raises(ValueError, match="max_seq"):
        config_from_hf(config(), context)


@pytest.mark.parametrize("problem", ["missing", "unknown", "shape", "nonfinite", "integer"])
def test_corrupt_weights(tmp_path, problem):
    _, raw, weights = checkpoint(tmp_path)
    key = "model.embed_tokens.weight"
    if problem == "missing":
        weights.pop(key)
    elif problem == "unknown":
        weights["model.extra.weight"] = torch.ones(1)
    elif problem == "shape":
        weights[key] = torch.ones(1)
    elif problem == "nonfinite":
        weights[key][0, 0] = float("nan")
    else:
        weights[key] = weights[key].to(torch.int32)
    write(tmp_path, raw, weights)
    with pytest.raises(ValueError):
        load_hf_checkpoint(tmp_path, 32)


def test_tied_disagreement(tmp_path):
    _, raw, weights = checkpoint(tmp_path, tied=True)
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"] + 1
    write(tmp_path, raw, weights)
    with pytest.raises(ValueError, match="weights differ"):
        load_hf_checkpoint(tmp_path, 32)


@pytest.mark.parametrize("filename", ["../outside.safetensors", "/tmp/outside.safetensors"])
def test_reject_shard_escape(tmp_path, filename):
    checkpoint(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": filename}}))
    with pytest.raises(ValueError):
        load_hf_checkpoint(tmp_path, 32)


def test_index_mismatch(tmp_path):
    checkpoint(tmp_path, sharded=True)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["model.missing.weight"] = "model-0.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="missing tensors"):
        load_hf_checkpoint(tmp_path, 32)


@pytest.mark.parametrize("kind,changes", [
    ("llama", {}), ("qwen2", {}), ("qwen3", {"head_dim": 16}), ("qwen3", {}),
    ("llama", {"head_dim": 32}),
    ("llama", {"attention_bias": True, "mlp_bias": True}),
    ("qwen3", {"head_dim": 32, "attention_bias": True}),
    ("llama", {"rope_scaling": {"rope_type": "linear", "factor": 4.0}}),
    ("qwen2", {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}),
    ("qwen3", {"head_dim": 32, "rope_scaling": {"rope_type": "yarn", "factor": 4.0,
        "beta_fast": 16.0, "beta_slow": 2.0, "truncate": False, "attention_factor": 1.2,
        "original_max_position_embeddings": 64}}),
    ("llama", {"rope_scaling": {"rope_type": "llama3", "factor": 8.0,
        "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 64}}),
])
def test_transformers_independent_logits(tmp_path, kind, changes):
    transformers = pytest.importorskip("transformers")
    raw = config(kind) | changes
    family = {"llama": "Llama", "qwen2": "Qwen2", "qwen3": "Qwen3"}[kind]
    cls_config = getattr(transformers, family + "Config")
    cls_model = getattr(transformers, family + "ForCausalLM")
    hf_config = cls_config(**{k: v for k, v in raw.items() if k != "model_type"})
    hf_config._attn_implementation = "eager"
    torch.manual_seed(9)
    original = cls_model(hf_config).float().eval()
    with torch.no_grad():
        for name, parameter in original.named_parameters():
            if name.endswith(".bias"):
                parameter.uniform_(-0.03, 0.03)
            elif name.endswith(("q_norm.weight", "k_norm.weight")):
                parameter.uniform_(0.5, 1.5)
    original.save_pretrained(tmp_path, safe_serialization=True)
    imported, _ = load_hf_checkpoint(tmp_path, 32)
    for length in (1, 7, 16, 32):
        tokens = torch.arange(length).unsqueeze(0) % raw["vocab_size"]
        with torch.no_grad():
            actual = imported(tokens)
            expected = original(tokens).logits
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_bfloat16_checkpoint(tmp_path):
    _, raw, weights = checkpoint(tmp_path, kind="qwen2")
    weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}
    write(tmp_path, raw, weights)
    model, _ = load_hf_checkpoint(tmp_path, 32)
    assert model.embed_tokens.weight.dtype == torch.float32
    torch.testing.assert_close(model.embed_tokens.weight, weights["model.embed_tokens.weight"].float(), rtol=0, atol=0)


def test_duplicate_json_keys(tmp_path):
    checkpoint(tmp_path)
    (tmp_path / "config.json").write_text('{"model_type":"llama","model_type":"qwen2"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_hf_checkpoint(tmp_path, 32)


def test_symlink_escape(tmp_path):
    checkpoint(tmp_path)
    (tmp_path / "outside.safetensors").symlink_to("/etc/passwd")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "outside.safetensors"}}))
    with pytest.raises(ValueError, match="outside directory"):
        load_hf_checkpoint(tmp_path, 32)


@pytest.mark.parametrize("rope", [
    {"rope_type": "linear", "factor": 0},
    {"rope_type": "linear", "factor": True},
    {"rope_type": "linear", "factor": float("nan")},
    {"rope_type": "llama3", "factor": 8.0},
    {"rope_type": "default", "unexpected": 1},
    {"rope_type": "yarn", "factor": 4, "attention_factor": 2.1},
    "linear",
])
def test_reject_malformed_rope(rope):
    with pytest.raises(ValueError):
        config_from_hf(config() | {"rope_scaling": rope}, 32)


@pytest.mark.parametrize("kind", ["qwen", "qwen3_moe", "qwen3_next", "qwen2_vl", "llama4", "mamba"])
def test_reject_other_architectures(kind):
    with pytest.raises(ValueError, match="only dense"):
        config_from_hf(config(kind), 32)
