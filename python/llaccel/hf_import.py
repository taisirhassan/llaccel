"""Strict local Hugging Face Llama/Qwen2/Qwen3 safetensors checkpoint import.

``load_hf_checkpoint(directory, max_seq=256)`` returns an eval-mode CPU fp32
TinyLlama and provenance metadata. The explicit context cap is an accelerator
limit, not a claim to support the checkpoint's original context length. No
remote Python code, pickle, or network requests are used. Tokenization belongs
to the caller and must use this checkpoint's tokenizer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .model import ModelConfig, TinyLlama
from .rope import build_rope_tables, rope_config_from_hf


def _json(path: Path) -> dict:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out
    result = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object in {path.name}")
    return result


def _local(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("checkpoint filenames must be relative paths")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"checkpoint file missing or outside directory: {name}")
    return path


def config_from_hf(raw: dict, max_seq: int = 256) -> ModelConfig:
    """Validate the architecture contract before allocating model weights."""
    kind = raw.get("model_type")
    if kind not in ("llama", "qwen2", "qwen3"):
        raise ValueError("only dense model_type llama, qwen2 and qwen3 are supported")
    if type(max_seq) is not int or not 1 <= max_seq <= 4096:
        raise ValueError("max_seq must be an integer in [1, 4096]")
    original_context = raw.get("max_position_embeddings", 2048)
    if type(original_context) is not int or original_context < max_seq:
        raise ValueError("max_seq exceeds checkpoint max_position_embeddings")
    for key in ("attention_bias", "mlp_bias", "tie_word_embeddings", "use_sliding_window"):
        if key in raw and type(raw[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if raw.get("hidden_act", "silu") != "silu":
        raise ValueError("only silu activation is supported")
    if kind != "llama" and raw.get("mlp_bias", False):
        raise ValueError("Qwen MLP bias is unsupported")
    if kind == "qwen2" and raw.get("attention_bias", True) is False:
        raise ValueError("Qwen2 requires Q/K/V projection biases")
    rope_params = rope_config_from_hf(raw, kind)
    if raw.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("partial rotary embeddings are unsupported")
    if raw.get("use_sliding_window", False) or (kind == "llama" and raw.get("sliding_window") is not None):
        raise ValueError("sliding window attention is unsupported")
    layer_types = raw.get("layer_types")
    if layer_types is not None and (not isinstance(layer_types, list) or len(layer_types) != raw.get("num_hidden_layers") or any(x != "full_attention" for x in layer_types)):
        raise ValueError("only full_attention layers are supported")
    if raw.get("quantization_config") is not None:
        raise ValueError("prequantized checkpoints are unsupported; import floating point weights")
    if raw.get("pretraining_tp", 1) != 1:
        raise ValueError("pretraining_tp other than 1 is unsupported")
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "vocab_size"):
        if type(raw.get(key)) is not int or raw[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if raw.get("head_dim") is not None and (type(raw["head_dim"]) is not int or raw["head_dim"] <= 0):
        raise ValueError("head_dim must be a positive integer")
    heads = raw["num_attention_heads"]
    rope_base = (rope_params or {}).get("rope_theta", raw.get("rope_theta", 10000.0 if kind == "llama" else 1000000.0))
    return ModelConfig(dim=raw["hidden_size"], n_layers=raw["num_hidden_layers"],
                       n_heads=heads, n_kv_heads=heads if raw.get("num_key_value_heads") is None else raw["num_key_value_heads"],
                       head_dim=raw.get("head_dim") or (128 if kind == "qwen3" else raw["hidden_size"] // heads),
                       ffn=raw["intermediate_size"], vocab=raw["vocab_size"],
                       max_seq=max_seq, rope_base=rope_base,
                       rope_scaling=rope_params if rope_params["rope_type"] != "default" else None,
                       rms_eps=raw.get("rms_norm_eps", 1e-6),
                       qkv_bias=kind == "qwen2" or raw.get("attention_bias", False),
                       o_bias=kind != "qwen2" and raw.get("attention_bias", False),
                       mlp_bias=raw.get("mlp_bias", False), qk_norm=kind == "qwen3")


def load_hf_checkpoint(path: str | Path, max_seq: int = 256) -> tuple[TinyLlama, dict]:
    """Load single/sharded safe weights with exact names, shapes and finiteness.

    Tied embeddings may omit lm_head.weight. If both weights are present they
    must agree exactly. Unknown tensor names are errors, preventing silently
    ignoring unsupported architectural features.
    """
    root = Path(path).expanduser().resolve()
    raw = _json(_local(root, "config.json"))
    cfg = config_from_hf(raw, max_seq)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = _json(_local(root, index_path.name))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index requires a nonempty weight_map")
        if any(not isinstance(key, str) or not isinstance(name, str) for key, name in weight_map.items()):
            raise ValueError("safetensors weight_map must map tensor names to filenames")
        files = {name: _local(root, name) for name in weight_map.values()}
        if any(not name.endswith(".safetensors") for name in files):
            raise ValueError("all checkpoint shards must be safetensors")
    else:
        files = {"model.safetensors": _local(root, "model.safetensors")}
        weight_map = None
    # Meta construction avoids random initialization and a duplicate model copy.
    with torch.device("meta"):
        model = TinyLlama(cfg)
    expected = {("lm_head.weight" if key == "lm_head.weight" else "model." + key): key
                for key in model.state_dict()}
    shapes = {source: tuple(model.state_dict()[dest].shape) for source, dest in expected.items()}
    found = set()
    for name, filename in files.items():
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in found:
                    raise ValueError(f"duplicate tensor: {key}")
                if key not in expected:
                    raise ValueError(f"unsupported tensor: {key}")
                if weight_map is not None and weight_map.get(key) != name:
                    raise ValueError(f"index/shard mismatch: {key}")
                if tuple(handle.get_slice(key).get_shape()) != shapes[key]:
                    raise ValueError(f"wrong tensor shape: {key}; expected {shapes[key]}")
                found.add(key)
    if weight_map is not None and found != set(weight_map):
        raise ValueError("safetensors index references missing tensors")
    missing = set(expected) - found
    tied = raw.get("tie_word_embeddings", False)
    if missing - ({"lm_head.weight"} if tied else set()):
        raise ValueError(f"missing tensors: {sorted(missing)}")
    model.to_empty(device="cpu")
    destination = model.state_dict()
    digests = {}
    with torch.no_grad():
        for name, filename in files.items():
            with filename.open("rb") as stream:
                digests[name] = hashlib.file_digest(stream, "sha256").hexdigest()
            with safe_open(filename, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    tensor = handle.get_tensor(key)
                    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                        raise ValueError(f"tensor must be floating point: {key}")
                    tensor = tensor.to(torch.float32)
                    if not torch.isfinite(tensor).all().item():
                        raise ValueError(f"nonfinite or fp32-overflowing tensor: {key}")
                    destination[expected[key]].copy_(tensor)
        if tied:
            if "lm_head.weight" in found and not torch.equal(model.lm_head.weight, model.embed_tokens.weight):
                raise ValueError("tie_word_embeddings is true but embedding and output weights differ")
            model.lm_head.weight = model.embed_tokens.weight
        cos, sin = build_rope_tables(cfg)
        model.rope_cos.copy_(cos)
        model.rope_sin.copy_(sin)
    metadata = {"source": str(root), "model_type": raw["model_type"],
                "config": cfg.to_dict(), "hf_config": raw,
                "checkpoint_files": digests,
                "context_reduced_from": raw.get("max_position_embeddings", 2048)}
    return model.eval(), metadata
