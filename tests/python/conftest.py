from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def tiny_export(tmp_path_factory):
    """A complete export directory (model.mlir, weights.*, calib.json, tokenizer.json) of a tiny
    2-layer GQA (n_kv_heads=1) model with QKV bias, calibrated on a synthetic 16-symbol corpus.
    Returns (export_dir, model, cfg, itos). Deterministic (seeded)."""
    import torch

    from llaccel.export import export_dir
    from llaccel.model import ModelConfig, TinyLlama

    cfg = ModelConfig(dim=32, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=16, ffn=64, vocab=16, max_seq=64, qkv_bias=True)
    torch.manual_seed(5)
    model = TinyLlama(cfg).eval()
    with torch.no_grad():  # give the untrained model some structure
        for p in model.parameters():
            p.mul_(4.0)
    itos = [chr(97 + i) for i in range(16)]
    base = tmp_path_factory.mktemp("tiny")
    data = base / "data"
    data.mkdir()
    rng = np.random.default_rng(0)
    (data / "input.txt").write_text("".join(itos[i] for i in rng.integers(0, 16, 20000)))
    ex = base / "export"
    export_dir(model, ex, cfg, {"itos": itos}, calib_seqs=8, calib_len=48, calib_data=data)
    return ex, model, cfg, itos
