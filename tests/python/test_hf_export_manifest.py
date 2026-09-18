from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from llaccel import hf, hf_import, calibrate
from llaccel.model import TinyLlama, ModelConfig


@pytest.fixture
def export_args(tmp_path, monkeypatch):
    torch.manual_seed(30)
    model = TinyLlama(ModelConfig(dim=16, n_layers=1, n_heads=1, n_kv_heads=1,
        head_dim=16, ffn=32, vocab=32, max_seq=16)).eval()
    monkeypatch.setattr(hf_import, 'load_hf_checkpoint', lambda *_a, **_k:
        (model, {'context_reduced_from': 16, 'model_type': 'llama'}))
    class Tokenizer:
        def get_vocab(self): return {'a': 1, 'b': 2}
        def encode(self, *_a, **_k): return [1, 2, 1, 2]
        def save_pretrained(self, directory):
            Path(directory).mkdir(parents=True)
            (Path(directory)/'tokenizer.json').write_text('{}')
    monkeypatch.setattr(hf, 'tokenizer_at', lambda _p: Tokenizer())
    corpus = tmp_path/'calibration.txt'
    corpus.write_text('a b a b\n')
    return SimpleNamespace(checkpoint=tmp_path, out=tmp_path/'export', context=16,
        calibration=corpus, calibration_sequences=1, calibration_length=4,
        smoothquant_alpha=None, smoothquant_auto_alpha=False,
        optimize_quantization=False, skip_float_check=True)


def test_manifest_is_published_with_hashes(export_args):
    hf.export_checkpoint(export_args)
    manifest=json.loads((export_args.out/'hf-import.json').read_text())
    assert manifest['calibration_length'] == 4
    for name, digest in manifest['export_files_sha256'].items():
        assert hf.file_sha256(export_args.out/name) == digest


def test_partial_overwrite_cannot_retain_success_manifest(export_args, monkeypatch):
    export_args.out.mkdir()
    manifest=export_args.out/'hf-import.json'
    manifest.write_text('{"old_success":true}')
    def fail(*_args, **_kwargs): raise RuntimeError('calibration failed')
    monkeypatch.setattr(calibrate, 'calibrate_tokens', fail)
    with pytest.raises(RuntimeError, match='calibration failed'):
        hf.export_checkpoint(export_args)
    assert not manifest.exists()


def test_uniform_calibration_windows_cover_entire_corpus():
    class Tokenizer:
        def encode(self, text, **kwargs): return list(range(100))
    sequences, metadata = hf.calibration_sequences(Tokenizer(), 'text', 3, 10, 'uniform-windows')
    assert sequences == [list(range(10)), list(range(45,55)), list(range(90,100))]
    assert metadata['window_starts'] == [0,45,90]
    assert metadata['sampled_tokens'] == 30
    assert metadata['token_ids_sha256'] == hf.calibration_sequences(Tokenizer(), 'text', 3, 10, 'uniform-windows')[1]['token_ids_sha256']
    with pytest.raises(ValueError, match='shorter'):
        hf.calibration_sequences(Tokenizer(), 'text', 3, 101, 'uniform-windows')
