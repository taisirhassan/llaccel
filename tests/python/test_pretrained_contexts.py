"""Deterministic tile/context case construction without model downloads."""
import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('pretrained_contexts', scripts / 'regress_pretrained_contexts.py')
    contexts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contexts)
finally:
    sys.path.pop(0)


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return ([0] if add_special_tokens else []) + [ord(c) for c in text]


def test_default_cases_cross_every_kv_tile_boundary():
    cases = contexts.make_cases(Tokenizer(), ['Hi', 'Hello'], 1024, 16, 256)
    boundaries = {len(c['ids']): c for c in cases if c['name'].startswith('boundary-')}
    assert set(boundaries) == {1,255,256,257,511,512,513,767,768,769,1023,1024}
    assert boundaries[1024]['tokens'] == 0
    assert boundaries[1023]['tokens'] == 1
    assert next(c for c in cases if c['name'] == 'long-decode')['tokens'] == 256
    assert all(len(c['ids']) + c['tokens'] <= 1024 for c in cases)


def test_invalid_case_lengths_fail_instead_of_silent_skipping():
    with pytest.raises(ValueError, match='long decode'):
        contexts.make_cases(Tokenizer(), ['Hi'], 32, 2, 256)
    with pytest.raises(ValueError, match='boundary length'):
        contexts.make_cases(Tokenizer(), ['Hi'], 32, 2, 8, [33])
    with pytest.raises(ValueError, match='prompt 0'):
        contexts.make_cases(Tokenizer(), ['too long'], 8, 2, 2)


def test_small_context_smoke_cases_are_explicit():
    cases = contexts.make_cases(Tokenizer(), ['Hi'], 32, 2, 8, [1,15,16,17,31,32])
    assert len(cases) == 8
    assert cases[-1]['tokens'] == 0


def test_successful_trace_cleanup_preserves_hashes_logs_and_inputs(tmp_path):
    import json
    for name in ['golden.json','func.json','rtl.json','prompt-ids.json','rtl.log']:
        (tmp_path / name).write_text('fixture')
    hashes = {name:'hash-'+name for name in ['golden.json','func.json','rtl.json']}
    contexts.finish_traces(tmp_path, hashes, [1,2], False)
    assert all(not (tmp_path/name).exists() for name in hashes)
    assert (tmp_path/'prompt-ids.json').exists() and (tmp_path/'rtl.log').exists()
    report=json.loads((tmp_path/'verified-traces.json').read_text())
    assert report['sha256']==hashes and report['generated']==[1,2]
    assert report['retained'] is False


def test_keep_traces_is_explicit(tmp_path):
    (tmp_path/'golden.json').write_text('fixture')
    contexts.finish_traces(tmp_path, {'golden.json':'hash'}, [1], True)
    assert (tmp_path/'golden.json').read_text()=='fixture'


def test_prompt_file_must_be_array_and_boundaries_can_be_separate():
    with pytest.raises(ValueError, match='JSON array'):
        contexts.make_cases(Tokenizer(), 'Hello', 1024, 16, 64)
    cases = contexts.make_cases(Tokenizer(), ['Hello'], 1024, 16, 64, [])
    assert [case['name'] for case in cases] == ['prompt-0', 'long-decode']


def test_named_subsets_cannot_silently_pass_zero_cases():
    cases = contexts.make_cases(Tokenizer(), ['Hi'], 1024, 16, 64, [])
    assert [case['name'] for case in contexts.select_cases(cases, ['long-decode'])] == ['long-decode']
    with pytest.raises(ValueError, match='unknown requested'):
        contexts.select_cases(cases, ['misspelled'])
    with pytest.raises(ValueError, match='empty'):
        contexts.select_cases(cases, [])
