"""A repeated checkpoint run must never expose a stale successful report."""
import argparse
import json

import pytest

from llaccel import hf


@pytest.fixture
def run_args(tmp_path):
    args = argparse.Namespace(out=tmp_path / 'run', prompt='Hello', backends=['func', 'rtl'],
                              tokenizer=tmp_path / 'tokenizer')
    args.out.mkdir()
    (args.out / 'report.json').write_text(json.dumps({
        'status': 'PASS', 'exact_integer_verification': 'MATCH',
        'fp32_teacher_forced': {'mean_cosine': 0.1}}))
    return args


def read_report(args):
    return json.loads((args.out / 'report.json').read_text())


@pytest.mark.parametrize('error', [RuntimeError('simulation failed'), KeyboardInterrupt()])
def test_failed_run_replaces_old_match(run_args, monkeypatch, error):
    def fail(args):
        current = read_report(args)
        assert current['status'] == 'RUNNING'
        assert 'exact_integer_verification' not in current
        assert 'fp32_teacher_forced' not in current
        raise error
    monkeypatch.setattr(hf, '_run_checkpoint', fail)
    with pytest.raises(type(error)):
        hf.run_checkpoint(run_args)
    current = read_report(run_args)
    assert current['status'] == 'FAILED'
    assert current['error'].startswith(type(error).__name__)
    assert 'exact_integer_verification' not in current
    assert 'fp32_teacher_forced' not in current
    assert not (run_args.out / 'report.json.tmp').exists()


def test_tokenizer_failure_invalidates_old_report(run_args, monkeypatch):
    def fail(_):
        assert read_report(run_args)['status'] == 'RUNNING'
        raise ValueError('invalid tokenizer')
    monkeypatch.setattr(hf, 'tokenizer_at', fail)
    with pytest.raises(ValueError, match='invalid tokenizer'):
        hf.run_checkpoint(run_args)
    assert read_report(run_args)['status'] == 'FAILED'
    assert 'exact_integer_verification' not in read_report(run_args)


def test_success_retains_complete_report_schema(run_args, monkeypatch):
    fresh = {'exact_integer_verification': 'MATCH', 'backends': ['func', 'rtl'],
             'fp32_teacher_forced': {'mean_cosine': 0.98},
             'performance': {'rtl': {'total_cycles': 42}}, 'generated': [1, 2]}
    def succeed(args):
        assert read_report(args)['status'] == 'RUNNING'
        return fresh
    monkeypatch.setattr(hf, '_run_checkpoint', succeed)
    hf.run_checkpoint(run_args)
    assert read_report(run_args) == {**fresh, 'status': 'PASS'}
    assert not (run_args.out / 'report.json.tmp').exists()
