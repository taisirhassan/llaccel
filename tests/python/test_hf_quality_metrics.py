"""Small numerical/selection checks for the bounded HF quality diagnostic."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('evaluate_hf_quality',
    Path(__file__).resolve().parents[2] / 'scripts/evaluate_hf_quality.py')
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def test_reference_windows_do_not_overlap():
    assert quality.make_windows(list(range(12)), 3, 3) == [list(range(4)), list(range(4, 8)), list(range(8, 12))]
    with pytest.raises(ValueError, match='needs 12'):
        quality.make_windows(list(range(11)), 3, 3)


def test_stable_cross_entropy_and_perplexity():
    row = quality.row_metrics(np.array([10000., 10000.]), np.array([0., 0.]), 1)
    assert row['integer_nll'] == pytest.approx(np.log(2))
    assert row['fp32_nll'] == pytest.approx(np.log(2))
    summary = quality.aggregate([row, row])
    assert summary['integer_perplexity'] == pytest.approx(2)
    assert summary['fp32_perplexity'] == pytest.approx(2)
    assert summary['scored_tokens'] == 2


def test_comparison_counts_actual_target_not_top_one():
    row = quality.row_metrics(np.array([0., 2.]), np.array([2., 0.]), 0)
    assert row['top1_agreement'] is False
    assert row['integer_nll'] - row['fp32_nll'] == pytest.approx(2)


def test_nonfinite_rejected_and_perplexity_overflow_explicit():
    with pytest.raises(ValueError, match='nonfinite'):
        quality.row_metrics(np.array([np.nan]), np.array([0.]), 0)
    row = quality.row_metrics(np.array([-1000., 0.]), np.array([-1000., 0.]), 0)
    result = quality.aggregate([row])
    assert result['integer_perplexity'] is None
    assert result['integer_cross_entropy_nats'] == pytest.approx(1000)


def test_thousands_of_reference_tokens_use_disk_memmap(tmp_path):
    path = tmp_path / 'refs.npy'
    references = quality.allocate_references(path, 128, 32, 17)
    assert isinstance(references, np.memmap)
    assert references.shape == (128, 32, 17)  # 4096 scored tokens
    references[127, 31] = np.arange(17)
    references.flush()
    del references
    reopened = np.load(path, mmap_mode='r')
    assert isinstance(reopened, np.memmap)
    np.testing.assert_array_equal(reopened[127, 31], np.arange(17))
    assert path.stat().st_size >= 4096 * 17 * 4


def test_quality_failure_replaces_stale_summary(tmp_path, monkeypatch):
    import argparse
    import json
    args = argparse.Namespace(out=tmp_path / 'quality.json')
    args.out.write_text('{"status":"PASS", "summary":{"scored_tokens":4096}}')
    def fail(_):
        assert json.loads(args.out.read_text()) == {'status': 'RUNNING'}
        raise RuntimeError('reference failure')
    monkeypatch.setattr(quality, '_evaluate', fail)
    with pytest.raises(RuntimeError, match='reference failure'):
        quality.evaluate(args)
    report = json.loads(args.out.read_text())
    assert report['status'] == 'FAILED'
    assert 'summary' not in report


def test_teacher_forced_chunks_preserve_order_and_partial_tail():
    class Golden:
        vocab = 2
        def __init__(self):
            self.history = []
            self.launches = []
        def embed_rows(self, tokens, count):
            assert count == len(tokens)
            return np.asarray(tokens)
        def run_chunk(self, tokens, pos):
            assert pos == len(self.history)
            self.launches.append((pos, len(tokens)))
            rows = []
            for token in tokens:
                self.history.append(token)
                rows.append([sum(self.history), token, 999])
            return np.asarray(rows)
    golden = Golden()
    rows = list(quality.teacher_forced_rows(golden, list(range(35)), 16))
    assert golden.launches == [(0,16), (16,16), (32,3)]
    assert [pos for pos, _ in rows] == list(range(35))
    assert rows[-1][1].tolist() == [sum(range(35)),34]
    assert all(row.shape == (2,) for _, row in rows)


def test_explicit_untouched_cohort_offset():
    assert quality.make_windows(list(range(20)), 2, 3, 8) == [list(range(8,12)), list(range(12,16))]
    with pytest.raises(ValueError, match='needs 21'):
        quality.make_windows(list(range(20)), 2, 3, 13)
    with pytest.raises(ValueError, match='nonnegative'):
        quality.make_windows(list(range(20)), 1, 3, -1)


def test_uniform_windows_disperse_without_overlap_or_offset_leak():
    starts = quality.window_starts(1000, 32, 16, 100, 'uniform')
    assert starts[0] == 100 and starts[-1] == 983
    assert all(b-a >= 17 for a,b in zip(starts,starts[1:]))
    assert quality.make_windows(list(range(100)), 3, 3, 20, 'uniform') == [list(range(20,24)),list(range(58,62)),list(range(96,100))]
    with pytest.raises(ValueError, match='needs'):
        quality.window_starts(100, 10, 16, 0, 'uniform')
