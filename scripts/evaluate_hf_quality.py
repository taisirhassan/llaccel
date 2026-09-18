#!/usr/bin/env python3
"""Bounded teacher-forced integer-vs-HF quality diagnostic on supplied text.

Example (no downloads):
  uv run --frozen --extra hf python scripts/evaluate_hf_quality.py \
    --checkpoint checkpoints/hf/Qwen2.5-0.5B --qgraph build/hf-qwen/qgraph \
    --text /path/to/held-out.txt --out build/hf-qwen/held-out-quality.json

Defaults score 4 contiguous windows of 16 next-token predictions (64 targets).
Each window starts with empty KV state and uses only the reference text history.
This tests the integer golden model's quality, not RTL conformance. It imposes
no quality pass threshold and does not establish corpus/calibration disjointness.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch


def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def window_starts(total: int, windows: int, length: int, token_offset: int = 0, sampling: str = 'contiguous') -> list[int]:
    if type(windows) is not int or windows < 1 or type(length) is not int or length < 1:
        raise ValueError('windows and window-length must be positive integers')
    if type(token_offset) is not int or token_offset < 0:
        raise ValueError('token-offset must be a nonnegative integer')
    required = token_offset + windows * (length + 1)
    if total < required:
        raise ValueError(f'evaluation text needs {required} token IDs; got {total}')
    if sampling == 'contiguous':
        return [token_offset + i * (length + 1) for i in range(windows)]
    if sampling != 'uniform':
        raise ValueError('unknown window sampling')
    if windows == 1:
        return [token_offset]
    span = total - token_offset - (length + 1)
    return [token_offset + i * span // (windows - 1) for i in range(windows)]


def make_windows(ids: list[int], windows: int, length: int, token_offset: int = 0, sampling: str = 'contiguous') -> list[list[int]]:
    """Non-overlapping reference blocks: length inputs plus one final target."""
    return [ids[start:start + length + 1] for start in window_starts(len(ids), windows, length, token_offset, sampling)]


def row_metrics(integer_logits: np.ndarray, float_logits: np.ndarray, target: int) -> dict:
    """Compute stable cross entropy in nats on actual (unpadded) vocab entries."""
    a, b = np.asarray(integer_logits, dtype=np.float64), np.asarray(float_logits, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or not 0 <= target < len(a):
        raise ValueError('logit shapes or target are invalid')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite logits')
    def nll(row):
        peak = float(row.max())
        return float(peak + np.log(np.exp(row - peak).sum()) - row[target])
    norms = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {'target_id': target, 'integer_top1': int(a.argmax()), 'fp32_top1': int(b.argmax()),
        'top1_agreement': bool(a.argmax() == b.argmax()),
        'cosine': float(np.dot(a, b) / norms) if norms else 0.0,
        'integer_nll': nll(a), 'fp32_nll': nll(b)}


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('cannot aggregate zero scored tokens')
    result = {'scored_tokens': len(rows),
        'top1_agreement': float(np.mean([r['top1_agreement'] for r in rows])),
        'mean_cosine': float(np.mean([r['cosine'] for r in rows]))}
    for backend in ('integer', 'fp32'):
        ce = float(np.mean([r[backend + '_nll'] for r in rows]))
        result[backend + '_cross_entropy_nats'] = ce
        # JSON has no standard infinity value. Preserve CE if exp cannot fit.
        result[backend + '_perplexity'] = math.exp(ce) if ce < math.log(np.finfo(np.float64).max) else None
    result['cross_entropy_delta_nats'] = result['integer_cross_entropy_nats'] - result['fp32_cross_entropy_nats']
    return result


def teacher_forced_rows(golden, tokens, batch_rows=16):
    """Batch actual reference embeddings without padding or feeding argmaxes."""
    if not 1 <= batch_rows <= 16:
        raise ValueError('integer batch rows must be 1..16')
    for start in range(0, len(tokens), batch_rows):
        chunk = tokens[start:start + batch_rows]
        logits = golden.run_chunk(golden.embed_rows(chunk, len(chunk)), start)
        for offset, row in enumerate(logits):
            yield start + offset, row[:golden.vocab]


def allocate_references(path: Path, windows: int, length: int, vocab: int):
    """Back even multi-thousand-token evaluations by disk, not a Python list."""
    if any(type(n) is not int or n < 1 for n in (windows, length, vocab)):
        raise ValueError('reference dimensions must be positive integers')
    return np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                    shape=(windows, length, vocab))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--qgraph', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, help='default: checkpoint tokenizer')
    parser.add_argument('--text', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--windows', type=int, default=4)
    parser.add_argument('--window-sampling', choices=['contiguous', 'uniform'], default='contiguous', help='spread independent windows across the corpus or take consecutive blocks')
    parser.add_argument('--token-offset', type=int, default=0, help='first corpus token for an explicitly separate evaluation cohort')
    parser.add_argument('--window-length', type=int, default=16, help='scored next-token predictions per window')
    parser.add_argument('--min-scored-tokens', type=int, default=0,
                        help='increase windows to score at least this many reference targets (e.g. 4096)')
    parser.add_argument('--reference-batch-windows', type=int, default=1,
                        help='maximum fp32 windows retained on disk at once (default 1)')
    parser.add_argument('--reference-dir', type=Path, help='temporary disk-backed fp32 logits directory')
    parser.add_argument('--fast-matmul', action=argparse.BooleanOptionalAction, default=True,
                        help='exact-bound-checked FP64 BLAS accumulation (disable for I64 oracle)')
    parser.add_argument('--integer-batch-rows', type=int, default=16, help='teacher-forced golden rows per chunk, 1..16')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or args.min_scored_tokens < 0 or args.reference_batch_windows < 1:
        parser.error('threads/reference-batch-windows must be positive and min-scored-tokens nonnegative')
    if not 1 <= args.integer_batch_rows <= 16:
        parser.error('integer-batch-rows must be 1..16')
    evaluate(args)


def publish(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def evaluate(args):
    publish(args.out, {'status': 'RUNNING'})
    try:
        report = _evaluate(args)
        publish(args.out, {**report, 'status': 'PASS'})
    except BaseException as error:
        publish(args.out, {'status': 'FAILED', 'error': f'{type(error).__name__}: {error}'})
        raise
    print(json.dumps(report['summary'], indent=2), flush=True)
    print(f'Wrote {args.out}', flush=True)


def _evaluate(args):
    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM
    from llaccel.hf import tokenizer_at
    from llaccel.golden import GoldenModel
    from llaccel.hf_import import config_from_hf
    checkpoint, qdir = args.checkpoint.resolve(), args.qgraph.resolve()
    tokenizer_dir = (args.tokenizer or checkpoint).resolve()
    qgraph_file = qdir / 'qgraph.json'
    graph = json.loads(qgraph_file.read_text())
    config = graph['model']
    if not 1 <= args.window_length <= config['max_seq']:
        raise ValueError('window-length must fit the compiled context')
    hf_config = json.loads((checkpoint / 'config.json').read_text())
    expected = config_from_hf(hf_config, config['max_seq']).to_dict()
    for key in ('dim', 'n_layers', 'n_heads', 'n_kv_heads', 'head_dim', 'ffn', 'vocab'):
        if expected[key] != config[key]:
            raise ValueError(f'checkpoint and qgraph disagree on {key}')
    tokenizer = tokenizer_at(tokenizer_dir)
    text = args.text.read_text()
    # One special-token pass on the complete corpus. Window boundaries reset
    # context and positions but do not inject an extra synthetic BOS per block.
    ids = tokenizer.encode(text, add_special_tokens=True)
    if any(type(token) is not int or not 0 <= token < config['vocab'] for token in ids):
        raise ValueError('evaluation token IDs exceed the model vocabulary')
    window_count = max(args.windows, math.ceil(args.min_scored_tokens / args.window_length))
    starts = window_starts(len(ids), window_count, args.window_length, args.token_offset, args.window_sampling)
    windows = make_windows(ids, window_count, args.window_length, args.token_offset, args.window_sampling)
    if args.reference_dir:
        args.reference_dir.mkdir(parents=True, exist_ok=True)
    # Keep HF weights and the integer graph resident; retain at most one
    # configurable batch of reference logits on disk, rather than a corpus.
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation='eager').eval()
    golden = GoldenModel(qdir, fast_matmul=args.fast_matmul)
    scale = 2.0 ** golden.E_LOGIT
    records, all_rows = [], []
    for batch_start in range(0, len(windows), args.reference_batch_windows):
        batch = windows[batch_start:batch_start + args.reference_batch_windows]
        with tempfile.TemporaryDirectory(prefix='llaccel-fp32-', dir=args.reference_dir) as store:
            references = allocate_references(Path(store) / 'logits.npy',
                                             len(batch), args.window_length, config['vocab'])
            with torch.no_grad():
                for index, window in enumerate(batch):
                    logits = model(torch.tensor([window[:-1]], dtype=torch.long), use_cache=False).logits[0]
                    if tuple(logits.shape) != (args.window_length, config['vocab']):
                        raise ValueError('HF output shape differs from expected vocabulary/context')
                    references[index] = logits.float().cpu().numpy()
            del logits
            references.flush()
            for offset, (window, reference) in enumerate(zip(batch, references, strict=True)):
                index = batch_start + offset
                golden.reset()
                rows = []
                for pos, row in teacher_forced_rows(golden, window[:-1], args.integer_batch_rows):
                    logits = row.astype(np.float64) * scale
                    rows.append({'position': pos, **row_metrics(logits, reference[pos], window[pos + 1])})
                all_rows.extend(rows)
                records.append({'window': index, 'corpus_start_token': starts[index],
                                'token_ids': window, 'rows': rows, 'summary': aggregate(rows)})
                print(f'Window {index + 1}/{len(windows)}: {len(rows)} scored reference tokens', flush=True)
            del reference, references
    del model, golden
    gc.collect()
    checkpoint_files = [checkpoint / 'config.json']
    index = checkpoint / 'model.safetensors.index.json'
    if index.is_file():
        checkpoint_files.append(index)
        checkpoint_files += [checkpoint / name for name in sorted(set(json.loads(index.read_text())['weight_map'].values()))]
    else:
        checkpoint_files.append(checkpoint / 'model.safetensors')
    report = {'diagnostic': 'teacher-forced integer golden vs original HF; no quality pass threshold',
        'held_out_status': 'user-supplied text; calibration disjointness is not automatically established',
        'checkpoint': str(checkpoint), 'qgraph': str(qdir), 'text': str(args.text.resolve()),
        'hashes': {'text_sha256': file_hash(args.text), 'qgraph_sha256': file_hash(qgraph_file),
            'qgraph_weights_sha256': file_hash(qdir / graph['weights_file']),
            'script_sha256': file_hash(Path(__file__)),
            'checkpoint_files': {str(p.relative_to(checkpoint)): file_hash(p) for p in checkpoint_files},
            'tokenizer_files': {p.name: file_hash(p) for p in sorted(tokenizer_dir.glob('*.json'))}},
        'selection': 'non-overlapping windows from token_offset; positions/KV reset per window',
        'window_sampling': args.window_sampling,
        'window_start_tokens': starts,
        'token_offset': args.token_offset,
        'reference_storage': 'bounded-batch temporary float32 memmap; HF and integer weights resident together',
        'reference_storage_bytes': min(len(windows), args.reference_batch_windows) * args.window_length * config['vocab'] * 4,
        'reference_batch_windows': args.reference_batch_windows,
        'integer_batch_rows': args.integer_batch_rows,
        'integer_matmul': 'exact-bound-checked FP64 BLAS' if args.fast_matmul else 'I64 NumPy',
        'requested_min_scored_tokens': args.min_scored_tokens,
        'text_token_count': len(ids), 'window_count': len(windows),
        'predictions_per_window': args.window_length, 'context': config['max_seq'],
        'summary': aggregate(all_rows), 'windows': records}
    return report


if __name__ == '__main__':
    main()
