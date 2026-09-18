#!/usr/bin/env python3
"""Multi-prompt, long decode, and KV-tile/context boundary RTL conformance.

Consumes an existing compiled pretrained image and graph; never downloads or
recompiles. Boundary IDs are deterministic prefixes of repeated real BPE text.
This is exact integer conformance, separate from held-out language quality.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

from regress_pretrained import run, sha256, write_json


def make_cases(tokenizer, prompts, context, tokens, long_tokens, boundaries=None):
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p for p in prompts):
        raise ValueError('prompts must be a nonempty JSON array of nonempty strings')
    if tokens < 1 or long_tokens < 1:
        raise ValueError('decode counts must be positive')
    cases = []
    for index, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        if not ids or len(ids) + tokens > context:
            raise ValueError(f'prompt {index} and decode count do not fit context {context}')
        cases.append({'name': f'prompt-{index}', 'kind': 'natural prompt', 'text': prompt,
                      'ids': ids, 'tokens': tokens})
    short = cases[0]['ids']
    if len(short) + long_tokens > context:
        raise ValueError('long decode and first prompt do not fit the compiled context')
    cases.append({'name': 'long-decode', 'kind': 'long autoregressive decode',
                  'ids': short, 'tokens': long_tokens})
    # Test both sides of 256-row KV tiles, plus the complete compiled context.
    lengths = boundaries if boundaries is not None else sorted({1, context - 1, context} |
        {length for edge in range(256, context + 1, 256)
         for length in (edge - 1, edge, edge + 1) if 1 <= length <= context})
    source = tokenizer.encode('\n'.join(prompts), add_special_tokens=False)
    if not source:
        raise ValueError('boundary source text encodes no tokens')
    for length in lengths:
        if not 1 <= length <= context:
            raise ValueError(f'boundary length {length} exceeds context {context}')
        ids = (source * ((length + len(source) - 1) // len(source)))[:length]
        cases.append({'name': f'boundary-{length}', 'kind': 'repeated BPE source prefix',
                      'ids': ids, 'tokens': min(tokens, context - length)})
    return cases


def select_cases(cases, names):
    if names is None:
        return cases
    unknown = set(names) - {case['name'] for case in cases}
    if not names or unknown:
        raise ValueError(f'empty or unknown requested cases: {sorted(unknown)}')
    return [case for case in cases if case['name'] in names]


def finish_traces(root, trace_hashes, generated, keep):
    """Delete only successfully verified trace files; preserve logs and inputs."""
    write_json(root / 'verified-traces.json', {'result': 'MATCH', 'sha256': trace_hashes,
                                              'retained': keep, 'generated': generated})
    if not keep:
        for name in trace_hashes:
            if name not in {'golden.json', 'func.json', 'rtl.json'}:
                raise ValueError(f'unexpected trace filename: {name}')
        for name in trace_hashes:
            (root / name).unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--qgraph', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--simulator', type=Path, default=Path('build/dma32/llaccel-sim'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prompts-file', type=Path, help='JSON array of natural-language strings')
    parser.add_argument('--tokens', type=int, default=16)
    parser.add_argument('--long-decode-tokens', type=int, default=256)
    parser.add_argument('--boundary-lengths', nargs='+', type=int)
    parser.add_argument('--skip-boundaries', action='store_true', help='run natural prompts and long decode only; boundary validation must be reported separately')
    parser.add_argument('--backends', nargs='+', choices=['func', 'rtl'], default=['func', 'rtl'])
    parser.add_argument('--case-names', nargs='+', help='explicit case subset for separate parallel invocations; reports name the subset')
    parser.add_argument('--keep-traces', action='store_true', help='retain full golden/backend logits after successful verification')
    parser.add_argument('--max-logit-values', type=int, default=50_000_000,
                        help='per-case full-logit budget; long prompts should use prefill M>=4')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    import torch
    from llaccel.hf import tokenizer_at
    from llaccel.golden import GoldenModel
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {'status': 'RUNNING', 'cases': [], 'backends': args.backends,
                'image_sha256': sha256(args.image), 'simulator_sha256': sha256(args.simulator),
                'qgraph_sha256': sha256(args.qgraph / 'qgraph.json'),
                'qgraph_files_sha256': {str(p.relative_to(args.qgraph)): sha256(p) for p in sorted(args.qgraph.rglob('*')) if p.is_file()},
                'tokenizer_files_sha256': {str(p.relative_to(args.tokenizer)): sha256(p) for p in sorted(args.tokenizer.rglob('*')) if p.is_file()},
                'script_sha256': sha256(Path(__file__))}
    report = args.out / 'results.json'
    write_json(report, manifest)
    try:
        prompts = json.loads(args.prompts_file.read_text()) if args.prompts_file else [
            'Hello', 'The capital of France is', 'Write a Python function that adds two numbers.',
            'Explain why the sky appears blue.']
        config = json.loads((args.qgraph / 'qgraph.json').read_text())['model']
        cases = make_cases(tokenizer_at(args.tokenizer), prompts, config['max_seq'],
                           args.tokens, args.long_decode_tokens, [] if args.skip_boundaries else args.boundary_lengths)
        cases = select_cases(cases, args.case_names)
        manifest['requested_cases'] = [case['name'] for case in cases]
        for case in cases:
            launches = math.ceil(len(case['ids']) / config.get('prefill_m', 16)) + case['tokens']
            values = launches * config['vocab']
            if values > args.max_logit_values:
                raise ValueError(f'{case["name"]}: estimated {values} full-logit values exceeds budget; compile with larger prefill-m (16 recommended, 4 if SRAM limited) or explicitly raise --max-logit-values')
        manifest['prefill_m'] = config.get('prefill_m', 16)
        manifest['keep_traces'] = args.keep_traces
        manifest['boundaries_skipped'] = args.skip_boundaries
        manifest['golden_matmul'] = 'exact-bounded-fp64'
        for case in cases:
            root = args.out / case['name']
            root.mkdir(exist_ok=True)
            ids_path, golden_path = root / 'prompt-ids.json', root / 'golden.json'
            write_json(ids_path, case['ids'])
            golden = GoldenModel(args.qgraph, fast_matmul=True)
            expected = golden.generate(case['ids'], case['tokens'])
            # Backend comparisons only need the materialized trace. Release
            # mapped model/state storage before the long simulator subprocess.
            del golden
            write_json(golden_path, expected)
            trace_hashes = {'golden.json': sha256(golden_path)}
            for backend in dict.fromkeys(args.backends):
                output = root / f'{backend}.json'
                command = [args.simulator.resolve(), args.image.resolve(), '--backend', backend,
                           '--prompt-ids', ids_path.resolve(), '--tokens', case['tokens'],
                           '--verify', golden_path.resolve(), '--out', output.resolve(), '--quiet']
                if backend == 'func':
                    command += ['--interleave', 7]
                run(command, root / f'{backend}.log')
                actual = json.loads(output.read_text())
                for field in ['prompt_tokens', 'generated', 'argmax_per_step', 'logits_last_rows']:
                    if actual[field] != expected[field]:
                        raise ValueError(f'{case["name"]}/{backend} differs in {field}')
                actual_steps = [(step['pos'], step['M']) for step in actual['steps']]
                expected_steps = [(step['pos'], step['rows']) for step in expected['steps']]
                if actual_steps != expected_steps:
                    raise ValueError(f'{case["name"]}/{backend} differs in launch metadata')
                if 'VERIFY: MATCH' not in (root / f'{backend}.log').read_text():
                    raise ValueError('missing positive runtime verification')
                trace_hashes[f'{backend}.json'] = sha256(output)
                manifest['cases'].append({'name': case['name'], 'kind': case['kind'],
                    'backend': backend, 'prompt_length': len(case['ids']),
                    'generated_tokens': len(actual['generated']), 'context': config['max_seq'],
                    'result': 'MATCH', 'launches': len(actual['steps']), 'perf': actual['perf_total'],
                    'result_sha256': trace_hashes[f'{backend}.json'], 'golden_sha256': trace_hashes['golden.json']})
                write_json(report, manifest)
            finish_traces(root, trace_hashes, expected['generated'], args.keep_traces)
            print(f'PASS {case["name"]}: {len(case["ids"])} prompt + {case["tokens"]} generated', flush=True)
        manifest['status'] = 'PASS'
        manifest['rtl_verified'] = 'rtl' in args.backends
        write_json(report, manifest)
    except BaseException as error:
        write_json(report, {**manifest, 'status': 'FAILED', 'error': f'{type(error).__name__}: {error}'})
        raise
    print(f'Completed {len(manifest["cases"])} backend cases: {report}', flush=True)


if __name__ == '__main__':
    main()
