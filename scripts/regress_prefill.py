#!/usr/bin/env python3
"""Check configurable prefill shapes through compiler, golden, funcsim and RTL.

Run with uv run python scripts/regress_prefill.py --build build/dma32.
Uses a seeded generated transformer; no checkpoint downloads are required.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import torch
from llaccel.export import export_dir
from llaccel.golden import GoldenModel
from llaccel.model import ModelConfig, TinyLlama


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', type=Path, default=Path('build/cmake'))
    ap.add_argument('--out', type=Path, default=Path('build/prefill-regressions'))
    args = ap.parse_args()
    compiler = (args.build / 'compiler/bin/llaccel-compile').resolve()
    simulator = (args.build / 'llaccel-sim').resolve()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(731)
    torch.set_num_threads(1)
    cfg = ModelConfig(dim=32, n_layers=1, n_heads=2, n_kv_heads=1,
                      head_dim=16, ffn=48, vocab=17, max_seq=32, qkv_bias=True)
    tokenizer = {'itos': list('abcdefghijklmnopq')}
    corpus = root / 'calibration'
    corpus.mkdir(exist_ok=True)
    (corpus / 'input.txt').write_text('abcdefghijklmnopq' * 100)
    exported = root / 'export'
    export_dir(TinyLlama(cfg).eval(), exported, cfg, tokenizer,
               calib_seqs=8, calib_len=32, calib_data=corpus)
    shutil.copyfile(exported / 'tokenizer.json', root / 'tokenizer.json')

    def run(command, log):
        result = subprocess.run(list(map(str, command)), text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f'command failed: {command}\n{result.stdout}\nlog: {log}')
        return result.stdout

    records = []
    for rows in (1, 4):
        for target in ('v1', 'v2'):
            for schedule in ('inorder', 'overlap'):
                variant = f'm{rows}-{target}-{schedule}'
                image, graph = root / f'{variant}.llbin', root / f'q-{variant}'
                cmd = [compiler, exported / 'model.mlir', '--weights', exported / 'weights.bin',
                       '--weights-json', exported / 'weights.json', '--calib', exported / 'calib.json',
                       '--target', f'llaccel-{target}', '--schedule', schedule, '--prefill-m', rows,
                       '--weight-chunk-bytes', 768, '--dump-qgraph', graph, '-o', image]
                if target == 'v2': cmd.append('--enable-fusion')
                run(cmd, root / f'compile-{variant}.log')
                golden = GoldenModel(graph)
                assert golden.prefill_m == rows
                # Partial/full chunk, crossing a boundary, and exact final context row.
                for length, count in ((3, 2), (4, 2), (5, 2), (31, 1), (32, 0), (1, 31)):
                    prompt = [i % cfg.vocab for i in range(length)]
                    reference = golden.generate(prompt, count)
                    expected_prefills = (length + rows - 1) // rows
                    assert sum(s['kind'] == 'prefill' for s in reference['steps']) == expected_prefills
                    label = f'{variant}-p{length}-t{count}'
                    ref = root / f'{label}.json'
                    ref.write_text(json.dumps(reference))
                    for backend, interleave in (('func', None), ('func', 7), ('rtl', None)):
                        case = label + f'-{backend}' + ('-seed7' if interleave else '')
                        cmd = [simulator, image, '--backend', backend, '--prompt',
                               ''.join(tokenizer['itos'][i] for i in prompt), '--tokens', count,
                               '--verify', ref, '--quiet']
                        if interleave: cmd += ['--interleave', interleave]
                        output = run(cmd, root / f'{case}.log')
                        assert 'VERIFY: MATCH' in output, case
                        records.append({'prefill_m': rows, 'target': target, 'schedule': schedule,
                                        'prompt_length': length, 'tokens': count, 'backend': backend,
                                        'interleave': interleave, 'result': 'MATCH'})
                print(f'PASS {variant}: 18 backend/context runs', flush=True)
    (root / 'results.json').write_text(json.dumps({'seed': 731, 'cases': records}, indent=2) + '\n')
    print(f'PASS {len(records)} configurable-prefill runs; report: {root / "results.json"}')


if __name__ == '__main__':
    main()
