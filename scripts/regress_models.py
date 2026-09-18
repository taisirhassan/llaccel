#!/usr/bin/env python3
"""Seeded, generated transformer E2E regression (no checkpoint downloads).

Run under `uv run python scripts/regress_models.py --build build/cmake`.
Failures keep exports, binaries, references, and captured command logs under --out.
These random models test numerical/dataflow equivalence, not language quality.
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
    ap.add_argument('--compiler', type=Path)
    ap.add_argument('--out', type=Path, default=Path('build/model-regressions'))
    ap.add_argument('--func-only', action='store_true', help='explicitly omit RTL validation')
    args = ap.parse_args()
    compiler = (args.compiler or args.build / 'compiler/bin/llaccel-compile').resolve()
    simulator = (args.build / 'llaccel-sim').resolve()
    for tool in (compiler, simulator):
        if not tool.is_file(): ap.error(f'missing tool: {tool}; build first')
    args.out.mkdir(parents=True, exist_ok=True)
    # Covers D=16/GQA/bias/odd vocab and D=64/MHA; both force N-chunk relayout.
    configs = [
        ('d16-gqa-bias', ModelConfig(dim=32, n_layers=1, n_heads=2, n_kv_heads=1,
             head_dim=16, ffn=48, vocab=17, max_seq=32, qkv_bias=True)),
        ('d64-mha', ModelConfig(dim=64, n_layers=1, n_heads=1, n_kv_heads=1,
             head_dim=64, ffn=96, vocab=17, max_seq=32)),
    ]
    tokenizer = {'itos': list('abcdefghijklmnopq')}
    cases = [(1, 2), (15, 1), (16, 1), (17, 3), (31, 1), (32, 0), (1, 31)]
    records = []
    def run(cmd, log):
        p = subprocess.run(list(map(str, cmd)), text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
        log.write_text(p.stdout)
        if p.returncode:
            raise RuntimeError(f'command failed ({p.returncode}): {cmd}\n{p.stdout}\nlog: {log}')
        return p.stdout
    for idx, (name, cfg) in enumerate(configs):
        root = args.out / name
        exported = root / 'export'
        torch.manual_seed(1729 + idx)
        torch.set_num_threads(1)
        calibration = root / 'calibration'
        calibration.mkdir(parents=True, exist_ok=True)
        (calibration / 'input.txt').write_text(''.join(tokenizer['itos']) * 100)
        export_dir(TinyLlama(cfg).eval(), exported, cfg, tokenizer, calib_seqs=8, calib_len=32,
                   calib_data=calibration)
        shutil.copyfile(exported / 'tokenizer.json', root / 'tokenizer.json')
        for target in ('v1', 'v2'):
            for schedule in ('inorder', 'overlap'):
                variant = f'{target}-{schedule}'
                image, qgraph = root / f'{variant}.llbin', root / f'q-{variant}'
                # At least one 16-column chunk for the largest K, while forcing
                # chunking of wider linears. This includes fused aux relayout.
                chunk = 16 * cfg.ffn
                command = [compiler, exported / 'model.mlir', '--weights', exported / 'weights.bin',
                    '--weights-json', exported / 'weights.json', '--calib', exported / 'calib.json',
                    '--target', f'llaccel-{target}', '--schedule', schedule,
                    '--weight-chunk-bytes', chunk, '--dump-qgraph', qgraph, '-o', image]
                if target == 'v2': command.append('--enable-fusion')
                run(command, root / f'compile-{variant}.log')
                golden = GoldenModel(qgraph)
                for length, tokens in cases:
                    prompt = ''.join(tokenizer['itos'][i % cfg.vocab] for i in range(length))
                    reference = golden.generate([i % cfg.vocab for i in range(length)], tokens)
                    case = f'{variant}-p{length}-t{tokens}'
                    ref_path = root / f'golden-{case}.json'
                    ref_path.write_text(json.dumps(reference))
                    modes = [('func', None), ('func', 7)]
                    if not args.func_only: modes.append(('rtl', None))
                    for backend, seed in modes:
                        label = f'{case}-{backend}' + (f'-seed{seed}' if seed else '')
                        command = [simulator, image, '--backend', backend, '--prompt', prompt,
                                   '--tokens', tokens, '--verify', ref_path, '--quiet']
                        if seed is not None: command += ['--interleave', seed]
                        output = run(command, root / f'{label}.log')
                        if 'VERIFY: MATCH' not in output:
                            raise RuntimeError(f'missing positive verification result: {label}')
                        records.append({'shape': name, 'variant': variant, 'prompt_length': length,
                            'tokens': tokens, 'backend': backend, 'seed': seed,
                            'launches': len(reference['steps']), 'result': 'MATCH'})
                print(f'PASS {name} {variant}: {len(cases)} prompt/context cases', flush=True)
    report = {'seed': 1729, 'rtl_required': not args.func_only, 'cases': records}
    (args.out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS {len(records)} model/backend runs; report: {args.out / "results.json"}')

if __name__ == '__main__':
    main()
