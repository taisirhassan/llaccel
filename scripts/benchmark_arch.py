"""Reproducible compiler tiling × DMA window × DRAM latency experiment.

Every measurement requires exact logits/tokens against an independently
executed golden graph. Simulator paths identify separately elaborated hardware.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from llaccel.data import CharTokenizer
from llaccel.golden import GoldenModel
from regress_pretrained import write_json, sha256


def run(args, log):
    with log.open('w') as f:
        subprocess.run([str(x) for x in args], stdout=f, stderr=subprocess.STDOUT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compiler', type=Path, default=Path('build/dma32/compiler/bin/llaccel-compile'))
    ap.add_argument('--baseline', type=Path, default=Path('build/review-rtl/llaccel-sim'))
    ap.add_argument('--candidate', type=Path, default=Path('build/dma32/llaccel-sim'))
    ap.add_argument('--export', type=Path, default=Path('build/export'))
    ap.add_argument('--out', type=Path, default=Path('build/architecture-study'))
    ap.add_argument('--tokens', type=int, default=16)
    ap.add_argument('--prompt', default='ROMEO:')
    ap.add_argument('--prompt-ids', type=Path, help='externally tokenized JSON integer array; skips char tokenizer')
    ap.add_argument('--prefill-m', type=int, default=16)
    ap.add_argument('--chunks', nargs='+', type=int, default=[8192,16384,32768,65536])
    ap.add_argument('--latencies', nargs='+', type=int, default=[1,20,100,400])
    ap.add_argument('--schedules', nargs='+', choices=['inorder','overlap'], default=['overlap'])
    ap.add_argument('--targets', nargs='+', choices=['v1','v2'], default=['v1','v2'])
    ap.add_argument('--simulator', action='append', help='LABEL=PATH, replaces the default depth16/depth32 pair')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.tokens < 1 or not 1 <= args.prefill_m <= 16 or any(x < 1 for x in args.chunks) or any(x < 0 for x in args.latencies):
        ap.error('tokens/chunks must be positive and latencies nonnegative; prefill-m must be 1..16')
    ids = json.loads(args.prompt_ids.read_text()) if args.prompt_ids else CharTokenizer.from_json(args.export / 'tokenizer.json').encode(args.prompt)
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        ap.error('prompt IDs must be a nonempty integer array')
    prompt_path = args.out / 'prompt-ids.json'
    write_json(prompt_path, ids)
    simulators = [(16, args.baseline), (32, args.candidate)]
    if args.simulator:
        simulators = []
        for item in args.simulator:
            label, separator, path = item.partition('=')
            if not separator or not label or not path or not all(c.isalnum() or c in '-_' for c in label):
                ap.error('--simulator requires filesystem-safe LABEL=PATH')
            simulators.append((label, Path(path)))
    for _, simulator in simulators:
        if not simulator.is_file(): ap.error(f'missing simulator: {simulator}')
    write_json(args.out / 'status.json', {'status':'RUNNING', 'compiler_sha256':sha256(args.compiler)})
    try:
        records = []
        for target in args.targets:
            for schedule in args.schedules:
                for chunk in args.chunks:
                    label = f'{target}-{schedule}-chunk{chunk}'
                    graph = args.out / ('q-' + label)
                    image = args.out / (label + '.llbin')
                    command = [args.compiler, args.export / 'model.mlir', '--weights', args.export / 'weights.bin',
                               '--weights-json', args.export / 'weights.json', '--calib', args.export / 'calib.json',
                               '--target', 'llaccel-' + target, '--schedule', schedule, '--weight-chunk-bytes', chunk,
                               '--prefill-m', args.prefill_m, '--dump-qgraph', graph, '-o', image]
                    if target == 'v2': command += ['--enable-fusion']
                    run(command, args.out / (label + '-compile.log'))
                    reference = GoldenModel(graph).generate(ids, args.tokens)
                    golden = args.out / (label + '-golden.json')
                    golden.write_text(json.dumps(reference))
                    for depth, simulator in simulators:
                        for latency in args.latencies:
                            name = f'{label}-depth{depth}-latency{latency}'
                            result = args.out / (name + '.json')
                            run([simulator, image, '--backend', 'rtl', '--prompt-ids', prompt_path, '--tokens', args.tokens, '--dram-latency', latency,
                                 '--verify', golden, '--out', result, '--quiet'], args.out / (name + '.log'))
                            d = json.loads(result.read_text())
                            for field in ['prompt_tokens', 'generated', 'argmax_per_step', 'logits_last_rows']:
                                if d[field] != reference[field]: raise ValueError(f'{name}: incomplete {field} match')
                            if 'VERIFY: MATCH' not in (args.out / (name + '.log')).read_text():
                                raise ValueError(f'{name}: no positive runtime verification')
                            steps = [s['perf'] for s in d['steps'] if s['pos'] >= len(ids)]
                            if not steps: raise ValueError('benchmark has no decode launches')
                            p = {k: sum(s[k] for s in steps) / len(steps) for k in steps[0]}
                            records.append({'target': target, 'schedule': schedule, 'chunk_bytes': chunk, 'dma_depth': depth if isinstance(depth,int) else None, 'simulator_label': str(depth),
                                            'dram_latency': latency, 'tokens': args.tokens, 'decode': p,
                                            'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                                            'simulator_sha256': hashlib.sha256(simulator.read_bytes()).hexdigest(),
                                            'result': 'MATCH'})
                    write_json(args.out / 'results.json', records)
                    print('PASS', label, len(simulators), 'simulators,', len(args.latencies), 'DRAM latencies', flush=True)
        (args.out / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
        for depth, _ in simulators:
            rows = [r for r in records if r['simulator_label'] == str(depth) and r['dram_latency'] == 100]
            if not rows: continue
            best = min(rows, key=lambda r:r['decode']['cycles'])
            print('BEST at latency100:', depth, best['target'], best['schedule'], best['chunk_bytes'], best['decode']['cycles'])
        write_json(args.out / 'status.json', {'status':'PASS', 'measurements':len(records), 'compiler_sha256':sha256(args.compiler)})
        print('PASS',len(records),'verified architecture measurements')
    except BaseException as error:
        write_json(args.out / 'status.json', {'status':'FAILED', 'error':f'{type(error).__name__}: {error}'})
        raise

if __name__ == '__main__':
    main()
