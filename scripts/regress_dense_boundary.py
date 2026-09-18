#!/usr/bin/env python3
"""Software-only context-end checks for an already compiled dense HF fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from llaccel.golden import GoldenModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=Path('work/dense-support/qwen3-independent-width'))
    parser.add_argument('--simulator', type=Path, default=Path('build/llama-qwen-software/llaccel-sim'))
    parser.add_argument('--out', type=Path, default=Path('work/dense-boundary'))
    args = parser.parse_args()
    image = (args.fixture / 'v2-overlap.llbin').resolve()
    graph = (args.fixture / 'q-v2-overlap').resolve()
    simulator = args.simulator.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    files = [image, simulator, graph / 'qgraph.json', graph / 'qweights.bin']
    hashes = {str(path): sha(path) for path in files}
    report = {'complete': False, 'scope': 'software only; seeded Qwen3 context-end prefill/decode',
              'inputs_sha256': hashes, 'cases': []}
    def save():
        temp = args.out / 'results.json.tmp'
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(args.out / 'results.json')
    save()
    for length, tokens in ((31, 1), (32, 0)):
        case = args.out / f'prefill-{length}-decode-{tokens}'
        case.mkdir(exist_ok=True)
        ids = [4 + i % 32 for i in range(length)]
        prompt = case / 'prompt.json'
        prompt.write_text(json.dumps(ids))
        golden = case / 'golden.json'
        golden.write_text(json.dumps(GoldenModel(graph).generate(ids, tokens)))
        for seed in (None, 7):
            name = 'ordered' if seed is None else 'interleaved'
            command = [str(simulator), str(image), '--backend', 'func', '--prompt-ids', str(prompt.resolve()),
                       '--tokens', str(tokens), '--verify', str(golden.resolve()),
                       '--out', str((case / f'{name}.json').resolve()), '--quiet']
            if seed is not None:
                command += ['--interleave', str(seed)]
            log = case / f'{name}.log'
            with log.open('w') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode or 'VERIFY: MATCH' not in log.read_text():
                report['failure'] = str(log)
                save()
                raise RuntimeError(f'boundary failed: {log}')
            report['cases'].append({'prefill_tokens': length, 'decode_tokens': tokens,
                                    'interleave_seed': seed, 'result': 'MATCH'})
            save()
    assert all(sha(path) == hashes[str(path)] for path in files), 'inputs changed during validation'
    report['complete'] = True
    save()
    print(f'PASS {len(report["cases"])} software boundary checks')


if __name__ == '__main__':
    main()
