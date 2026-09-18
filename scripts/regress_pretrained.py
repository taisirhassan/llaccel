#!/usr/bin/env python3
"""Reproduce pinned public Llama/Qwen checkpoint compilation and func/RTL checks.

Run with `uv run --extra hf python scripts/regress_pretrained.py` after building
compiler and RTL simulator. Models execute sequentially to conserve memory.
All stages retain logs; a failed or interrupted run never records aggregate PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
MODELS = {
    'qwen': ('Qwen/Qwen2.5-0.5B', '060db6499f32faf8b98477b0a26969ef7d8b9987',
             'hf-qwen', 'Qwen2.5-0.5B'),
    'llama': ('HuggingFaceTB/SmolLM2-135M', '93efa2f097d58c2a74874c7e644dbc9b0cee75a2',
              'hf-llama', 'SmolLM2-135M'),
}
ALLOW = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
         '*.safetensors', '*.safetensors.index.json']


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run(command, log):
    print(f'Running {log.parent.name}/{log.name}', flush=True)
    with log.open('w') as stream:
        stream.write('Command: ' + json.dumps(list(map(str, command))) + '\n')
        stream.flush()
        result = subprocess.run(list(map(str, command)), cwd=REPO,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'stage failed with exit code {result.returncode}; see {log}')


def checkpoint_manifest(checkpoint, revision):
    files = sorted({p for pattern in ALLOW for p in checkpoint.glob(pattern) if p.is_file()})
    for required in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
        if not (checkpoint / required).is_file():
            raise ValueError(f'missing checkpoint file: {checkpoint / required}')
    if not any(p.suffix == '.safetensors' for p in files):
        raise ValueError(f'checkpoint has no safetensors weights: {checkpoint}')
    digests = {}
    for path in files:
        metadata = checkpoint / '.cache/huggingface/download' / (path.name + '.metadata')
        revision_lines = metadata.read_text().splitlines() if metadata.is_file() else []
        if not revision_lines or revision_lines[0] != revision:
            raise ValueError(f'cannot establish pinned revision for {path}; rerun without --skip-download')
        digests[path.name] = sha256(path)
    return digests


def validate_export(exported, checkpoint, args, calibration_hash, source_files):
    metadata = json.loads((exported / 'hf-import.json').read_text())
    expected = {'context': args.context, 'calibration_sequences': args.calibration_sequences,
                'calibration_sha256': calibration_hash, 'source': str(checkpoint)}
    if hasattr(args, 'calibration_sampling') and metadata.get('calibration_selection', {}).get('sampling') != args.calibration_sampling:
        raise ValueError('export calibration sampling differs')
    if hasattr(args, 'calibration_length'):
        expected['calibration_length'] = args.calibration_length or args.context
        expected['optimize_quantization'] = False
        if metadata.get('smoothquant', {}).get('auto_alpha', False) != args.smoothquant_auto_alpha:
            raise ValueError('export automatic alpha setting differs')
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'export metadata {key} differs: expected {value!r}; export again')
    if metadata.get('hf_config') != json.loads((checkpoint / 'config.json').read_text()):
        raise ValueError('export HF configuration differs from checkpoint')
    if metadata.get('smoothquant', {}).get('alpha') != 0.5:
        raise ValueError('export must use SmoothQuant alpha 0.5')
    for feature in ['smooth_values', 'smooth_ffn', 'smooth_lm_head']:
        if metadata.get('smoothquant', {}).get(feature) is not True:
            raise ValueError(f'export must enable SmoothQuant {feature}; export again')
    if not isinstance(metadata.get('float_import_validation'), list) or not metadata['float_import_validation']:
        raise ValueError('export has no independent HF float import validation')
    for name, digest in metadata.get('checkpoint_files', {}).items():
        if source_files.get(name) != digest:
            raise ValueError(f'export checkpoint hash differs: {name}')
    if not metadata.get('checkpoint_files'):
        raise ValueError('export lacks checkpoint weight hashes')
    for name in ['model.mlir', 'weights.bin', 'weights.json', 'calib.json', 'hf-tokenizer/tokenizer.json']:
        if not (exported / name).is_file():
            raise ValueError(f'incomplete export: {exported / name}')
        if metadata.get('export_files_sha256', {}).get(name) != sha256(exported / name):
            raise ValueError(f'export artifact hash differs or is missing: {name}')
    return metadata


def collect_run(root, image, simulator):
    directory = root / 'run'
    report = json.loads((directory / 'report.json').read_text())
    if set(report.get('backends', [])) != {'func', 'rtl'} or report.get('exact_integer_verification') != 'MATCH':
        raise ValueError('both func and RTL exact verification are required')
    if not report.get('fp32_teacher_forced', {}).get('rows'):
        raise ValueError('missing original-checkpoint fp32 quality comparison')
    if report.get('image_sha256') != sha256(image) or report.get('simulator_sha256') != sha256(simulator):
        raise ValueError('run provenance hashes do not match current image/simulator')
    golden = json.loads((directory / 'golden.json').read_text())
    perf = {}
    for backend in ['func', 'rtl']:
        actual = json.loads((directory / f'{backend}.json').read_text())
        for field in ['prompt_tokens', 'generated', 'argmax_per_step', 'logits_last_rows']:
            if actual.get(field) != golden.get(field) or field not in golden:
                raise ValueError(f'{backend} does not match complete golden {field}')
        if 'VERIFY: MATCH' not in (directory / f'{backend}.log').read_text():
            raise ValueError(f'{backend} lacks positive simulator verification')
        perf[backend] = {'launches': len(actual['steps']), 'total': actual['perf_total'],
                         'steps': actual['steps']}
    return {'verification': report, 'performance': perf}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--build', type=Path, default=Path('build/dma32'))
    parser.add_argument('--out', type=Path, default=Path('build'))
    parser.add_argument('--checkpoint-root', type=Path, default=Path('checkpoints/hf'))
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--context', type=int, default=32)
    parser.add_argument('--calibration-sequences', type=int, default=64)
    parser.add_argument('--prompt', default='Hello')
    parser.add_argument('--prefill-m', type=int, default=1)
    parser.add_argument('--calibration', type=Path, default=REPO/'tests/data/hf-calibration.txt')
    parser.add_argument('--calibration-length', type=int)
    parser.add_argument('--calibration-sampling', choices=['line-prefix', 'uniform-windows'], default='line-prefix')
    parser.add_argument('--smoothquant-auto-alpha', action='store_true')
    parser.add_argument('--reuse-export', action='store_true', help='require matching complete export provenance')
    parser.add_argument('--skip-download', action='store_true', help='require existing files with pinned HF cache metadata')
    args = parser.parse_args()
    if not 8 <= args.context <= 4096 or not 1 <= args.tokens < args.context:
        parser.error('context must be 8..4096 and tokens must be 1..context-1')
    if args.prefill_m not in (1,4,16) or args.context % args.prefill_m:
        parser.error('prefill-m must be 1/4/16 and divide context')
    if args.calibration_length is not None and not 2 <= args.calibration_length <= args.context:
        parser.error('calibration-length must be in [2,context]')
    if args.calibration_sequences < 1 or not args.prompt:
        parser.error('calibration-sequences and prompt must be nonempty/positive')
    for name in ['build', 'out', 'checkpoint_root']:
        setattr(args, name, getattr(args, name).resolve())
    compiler = args.build / 'compiler/bin/llaccel-compile'
    simulator = args.build / 'llaccel-sim'
    calibration = args.calibration.resolve()
    for path in [compiler, simulator, calibration]:
        if not path.is_file():
            parser.error(f'missing required input/tool: {path}')
    args.out.mkdir(parents=True, exist_ok=True)
    aggregate = args.out / 'pretrained-results.json'
    summary = {'status': 'RUNNING', 'models': [], 'settings': {
        'context': args.context, 'tokens': args.tokens, 'prompt': args.prompt,
        'calibration_sequences': args.calibration_sequences, 'smoothquant_alpha': 0.5,
        'calibration_sha256': sha256(calibration), 'compiler_sha256': sha256(compiler),
        'simulator_sha256': sha256(simulator), 'target': 'llaccel-v1', 'schedule': 'overlap',
        'prefill_m': args.prefill_m, 'weight_chunk_bytes': 131072,
        'calibration_length': args.calibration_length or args.context,
        'calibration_sampling': args.calibration_sampling,
        'smoothquant_auto_alpha': args.smoothquant_auto_alpha}}
    write_json(aggregate, summary)
    try:
        for name in dict.fromkeys(args.models):
            repository, revision, slug, checkpoint_name = MODELS[name]
            root = args.out / slug
            root.mkdir(parents=True, exist_ok=True)
            checkpoint = args.checkpoint_root / checkpoint_name
            if not args.skip_download:
                download = ('from huggingface_hub import snapshot_download; import sys,json; '
                            'snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], '
                            'local_dir=sys.argv[3], allow_patterns=json.loads(sys.argv[4]))')
                run([sys.executable, '-c', download, repository, revision, checkpoint, json.dumps(ALLOW)], root / 'download.log')
            source_files = checkpoint_manifest(checkpoint, revision)
            exported = root / 'export'
            if not args.reuse_export:
                run([sys.executable, '-m', 'llaccel.hf', 'export', checkpoint, '--out', exported,
                     '--context', args.context, '--calibration', calibration,
                     '--calibration-sequences', args.calibration_sequences,
                     '--smoothquant-alpha', '0.5', '--calibration-sampling', args.calibration_sampling] +
                    (['--calibration-length', args.calibration_length] if args.calibration_length else []) +
                    (['--smoothquant-auto-alpha'] if args.smoothquant_auto_alpha else []), root / 'export.log')
            metadata = validate_export(exported, checkpoint, args, summary['settings']['calibration_sha256'], source_files)
            image, qgraph = root / 'model.llbin', root / 'qgraph'
            run([compiler, exported / 'model.mlir', '--weights', exported / 'weights.bin',
                 '--weights-json', exported / 'weights.json', '--calib', exported / 'calib.json',
                 '--target', 'llaccel-v1', '--schedule', 'overlap', '--prefill-m', args.prefill_m,
                 '--weight-chunk-bytes', 131072, '--dump-qgraph', qgraph, '-o', image], root / 'compile.log')
            run([sys.executable, '-m', 'llaccel.hf', 'run', image, '--qgraph', qgraph,
                 '--tokenizer', exported / 'hf-tokenizer', '--checkpoint', checkpoint,
                 '--simulator', simulator, '--prompt', args.prompt, '--tokens', args.tokens,
                 '--backends', 'func', 'rtl', '--out', root / 'run'], root / 'run.log')
            result = collect_run(root, image, simulator)
            summary['models'].append({'name': name, 'repository': repository, 'revision': revision,
                'checkpoint': str(checkpoint), 'source_files_sha256': source_files,
                'import': metadata, **result})
            write_json(aggregate, summary)
            print(f'PASS {repository}: complete func/RTL integer verification; quality recorded separately', flush=True)
        summary['status'] = 'PASS'
        write_json(aggregate, summary)
        print(f'Results: {aggregate}', flush=True)
    except BaseException as error:
        summary['status'] = 'FAILED'
        summary['error'] = f'{type(error).__name__}: {error}'
        write_json(aggregate, summary)
        raise


if __name__ == '__main__':
    main()
