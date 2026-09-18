#!/usr/bin/env python3
"""Offline HF checkpoint -> compiler -> software-only differential regression.

Run `uv run --extra hf python scripts/regress_dense_hf.py --build build/llama-qwen-software`.
Uses seeded random Transformers dense Llama, Qwen2 and Qwen3 checkpoints with real byte-level
BPE tokenizers. Tests compatibility/dataflow, not pretrained language quality.
Every intermediate, subprocess log, and final matrix is retained under --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', type=Path, default=Path('build/llama-qwen-software'))
    ap.add_argument('--out', type=Path, default=Path('work/dense-support'))
    ap.add_argument('--cases', nargs='+', help='Run only named matrix cases')
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    compiler = (args.build / 'compiler/bin/llaccel-compile').resolve()
    simulator = (args.build / 'llaccel-sim').resolve()
    for tool in (compiler, simulator):
        if not tool.is_file():
            ap.error(f'missing tool: {tool}; build first')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    corpus = repo / 'tests/data/hf-calibration.txt'
    if not corpus.is_file():
        ap.error(f'missing calibration corpus: {corpus}')
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors
    from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config,
                              Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast)
    torch.set_num_threads(2)
    raw_tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw_tokenizer.decoder = decoders.ByteLevel()
    raw_tokenizer.train([str(corpus)], trainers.BpeTrainer(vocab_size=128,
                        special_tokens=['[UNK]', '[PAD]', '[BOS]', '[EOS]'], show_progress=False))
    raw_tokenizer.post_processor = processors.TemplateProcessing(
        single='[BOS] $A', special_tokens=[('[BOS]', raw_tokenizer.token_to_id('[BOS]'))])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer,
        unk_token='[UNK]', pad_token='[PAD]', bos_token='[BOS]', eos_token='[EOS]')
    vocab = max(tokenizer.get_vocab().values()) + 1
    prompt = 'The history'
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if tokenizer.unk_token_id in prompt_ids or not 1 <= len(prompt_ids) <= 30:
        raise ValueError(f'probe prompt must tokenize without unknowns inside context: {prompt_ids}')
    records = []
    manifest = {'complete': False, 'seed': 9321, 'context': 32, 'prefill_m': 4, 'weight_chunk_policy': 'max(768, 16 * max(residual, FFN, query width))',
        'fixture': 'seeded random HF models; not pretrained language quality',
        'validation_scope': 'software only; no RTL or hardware validation',
        'calibration_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
        'compiler_sha256': hashlib.sha256(compiler.read_bytes()).hexdigest(),
        'simulator_sha256': hashlib.sha256(simulator.read_bytes()).hexdigest(),
        'vocab': vocab, 'cases': records}

    def save_report():
        temporary = out / 'results.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(out / 'results.json')

    def run(command, log):
        with log.open('w') as stream:
            result = subprocess.run(list(map(str, command)), cwd=repo,
                stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            manifest['failure'] = {'returncode': result.returncode, 'log': str(log),
                                   'command': list(map(str, command))}
            save_report()
            raise RuntimeError(f'command failed ({result.returncode}); inspect {log}')

    save_report()
    from llaccel.hf_import import config_from_hf
    negative = dict(model_type='qwen2', hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, intermediate_size=48,
        vocab_size=vocab, max_position_embeddings=32, use_sliding_window=True,
        sliding_window=8, layer_types=['sliding_attention'])
    try:
        config_from_hf(negative, max_seq=32)
    except ValueError as error:
        if 'sliding' not in str(error) and 'full_attention' not in str(error):
            raise
        manifest['negative_checks'] = [{'case': 'active-sliding-window',
            'result': 'REJECTED', 'error': str(error)}]
    else:
        raise RuntimeError('active sliding-window semantics were silently accepted')
    save_report()
    families = {'llama': (LlamaConfig, LlamaForCausalLM),
                'qwen2': (Qwen2Config, Qwen2ForCausalLM),
                'qwen3': (Qwen3Config, Qwen3ForCausalLM)}
    matrix = [
        ('llama-untied', 'llama', {}),
        ('llama-tied-bias', 'llama', {'tie_word_embeddings': True, 'attention_bias': True, 'mlp_bias': True}),
        ('llama-linear', 'llama', {'rope_scaling': {'rope_type': 'linear', 'factor': 2.0}}),
        ('llama3-rope', 'llama', {'rope_scaling': {'rope_type': 'llama3', 'factor': 8.0,
            'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 16}}),
        ('llama-head128', 'llama', {'hidden_size': 256, 'head_dim': 128, 'intermediate_size': 288}),
        ('llama-head256', 'llama', {'hidden_size': 256, 'num_attention_heads': 1,
            'head_dim': 256, 'intermediate_size': 288}),
        ('qwen2-untied', 'qwen2', {}),
        ('qwen2-tied-linear', 'qwen2', {'tie_word_embeddings': True,
            'rope_scaling': {'rope_type': 'linear', 'factor': 2.0}}),
        ('qwen2-yarn', 'qwen2', {'rope_parameters': {'rope_type': 'yarn', 'factor': 4.0,
            'original_max_position_embeddings': 32, 'rope_theta': 10000.0}}),
        ('qwen3-qknorm', 'qwen3', {}),
        ('qwen3-independent-width', 'qwen3', {'hidden_size': 64, 'head_dim': 128,
            'intermediate_size': 96, 'tie_word_embeddings': True}),
    ]
    selected = set(args.cases) if args.cases else {case[0] for case in matrix}
    unknown = selected - {case[0] for case in matrix}
    if unknown:
        ap.error(f'unknown cases: {sorted(unknown)}')
    manifest['requested_cases'] = sorted(selected)
    for idx, (name, kind, overrides) in enumerate(matrix):
        if name not in selected:
            continue
        config_cls, model_cls = families[kind]
        root = out / name
        checkpoint, exported = root / 'checkpoint', root / 'export'
        checkpoint.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(9321 + idx)
        config_args = dict(hidden_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            intermediate_size=48, vocab_size=vocab, max_position_embeddings=32,
            rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id)
        config_args.update(overrides)
        config = config_cls(**config_args)
        weight_chunk = max(768, 16 * max(config.hidden_size, config.intermediate_size,
                                      config.num_attention_heads * config_args['head_dim']))
        model = model_cls(config).float().eval()
        # Constructor defaults make biases zero and Q/K norm weights one;
        # perturb both so omission cannot silently pass the float oracle.
        with torch.no_grad():
            for parameter_name, parameter in model.named_parameters():
                if parameter_name.endswith('.bias'):
                    parameter.uniform_(-0.03, 0.03)
                elif '.q_norm.weight' in parameter_name or '.k_norm.weight' in parameter_name:
                    parameter.uniform_(0.75, 1.25)
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        del model
        export_command = [sys.executable, '-m', 'llaccel.hf', 'export', checkpoint, '--context', 32,
             '--calibration', corpus, '--out', exported]
        export_command += ['--calibration-sequences', 4]
        run(export_command, root / 'export.log')
        metadata = json.loads((exported / 'hf-import.json').read_text())
        if not isinstance(metadata.get('float_import_validation'), list):
            raise RuntimeError('independent Transformers float validation was not run')
        for target in ('v1', 'v2'):
            for schedule in ('inorder', 'overlap'):
                variant = f'{target}-{schedule}'
                image, qgraph = root / f'{variant}.llbin', root / f'q-{variant}'
                command = [compiler, exported / 'model.mlir', '--weights', exported / 'weights.bin',
                    '--weights-json', exported / 'weights.json', '--calib', exported / 'calib.json',
                    '--target', f'llaccel-{target}', '--schedule', schedule, '--prefill-m', 4,
                    '--weight-chunk-bytes', weight_chunk, '--dump-qgraph', qgraph, '-o', image]
                if target == 'v2':
                    command.append('--enable-fusion')
                run(command, root / f'compile-{variant}.log')
                case_dir = root / variant
                run([sys.executable, '-m', 'llaccel.hf', 'run', image, '--qgraph', qgraph,
                    '--tokenizer', exported / 'hf-tokenizer', '--checkpoint', checkpoint,
                    '--simulator', simulator, '--prompt', prompt, '--tokens', 2,
                    '--backends', 'func', '--out', case_dir], root / f'run-{variant}.log')
                report = json.loads((case_dir / 'report.json').read_text())
                if report['exact_integer_verification'] != 'MATCH':
                    raise RuntimeError(f'HF runner did not verify {name} {variant}')
                # HF CLI uses seed 7 for func; additionally cover ordered execution.
                run([simulator, image, '--backend', 'func', '--prompt-ids', case_dir / 'prompt-ids.json',
                    '--tokens', 2, '--verify', case_dir / 'golden.json',
                    '--out', case_dir / 'func-ordered.json', '--quiet'], case_dir / 'func-ordered.log')
                for backend, seed, logfile in [('func', None, 'func-ordered.log'),
                                               ('func', 7, 'func.log')]:
                    if 'VERIFY: MATCH' not in (case_dir / logfile).read_text():
                        raise RuntimeError(f'missing positive verification: {case_dir / logfile}')
                    records.append({'model': name, 'variant': variant, 'backend': backend,
                        'interleave_seed': seed, 'weight_chunk_bytes': weight_chunk, 'prompt_ids': prompt_ids, 'new_tokens': 2,
                        'result': 'MATCH', 'image_sha256': report['image_sha256'],
                        'float_import_validation': metadata['float_import_validation'],
                        'fp32_teacher_forced': report.get('fp32_teacher_forced')})
                save_report()
                print(f'PASS {name} {variant}: ordered func, randomized func', flush=True)
    for tool, key in ((compiler, 'compiler_sha256'), (simulator, 'simulator_sha256')):
        if hashlib.sha256(tool.read_bytes()).hexdigest() != manifest[key]:
            raise RuntimeError(f'tool changed during campaign: {tool}')
    manifest['complete'] = True
    save_report()
    print(f'PASS {len(records)} HF model/backend runs; report: {out / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
