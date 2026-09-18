"""Hugging Face checkpoint import and tokenized RTL execution.

Requires `uv sync --extra hf`. No model code from a checkpoint is executed.
The host tokenizes, copies embeddings and chooses tokens; transformer math runs
through the same compiled ISA/RTL as the character-model path.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import torch


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tokenizer_at(checkpoint):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)


def calibration_sequences(tokenizer, text, count, length, sampling="line-prefix"):
    """Construct reproducible real-token windows and record their exact identity."""
    if count < 1 or length < 2:
        raise ValueError("calibration-sequences must be positive and length at least two")
    texts = text.splitlines()
    pool = tokenizer.encode("\n".join(texts), add_special_tokens=True)
    starts = None
    if sampling == 'uniform-windows':
        if len(pool) < length:
            raise ValueError("calibration corpus is shorter than requested window")
        starts = [i * (len(pool) - length) // max(1, count - 1) for i in range(count)]
        sequences = [pool[start:start+length] for start in starts]
    elif sampling == 'line-prefix':
        sequences = [tokenizer.encode(t, add_special_tokens=True)[:length] for t in texts if t.strip()]
        sequences = [seq for seq in sequences if len(seq) >= 2][:count]
        import random
        rng = random.Random(0)
        while len(sequences) < count and len(pool) >= 2:
            size = min(length, len(pool))
            start = rng.randrange(len(pool) - size + 1)
            sequences.append(pool[start:start+size])
    else:
        raise ValueError("unknown calibration sampling mode")
    if not sequences:
        raise ValueError("calibration text needs nonempty lines encoding at least two tokens")
    return sequences, {"sampling": sampling, "seed": 0, "corpus_tokens": len(pool),
        "window_starts": starts, "sampled_tokens": sum(map(len, sequences)),
        "token_ids_sha256": hashlib.sha256(json.dumps(sequences, separators=(',', ':')).encode()).hexdigest()}


def export_checkpoint(args):
    if not 8 <= args.context <= 4096:
        raise ValueError("export context must be in [8,4096]")
    from .hf_import import load_hf_checkpoint
    from .export import import_model, emit_mlir, write_weights
    from .calibrate import calibrate_tokens
    model, metadata = load_hf_checkpoint(args.checkpoint, max_seq=args.context)
    tok = tokenizer_at(args.checkpoint)
    if max(tok.get_vocab().values()) >= model.cfg.vocab:
        raise ValueError("tokenizer IDs exceed model vocabulary")
    calibration_length = args.context if args.calibration_length is None else args.calibration_length
    if not 2 <= calibration_length <= args.context:
        raise ValueError("calibration-length must be in [2, context]")
    if args.smoothquant_auto_alpha and args.smoothquant_alpha is None:
        raise ValueError("smoothquant-auto-alpha requires smoothquant-alpha")
    sequences, sequence_metadata = calibration_sequences(tok, args.calibration.read_text(),
        args.calibration_sequences, calibration_length, getattr(args, 'calibration_sampling', 'line-prefix'))
    args.out.mkdir(parents=True, exist_ok=True)
    # A partial overwrite must not retain a prior successful export's manifest.
    (args.out / 'hf-import.json').unlink(missing_ok=True)
    if args.smoothquant_alpha is not None:
        from .smoothquant import smooth_model
        metadata["smoothquant"] = smooth_model(model, sequences, alpha=args.smoothquant_alpha,
                                              auto_alpha=args.smoothquant_auto_alpha)
    if not args.skip_float_check:
        from transformers import AutoModelForCausalLM
        reference = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
            trust_remote_code=False, dtype=torch.float32, attn_implementation="eager").eval()
        probes=[]
        with torch.no_grad():
            for ids in sequences[:2]:
                x=torch.tensor([ids[:min(len(ids), 8)]],dtype=torch.long)
                actual=model(x); expected=reference(x,use_cache=False).logits
                error=float((actual-expected).abs().max())
                torch.testing.assert_close(actual,expected,atol=2e-4,rtol=2e-4)
                probes.append({"length":x.shape[1],"max_abs_error":error})
        del reference
        metadata["float_import_validation"] = probes
    else:
        metadata["float_import_validation"] = "not run"
    imported = import_model(model, model.cfg)
    (args.out/'model.mlir').write_text(emit_mlir(imported))
    write_weights(imported,args.out)
    calibration=calibrate_tokens(imported,sequences,optimize_quantization=args.optimize_quantization)
    (args.out/'calib.json').write_text(json.dumps(calibration,indent=2)+'\n')
    tok.save_pretrained(args.out/'hf-tokenizer')
    metadata.update({"source":str(args.checkpoint.resolve()),"context":args.context,
        "calibration_sha256":hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        "calibration_sequences":len(sequences),"calibration_length":calibration_length,
        "calibration_selection":sequence_metadata,
        "optimize_quantization":args.optimize_quantization,"model_config":model.cfg.to_dict()})
    metadata['export_files_sha256'] = {name:file_sha256(args.out/name) for name in
        ['model.mlir','weights.bin','weights.json','calib.json','hf-tokenizer/tokenizer.json']}
    temporary = args.out/'hf-import.json.tmp'
    temporary.write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    temporary.replace(args.out/'hf-import.json')
    print(f'Exported {args.out}; {len(sequences)} tokenizer-correct calibration sequences; '
          f'compiled context {args.context}, checkpoint context {metadata["context_reduced_from"]}',flush=True)


def _write_run_report(path, report):
    """Publish one complete JSON state; readers never observe a partial write."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def run_checkpoint(args):
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'report.json'
    state = {'status': 'RUNNING', 'prompt': args.prompt, 'backends': args.backends}
    # Invalidate an earlier successful report before tokenization or model loading.
    _write_run_report(path, state)
    try:
        report = _run_checkpoint(args)
        _write_run_report(path, {**report, 'status': 'PASS'})
    except BaseException as error:
        _write_run_report(path, {**state, 'status': 'FAILED',
                                'error': f'{type(error).__name__}: {error}'})
        raise
    print(f'PASS exact token/logit verification: {args.out}', flush=True)


def _run_checkpoint(args):
    from .golden import GoldenModel
    tokenizer=tokenizer_at(args.tokenizer)
    ids=tokenizer.encode(args.prompt,add_special_tokens=True)
    args.out.mkdir(parents=True,exist_ok=True)
    prompt_file=args.out/'prompt-ids.json';prompt_file.write_text(json.dumps(ids))
    golden=GoldenModel(args.qgraph)
    expected=golden.generate(ids,args.tokens)
    reference=args.out/'golden.json';reference.write_text(json.dumps(expected))
    del golden
    for backend in args.backends:
        result=args.out/f'{backend}.json'
        command=[str(args.simulator),str(args.image),'--backend',backend,'--prompt-ids',str(prompt_file),
            '--tokens',str(args.tokens),'--verify',str(reference),'--out',str(result),'--quiet']
        if backend=='func':command+=['--interleave','7']
        with (args.out/f'{backend}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        record=json.loads(result.read_text())
        print(f'{backend}: {tokenizer.decode(record["generated"],skip_special_tokens=False)!r}',flush=True)
    report={"prompt":args.prompt,"prompt_ids":ids,"generated":expected['generated'],
        "text":tokenizer.decode(expected['generated'],skip_special_tokens=False),
        "backends":args.backends,"exact_integer_verification":"MATCH",
        "image_sha256":file_sha256(args.image),
        "simulator_sha256":file_sha256(args.simulator)}
    graph_config=json.loads((args.qgraph/'qgraph.json').read_text())['model']
    report['model']=graph_config
    report['tokenizer_files']={p.name:file_sha256(p) for p in sorted(args.tokenizer.glob('*.json'))}
    report['performance']={}
    for backend in args.backends:
        record=json.loads((args.out/f'{backend}.json').read_text())
        steps=record['steps']
        decode=[s['perf']['cycles'] for s in steps if s['pos']>=len(ids)]
        report['performance'][backend]={'launches':len(steps),
            'decode_cycles_mean':sum(decode)/len(decode) if decode else None,
            'total_cycles':sum(s['perf']['cycles'] for s in steps)}
    if args.checkpoint:
        from transformers import AutoModelForCausalLM
        import numpy as np
        from .verify import cosine
        model=AutoModelForCausalLM.from_pretrained(args.checkpoint,local_files_only=True,
            trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval()
        with torch.no_grad():
            logits=model(torch.tensor([ids+expected['generated']]),use_cache=False).logits[0].numpy()
        e=json.loads((args.qgraph/'qgraph.json').read_text())['model']['E_LOGIT']
        rows=[]
        for step,row in zip(expected['steps'],expected['logits_last_rows'],strict=True):
            fp=logits[step['pos']+step['valid_rows']-1]
            quant=np.asarray(row,dtype=np.float64)*2.0**e
            rows.append({'cosine':cosine(quant,fp),'top1_match':bool(np.argmax(quant)==np.argmax(fp))})
        report['fp32_teacher_forced']={'rows':rows,'top1_agreement':float(np.mean([r['top1_match'] for r in rows])),
            'mean_cosine':float(np.mean([r['cosine'] for r in rows]))}
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--threads',type=int,default=2)
    sub=ap.add_subparsers(dest='command',required=True)
    check=sub.add_parser('inspect',help='Check a local config without loading weights or claiming execution support')
    check.add_argument('checkpoint',type=Path)
    check.add_argument('--context',type=int,default=32)
    e=sub.add_parser('export');e.add_argument('checkpoint',type=Path)
    e.add_argument('--out',type=Path,required=True);e.add_argument('--context',type=int,default=32)
    e.add_argument('--calibration',type=Path,required=True)
    e.add_argument('--calibration-sequences',type=int,default=64)
    e.add_argument('--calibration-sampling', choices=['line-prefix','uniform-windows'], default='line-prefix',
                   help='uniform-windows disperses fixed-length token windows across the full corpus')
    e.add_argument('--smoothquant-alpha',type=float,default=None)
    e.add_argument('--smoothquant-auto-alpha',action='store_true',help='experimental: select per-group alpha using a local calibration proxy; verify end-to-end quality')
    e.add_argument('--optimize-quantization',action='store_true',help='experimental: select int8/int16 bounds by local calibration MSE; can worsen language quality')
    e.add_argument('--calibration-length',type=int,help='calibration sequence length, independent of larger compiled context')
    e.add_argument('--skip-float-check',action='store_true',help='explicitly omit independent HF import comparison')
    r=sub.add_parser('run');r.add_argument('image',type=Path);r.add_argument('--qgraph',type=Path,required=True)
    r.add_argument('--tokenizer',type=Path,required=True);r.add_argument('--checkpoint',type=Path)
    r.add_argument('--simulator',type=Path,default=Path('build/dma32/llaccel-sim'))
    r.add_argument('--prompt',default='Hello');r.add_argument('--tokens',type=int,default=2)
    r.add_argument('--backends',nargs='+',choices=['func','rtl'],default=['func'])
    r.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.threads < 1:ap.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.command == 'inspect':
        from .hf_import import _json
        from .hf_support import inspect_config
        path = args.checkpoint / 'config.json' if args.checkpoint.is_dir() else args.checkpoint
        report = inspect_config(_json(path), args.context)
        print(json.dumps(report, indent=2))
        if report['status'] != 'CONFIG_COMPATIBLE':
            raise SystemExit(1)
        return
    (export_checkpoint if args.command=='export' else run_checkpoint)(args)

if __name__=='__main__':main()
