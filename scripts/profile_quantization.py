#!/usr/bin/env python3
"""Single-prefix, per-operation quantization diagnostics; not a quality benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metrics(actual, reference) -> dict:
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    b = np.asarray(reference, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('metrics require equal, nonempty, finite arrays')
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    cosine = float(a @ b / (an * bn)) if an and bn else float(not an and not bn)
    return {'cosine': cosine, 'relative_l2_error': float(np.linalg.norm(a-b)/max(bn,1e-15)),
            'actual_absmax': float(np.abs(a).max()), 'reference_absmax': float(np.abs(b).max())}


def range_metrics(integer, local_float, bits: int, scale: float) -> dict:
    if bits not in (8,16) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('invalid representation')
    lo, hi = -(1 << (bits-1)), (1 << (bits-1))-1
    x, local = np.asarray(integer), np.asarray(local_float)
    if x.shape != local.shape or not x.size or np.any(x < lo) or np.any(x > hi):
        raise ValueError('invalid integer representation values or shapes')
    return {'bits':bits, 'scale':scale, 'integer_endpoint_occupancy':float(((x==lo)|(x==hi)).mean()),
            'local_float_outside_endpoint_range':float(((local < lo*scale)|(local > hi*scale)).mean())}


def output_bits(op) -> int:
    return 8 if op['op'] in ('quant','attention') or op.get('out_dtype') == 'i8' else 16


def float_operation(op, activations, keys, values, weights, config):
    import torch
    import torch.nn.functional as F
    kind = op['op']
    if kind == 'quant': return activations[op['in']]
    if kind == 'linear':
        w = weights[op['w']]
        x = activations[op['in']]
        # Exported float outputs can omit the compiler's zero-padded vocab tail.
        if w.shape[-1] > x.shape[-1] or w.shape[0] > int(op['N']):
            raise ValueError('float weight shape exceeds quantized operation shape')
        w = F.pad(w,(0,x.shape[-1]-w.shape[-1],0,int(op['N'])-w.shape[0]))
        result = x @ w.T
        if op.get('bias'):
            bias = weights[op['bias']]
            result = result + F.pad(bias,(0,result.shape[-1]-bias.shape[-1]))
        ep = op.get('epilogue','none')
        if ep == 'none': return result
        if ep == 'resadd': return result + activations[op['aux']]
        if ep == 'silu': return F.silu(result)
        if ep == 'mul': return result * activations[op['aux']]
        raise ValueError(f'unsupported epilogue {ep}')
    if kind == 'rmsnorm':
        x = activations[op['in']]
        shape = x.shape
        gamma = weights[op['gamma']]
        heads = int(op.get('heads', 1))
        if shape[-1] != heads * gamma.numel():
            raise ValueError('RMSNorm head geometry mismatch')
        x = x.reshape(-1, gamma.numel())
        return (x * torch.rsqrt(x.square().mean(-1,keepdim=True)+config['rms_eps']) * gamma).reshape(shape)
    if kind == 'rope':
        x = activations[op['in']]; rows = len(x); dim = int(op['D']); half = dim//2
        x = x.reshape(rows,int(op['H']),dim)
        if 'rope_cos_input' in weights and 'rope_sin_input' in weights:
            cos = weights['rope_cos_input'][:rows]
            sin = weights['rope_sin_input'][:rows]
            cos, sin = torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)
        else:
            frequency = torch.outer(torch.arange(rows,dtype=torch.float32),
                                    config['rope_base'] ** (-torch.arange(0,dim,2,dtype=torch.float32)/dim))
            angle = torch.cat([frequency,frequency],-1)
            cos, sin = angle.cos(), angle.sin()
        return (x*cos.unsqueeze(1)+torch.cat([-x[:,:,half:],x[:,:,:half]],-1)*sin.unsqueeze(1)).reshape(rows,-1)
    if kind == 'kv_write':
        keys[int(op['layer'])] = activations[op['k']]
        values[int(op['layer'])] = activations[op['v']]
        return None
    if kind == 'attention':
        h,hkv,dim = int(op['H']),int(op['Hkv']),int(op['D'])
        q = activations[op['q']]; rows = len(q)
        def heads(x,n): return x.reshape(rows,n,dim).transpose(0,1)
        k = heads(keys[int(op['layer'])],hkv).repeat_interleave(h//hkv,0)
        v = heads(values[int(op['layer'])],hkv).repeat_interleave(h//hkv,0)
        return F.scaled_dot_product_attention(heads(q,h),k,v,is_causal=True).transpose(0,1).reshape(rows,-1)
    if kind == 'silu': return F.silu(activations[op['in']])
    if kind == 'mul': return activations[op['a']]*activations[op['b']]
    if kind == 'add': return activations[op['a']]+activations[op['b']]
    raise ValueError(f'unsupported operation {kind}')


def profile(export: Path, qgraph: Path, text: Path, tokens: int, threads: int) -> dict:
    if not 1 <= tokens <= 16 or threads < 1: raise ValueError('tokens must be 1..16 and threads positive')
    import torch
    from llaccel.golden import GoldenModel
    from llaccel.hf import tokenizer_at
    torch.set_num_threads(threads)
    files = {'text':text}
    for label,directory in [('export',export),('qgraph',qgraph)]:
        files.update({f'{label}/{p.relative_to(directory)}':p for p in directory.rglob('*') if p.is_file()})
    hashes = {name:digest(path) for name,path in sorted(files.items())}
    meta = json.loads((export/'hf-import.json').read_text())
    config = meta['model_config']
    golden = GoldenModel(qgraph,fast_matmul=True)
    for key in ('dim','n_layers','n_heads','n_kv_heads','head_dim','vocab'):
        if int(config[key]) != int(golden.q.model[key]): raise ValueError(f'export/qgraph mismatch: {key}')
    for key in ('rms_eps','rope_base'):
        if float(config[key]) != float(golden.q.model[key]): raise ValueError(f'export/qgraph mismatch: {key}')
    blob = np.memmap(export/'weights.bin',dtype='<f4',mode='r')
    weights = {}
    for spec in json.loads((export/'weights.json').read_text()):
        offset = int(spec['offset'])//4; count = math.prod(spec['shape'])
        # Torch receives a writable copy per tensor; never mutate the mapped export.
        weights[spec['name']] = torch.from_numpy(np.array(blob[offset:offset+count].reshape(spec['shape']),copy=True))
    ids = tokenizer_at(export/'hf-tokenizer').encode(text.read_text(),add_special_tokens=True)[:tokens]
    if not ids or len(ids) > golden.max_seq: raise ValueError('empty prefix or prefix exceeds model context')
    if max(ids) >= golden.vocab: raise ValueError('tokenizer ID exceeds model vocabulary')
    integer = {}
    with torch.no_grad():
        golden.run_chunk(golden.embed_rows(ids,len(ids)),0,integer)
        dequant = {}
        for name,value in integer.items():
            scale = golden.q.scales[name] if name in golden.q.scales else 2.**golden.q.exps[name]
            dequant[name] = torch.from_numpy(np.asarray(value,dtype=np.float32).copy())*scale
        floating = {golden.q.input_name:weights['embed'][ids]}
        float_keys,float_values,local_keys,local_values = {},{},{},{}
        records = []
        for op in golden.q.ops:
            ideal = float_operation(op,floating,float_keys,float_values,weights,config)
            local = float_operation(op,dequant,local_keys,local_values,weights,config)
            if ideal is None: continue
            name = op['out']; floating[name] = ideal
            bits = output_bits(op)
            scale = golden.q.scales[name] if name in golden.q.scales else 2.**golden.q.exps[name]
            records.append({'name':name,'op':op['op'],'epilogue':op.get('epilogue','none'),
                            'local':metrics(dequant[name].numpy(),local.numpy()),
                            'global':metrics(dequant[name].numpy(),ideal.numpy()),
                            'representation':range_metrics(integer[name],local.numpy(),bits,scale)})
    if hashes != {name:digest(path) for name,path in sorted(files.items())}:
        raise ValueError('input files changed during profiling')
    return {'scope':'single-prefix diagnostic at position zero; not a model quality benchmark',
            'requested_tokens':tokens,'actual_tokens':len(ids),'token_ids':ids,'threads':threads,
            'input_sha256':hashes,'records':records,
            'interpretation':{'local':'Dequantized integer output versus float operation on dequantized integer inputs and float weights; includes weight quantization, fixed-point math and output rounding.',
                              'global':'Dequantized integer output versus the fully floating exported graph on this same prefix.',
                              'integer_endpoint_occupancy':'Fraction equal to the minimum/maximum integer code; not proof of clipping.',
                              'local_float_outside_endpoint_range':'Fraction of local float outputs beyond scaled integer endpoints; not a measured RTL clipping rate.'}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('export','qgraph','text','out'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--tokens',type=int,default=8,choices=range(1,17))
    ap.add_argument('--threads',type=int,default=2)
    args = ap.parse_args()
    if args.threads < 1: ap.error('--threads must be positive')
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    result = profile(args.export,args.qgraph,args.text,args.tokens,args.threads)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); temporary.replace(args.out)
    print(f'Wrote {len(result["records"])} operation diagnostics for {result["actual_tokens"]} prefix tokens to {args.out}')

if __name__ == '__main__': main()
