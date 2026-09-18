#!/usr/bin/env python3
"""Preinitialized-state boundary tests, complementary to end-to-end RTL prefill.

Prompt IDs must come from the intended model tokenizer. Golden generates state
from that real prefix; the device executes only the selected final decode.
"""
import argparse
import gc
import sys
import json
from pathlib import Path
import subprocess
import numpy as np
from llaccel.golden import GoldenModel
from regress_pretrained import sha256, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--export', type=Path, help='optional original export whose weights/calibration provenance is recorded')
    ap.add_argument('--image', type=Path, required=True)
    ap.add_argument('--qgraph', type=Path, required=True)
    ap.add_argument('--prompt-ids', type=Path, required=True)
    ap.add_argument('--tool', type=Path, required=True)
    ap.add_argument('--backend', choices=['func', 'rtl'], default='rtl')
    ap.add_argument('--contexts', nargs='+', type=int, default=[255,256,257,511,512,513,1023,1024])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'status.json', {'status':'RUNNING', 'validation_kind':'preinitialized-state boundary test'})
    try:
        if args.threads < 1: raise ValueError('threads must be positive')
        import torch
        torch.set_num_threads(args.threads)
        def input_hashes():
            inputs = {'tool':args.tool, 'image':args.image, 'prompt_ids':args.prompt_ids}
            for label, directory in [('qgraph',args.qgraph), ('export',args.export)]:
                if directory:
                    inputs.update({f'{label}/{p.relative_to(directory)}':p for p in directory.rglob('*') if p.is_file()})
            return {name:sha256(path) for name,path in sorted(inputs.items())}
        initial_hashes = input_hashes()
        golden = GoldenModel(args.qgraph, fast_matmul=True)
        ids = json.loads(args.prompt_ids.read_text())
        contexts = sorted(set(args.contexts))
        if not contexts or contexts[0] < 1 or contexts[-1] > golden.max_seq:
            raise ValueError('contexts outside model capacity')
        if not isinstance(ids,list) or len(ids) < contexts[-1] or any(type(t) is not int or not 0 <= t < golden.vocab for t in ids):
            raise ValueError('prompt IDs must contain a valid token for every tested context position')
        def snapshot(directory, when):
            for layer, pair in golden.kv.items():
                for kind, cache in zip(('k','v'), pair):
                    (directory/f'kv{layer}.{kind}.{when}.bin').write_bytes(cache.tobytes(order='C'))
        cases = []
        pos = 0
        for context in contexts:
            target = context-1
            # Never pad past the boundary: full chunks, then exact single rows.
            while pos < target:
                count = golden.prefill_m if pos+golden.prefill_m <= target else 1
                golden.run_chunk(golden.embed_rows(ids[pos:pos+count],count),pos)
                pos += count
            directory = args.out/f'T{context}'
            directory.mkdir(exist_ok=True)
            snapshot(directory,'before')
            logits = golden.decode(ids[target],target)
            snapshot(directory,'after')
            np.asarray(logits[:golden.vocab],dtype='<i2').tofile(directory/'logits.bin')
            cases.append({'pos':target,'token':ids[target],'directory':directory.name})
            pos = context
            print(f'Prepared independent state T={context}',flush=True)
        provenance = {'tool_sha256':sha256(args.tool), 'image_sha256':sha256(args.image),
                      'prompt_ids_sha256':sha256(args.prompt_ids), 'threads':args.threads,
                      'qgraph_files':{str(p.relative_to(args.qgraph)):sha256(p) for p in sorted(args.qgraph.rglob('*')) if p.is_file()},
                      'fixture_files':{str(p.relative_to(args.out)):sha256(p) for p in sorted(args.out.glob('T*/*.bin'))}}
        if args.export:
            provenance['export_files'] = {str(p.relative_to(args.export)):sha256(p) for p in sorted(args.export.rglob('*')) if p.is_file()}
        provenance['inputs_at_start'] = initial_hashes
        write_json(args.out/'provenance.json', provenance)
        fixture = args.out/'fixtures.json'
        write_json(fixture, {'validation_kind':'preinitialized-state boundary test', 'image_sha256':sha256(args.image),
                            'prompt_ids_sha256':sha256(args.prompt_ids),'cases':cases})
        # Release mapped weights and activation state before the device process.
        del golden
        gc.collect()
        command = [args.tool,args.image,fixture,args.backend,args.out/'results.json']
        with (args.out/'device.log').open('w') as log:
            subprocess.run(list(map(str,command)),stdout=log,stderr=subprocess.STDOUT,check=True)
        result = json.loads((args.out/'results.json').read_text())
        if (result.get('status') != 'PASS' or result.get('validation_kind') != 'preinitialized-state boundary test'
                or [c.get('context') for c in result.get('cases',[])] != contexts
                or any(c.get('result') != 'MATCH' or c.get('backend') != args.backend for c in result['cases'])):
            raise ValueError('incomplete device verification')
        final_hashes = input_hashes()
        if final_hashes != initial_hashes: raise ValueError('input artifacts changed during boundary campaign')
        provenance['inputs_at_end'] = final_hashes
        write_json(args.out/'provenance.json', provenance)
        write_json(args.out/'status.json', {'status':'PASS','validation_kind':'preinitialized-state boundary test',
                  'cases':len(contexts),'backend':args.backend,'tool_sha256':sha256(args.tool),'image_sha256':sha256(args.image)})
        print(f'PASS {len(contexts)} preinitialized-state boundary tests ({args.backend})')
    except BaseException as error:
        try:
            write_json(args.out/'status.json',{'status':'FAILED','error':f'{type(error).__name__}: {error}'})
        except OSError as status_error:
            print(f'Could not persist failed status: {status_error}', file=sys.stderr)
        raise

if __name__ == '__main__':
    main()
