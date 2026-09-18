#!/usr/bin/env python3
"""Long-context compiler/device equivalence using an existing tiny model export."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
from llaccel.golden import GoldenModel

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path('build/context-regressions'))
    parser.add_argument('--rtl',action='store_true')
    args=parser.parse_args()
    compiler=(args.build/'compiler/bin/llaccel-compile').resolve()
    simulator=(args.build/'llaccel-sim').resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    records=[]
    for context,cases in [(1024,[(257,2),(1023,1),(1024,0)]),(4096,[(4095,1),(4096,0)])]:
        root=args.out/f'context-{context}'
        exported=root/'export'
        exported.mkdir(parents=True,exist_ok=True)
        for name in ('weights.bin','weights.json','calib.json','tokenizer.json'):
            shutil.copy2(args.export/name,exported/name)
        mlir=(args.export/'model.mlir').read_text()
        mlir=re.sub(r'max_seq\s*=\s*\d+',f'max_seq = {context}',mlir)
        (exported/'model.mlir').write_text(mlir)
        shutil.copy2(exported/'tokenizer.json',root/'tokenizer.json')
        image=root/'model.llbin';qgraph=root/'qgraph'
        cmd=[compiler,exported/'model.mlir','--weights',exported/'weights.bin',
             '--weights-json',exported/'weights.json','--calib',exported/'calib.json',
             '--schedule','overlap','--dump-qgraph',qgraph,'-o',image,'--print-stats']
        p=subprocess.run(list(map(str,cmd)),capture_output=True,text=True)
        (root/'compile.log').write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError(p.stdout+p.stderr)
        tokenizer=json.loads((exported/'tokenizer.json').read_text())['itos']
        for prompt_len,tokens in cases:
            prompt_ids=[i%len(tokenizer) for i in range(prompt_len)]
            golden=GoldenModel(qgraph)
            reference=golden.generate(prompt_ids,tokens)
            label=f'p{prompt_len}-t{tokens}'
            ref=root/f'{label}-golden.json';ref.write_text(json.dumps(reference))
            modes=[('func',0),('func',7)]
            if args.rtl:modes.append(('rtl',0))
            for backend,seed in modes:
                run=[simulator,image,'--backend',backend,'--prompt',''.join(tokenizer[i] for i in prompt_ids),
                     '--tokens',tokens,'--verify',ref,'--quiet']
                if seed:run+=['--interleave',seed]
                p=subprocess.run(list(map(str,run)),capture_output=True,text=True)
                (root/f'{label}-{backend}-{seed}.log').write_text(p.stdout+p.stderr)
                if p.returncode or 'VERIFY: MATCH' not in p.stdout+p.stderr:
                    raise RuntimeError(p.stdout+p.stderr)
                records.append({'context':context,'prompt':prompt_len,'tokens':tokens,
                                'backend':backend,'seed':seed,'launches':len(reference['steps']),'result':'MATCH'})
            print(f'PASS context{context} prompt{prompt_len} tokens{tokens}',flush=True)
    (args.out/'results.json').write_text(json.dumps(records,indent=2))
    print(f'PASS {len(records)} context/model runs',flush=True)

if __name__=='__main__':main()
