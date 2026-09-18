"""Compiler-only geometry, grouped RMSNorm and explicit RoPE regressions."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

BIN = Path(sys.argv.pop(1)).resolve()

class GeometryTest(unittest.TestCase):
    def compile(self, root, dim=64, D=128, heads=2, norm_heads=None, explicit=True, sin_value=1.):
        q = heads * D
        groups = heads if norm_heads is None else norm_heads
        tables = (f'llaccel.weight @rope_cos_input : tensor<16x{D//2}xf32>\n'
                  f'llaccel.weight @rope_sin_input : tensor<16x{D//2}xf32>\n') if explicit else ''
        model = f'''module attributes {{llaccel.model = {{dim = {dim} : i64,
 n_layers = 1 : i64, n_heads = {heads} : i64, n_kv_heads = 1 : i64,
 head_dim = {D} : i64, ffn = 64 : i64, vocab = 17 : i64,
 max_seq = 16 : i64, rope_base = 10000.0 : f64, rms_eps = 1.0e-5 : f64}}}} {{
llaccel.weight @embed : tensor<17x{dim}xf32>
llaccel.weight @wq : tensor<{q}x{dim}xf32>
llaccel.weight @g : tensor<{D}xf32>
llaccel.weight @head : tensor<17x{q}xf32>
{tables}
func.func @forward(%x: tensor<?x{dim}xf32> {{llaccel.name = "input"}}) -> tensor<?x17xf32> {{
%q = llaccel.linear %x, @wq {{llaccel.name = "q"}} : tensor<?x{dim}xf32> -> tensor<?x{q}xf32>
%n = llaccel.rmsnorm %q, @g {{eps = 1.0e-5 : f64, heads = {groups} : i64, llaccel.name = "n"}} : tensor<?x{q}xf32>
%r = llaccel.rope %n {{heads = {heads} : i64, llaccel.name = "r"}} : tensor<?x{q}xf32>
%lg = llaccel.linear %r, @head {{llaccel.name = "logits"}} : tensor<?x{q}xf32> -> tensor<?x17xf32>
return %lg : tensor<?x17xf32>
}}
}}'''
        (root/'model.mlir').write_text(model)
        specs = [('embed', [17,dim], .01), ('wq', [q,dim], .01), ('g',[D],1.), ('head',[17,q], .01)]
        if explicit: specs += [('rope_cos_input',[16,D//2],0.), ('rope_sin_input',[16,D//2],sin_value)]
        data=bytearray(); index=[]
        for name,shape,value in specs:
            count=1
            for n in shape: count*=n
            index.append(dict(name=name, shape=shape, offset=len(data)))
            data.extend(struct.pack('<f',value)*count)
        (root/'weights.bin').write_bytes(data)
        (root/'weights.json').write_text(json.dumps(index))
        (root/'calib.json').write_text(json.dumps(dict.fromkeys(['input','q','n','r','logits'],1.)))
        return subprocess.run([str(BIN/'llaccel-compile'),'model.mlir','--weights','weights.bin',
          '--weights-json','weights.json','--calib','calib.json','--prefill-m','16',
          '--dump-qgraph','qgraph','-o','model.llbin'],cwd=root,capture_output=True,text=True)

    def test_independent_query_width_and_grouped_norm(self):
        for D in (128,256):
            with self.subTest(D=D), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); result=self.compile(root,D=D)
                self.assertEqual(result.returncode,0,result.stderr)
                graph=json.loads((root/'qgraph/qgraph.json').read_text())
                norm=next(op for op in graph['ops'] if op['op']=='rmsnorm')
                self.assertEqual((norm['K'],norm['heads']),(D,2))
                data=(root/'model.llbin').read_bytes()
                sections=[struct.unpack_from('<IIQQ',data,12+24*i) for i in range(struct.unpack_from('<I',data,8)[0])]
                prefill=next((o,n) for k,m,o,n in sections if k==2 and m==16)
                instructions=[struct.unpack_from('<16I',data,i) for i in range(prefill[0],sum(prefill),64)]
                norms=[i for i in instructions if i[0]&255==0x30]
                self.assertEqual(len(norms),2)
                self.assertTrue(all(i[5]==16 and i[6]==D for i in norms))
                # Explicit quarter-turn data is distinguishable from default cos(0)=1.
                weights=(root/'qgraph/qweights.bin').read_bytes()
                cos=graph['tensors']['rope_cos']; sin=graph['tensors']['rope_sin']
                self.assertEqual(struct.unpack_from('<h',weights,cos['offset'])[0],0)
                self.assertEqual(struct.unpack_from('<h',weights,sin['offset'])[0],16384)

    def test_scaled_rope_range(self):
        for value, valid in ((1.25, True), (-2., True), (2., False), (float('nan'), False)):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                result = self.compile(Path(tmp), sin_value=value)
                self.assertEqual(result.returncode == 0, valid, result.stderr)

    def test_bad_grouping_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=self.compile(Path(tmp),norm_heads=3)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('RMSNorm heads',result.stderr)

    def test_unrepresentable_head_width_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=self.compile(Path(tmp),D=512)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('head_dim',result.stderr)

if __name__=='__main__': unittest.main()
