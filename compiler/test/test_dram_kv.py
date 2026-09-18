"""Compile long-context attention without allocating the KV cache in SRAM."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
BIN=Path(sys.argv.pop(1)).resolve()

class DramKVTest(unittest.TestCase):
    def test_context_and_cache_placement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            model='''module attributes {llaccel.model = {dim = 64 : i64,
n_layers = 1 : i64, n_heads = 1 : i64, n_kv_heads = 1 : i64,
head_dim = 64 : i64, ffn = 64 : i64, vocab = 17 : i64,
max_seq = CONTEXT : i64, rope_base = 10000.0 : f64, rms_eps = 1.0e-5 : f64}} {
llaccel.weight @embed : tensor<17x64xf32>
llaccel.weight @wq : tensor<64x64xf32>
llaccel.weight @wk : tensor<64x64xf32>
llaccel.weight @wv : tensor<64x64xf32>
llaccel.weight @head : tensor<17x64xf32>
func.func @forward(%x: tensor<?x64xf32> {llaccel.name = "input"}) -> tensor<?x17xf32> {
%q = llaccel.linear %x, @wq {llaccel.name = "q"} : tensor<?x64xf32> -> tensor<?x64xf32>
%k = llaccel.linear %x, @wk {llaccel.name = "k"} : tensor<?x64xf32> -> tensor<?x64xf32>
%v = llaccel.linear %x, @wv {llaccel.name = "v"} : tensor<?x64xf32> -> tensor<?x64xf32>
%qr = llaccel.rope %q {heads = 1 : i64, llaccel.name = "qr"} : tensor<?x64xf32>
%kr = llaccel.rope %k {heads = 1 : i64, llaccel.name = "kr"} : tensor<?x64xf32>
%a = llaccel.attention %qr, %kr, %v {layer = 0 : i64, heads = 1 : i64, kv_heads = 1 : i64, head_dim = 64 : i64, llaccel.name = "a"} : (tensor<?x64xf32>, tensor<?x64xf32>, tensor<?x64xf32>) -> tensor<?x64xf32>
%lg = llaccel.linear %a, @head {llaccel.name = "logits"} : tensor<?x64xf32> -> tensor<?x17xf32>
return %lg : tensor<?x17xf32>
}
}'''
            data=bytearray();index=[]
            for name,rows in [('embed',17),('wq',64),('wk',64),('wv',64),('head',17)]:
                index.append({'name':name,'shape':[rows,64],'offset':len(data)})
                data.extend(struct.pack('<f',0.01)*(rows*64))
            (root/'weights.bin').write_bytes(data)
            (root/'weights.json').write_text(json.dumps(index))
            (root/'calib.json').write_text(json.dumps(dict.fromkeys(
                ['input','q','k','v','qr','kr','a','logits'],1.0)))
            for context in (1024,4096):
                (root/'model.mlir').write_text(model.replace('CONTEXT',str(context)))
                for rows in (1,16):
                    for schedule in ('inorder','overlap'):
                        cmd=[str(BIN/'llaccel-compile'),'model.mlir','--weights','weights.bin',
                             '--weights-json','weights.json','--calib','calib.json','-o','model.llbin',
                             '--prefill-m',str(rows),'--schedule',schedule]
                        p=subprocess.run(cmd,cwd=root,capture_output=True,text=True)
                        self.assertEqual(p.returncode,0,p.stderr)
                        binary=(root/'model.llbin').read_bytes()
                        self.assertEqual(struct.unpack_from('<I',binary,4)[0],2)
                        sections=[struct.unpack_from('<IIQQ',binary,12+24*i)
                                  for i in range(struct.unpack_from('<I',binary,8)[0])]
                        for kind,flags,off,size in sections:
                            if kind==1: dram=binary[off:off+size]
                            if kind==3: meta=json.loads(binary[off:off+size])
                        self.assertEqual(meta['sram']['kv_cache_bytes'],0)
                        self.assertLess(meta['sram']['peak_used'],1<<20)
                        self.assertEqual(meta['dram']['kv_cache_bytes'],2*context*64)
                        cache=meta['dram']['kv_cache']
                        self.assertEqual(len(cache),2)
                        for region in cache:
                            self.assertEqual(region['addr']%64,0)
                            self.assertEqual(region['bytes'],context*64)
                            self.assertFalse(any(dram[region['addr']:region['addr']+region['bytes']]))
                        addresses={region['addr'] for region in cache}
                        for kind,flags,off,size in sections:
                            if kind!=2:continue
                            for pc in range(off,off+size,64):
                                words=struct.unpack_from('<16I',binary,pc)
                                if words[0]&255==0x40:
                                    self.assertEqual({words[4],words[5]},addresses)
                                    self.assertEqual(words[10],context*64)
                                if words[0]&255==0x41:
                                    self.assertIn(words[3],addresses)
            (root/'model.mlir').write_text(model.replace('CONTEXT','4097'))
            p=subprocess.run(cmd,cwd=root,capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)

if __name__=='__main__':unittest.main()
