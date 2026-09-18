"""SRAM scaling regressions, with a vocabulary larger than the device SRAM."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

BIN = Path(sys.argv.pop(1)).resolve()
SIM = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else None


class StreamingTest(unittest.TestCase):
    def test_large_vocabulary_streams_output_and_constants(self):
        # RQ alone is >1MiB; full 16-row output is >4MiB. Both must stream.
        vocab = 151937
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = f'''module attributes {{llaccel.model = {{dim = 16 : i64,
n_layers = 1 : i64, n_heads = 1 : i64, n_kv_heads = 1 : i64,
head_dim = 16 : i64, ffn = 16 : i64, vocab = {vocab} : i64,
max_seq = 16 : i64, rope_base = 10000.0 : f64, rms_eps = 1.0e-5 : f64}}}} {{
llaccel.weight @embed : tensor<{vocab}x16xf32>
llaccel.weight @head : tensor<{vocab}x16xf32>
llaccel.weight @bias : tensor<{vocab}xf32>
func.func @forward(%x: tensor<?x16xf32> {{llaccel.name = "x"}}) -> tensor<?x{vocab}xf32> {{
%y = llaccel.linear %x, @head, @bias {{llaccel.name = "logits"}} : tensor<?x16xf32> -> tensor<?x{vocab}xf32>
return %y : tensor<?x{vocab}xf32>
}}
}}'''
            (root / 'model.mlir').write_text(module)
            matrix = struct.pack('<f', 0.01) * (vocab * 16)
            head = b''.join(struct.pack('<f', 0.01 * (1 + n % 3)) * 16 for n in range(vocab))
            bias = b''.join(struct.pack('<f', 0.01 * (n % 7)) for n in range(vocab))
            (root / 'weights.bin').write_bytes(matrix + head + bias)
            (root / 'weights.json').write_text(json.dumps([
                {'name': 'embed', 'shape': [vocab, 16], 'offset': 0},
                {'name': 'head', 'shape': [vocab, 16], 'offset': len(matrix)},
                {'name': 'bias', 'shape': [vocab], 'offset': 2 * len(matrix)}]))
            (root / 'calib.json').write_text('{"x": 1.0, "logits": 1.0}')
            tokenizer = {'itos': [chr(0x10000 + i) for i in range(vocab)]}
            (root / 'tokenizer.json').write_text(json.dumps(tokenizer))
            for rows in (1, 4, 16):
                for schedule in ('inorder', 'overlap'):
                    p = subprocess.run([str(BIN / 'llaccel-compile'), 'model.mlir',
                        '--weights', 'weights.bin', '--weights-json', 'weights.json',
                        '--calib', 'calib.json', '-o', 'model.llbin', '--prefill-m', str(rows),
                        '--schedule', schedule, '--sram-bytes', '524288', '--dump-qgraph', 'qgraph'],
                        cwd=root, text=True, capture_output=True)
                    self.assertEqual(p.returncode, 0, p.stderr)
                    data = (root / 'model.llbin').read_bytes()
                    count = struct.unpack_from('<I', data, 8)[0]
                    for i in range(count):
                        kind, _, off, size = struct.unpack_from('<IIQQ', data, 12 + 24 * i)
                        if kind == 3:
                            meta = json.loads(data[off:off + size])
                    self.assertLess(meta['sram']['peak_used'], 524288)
                    self.assertLess(meta['sram']['resident_const_bytes'], 4096)
                    self.assertGreater(meta['dram']['logits']['row_bytes'] * rows, 0)
                    if SIM:
                        # Independent scalar oracle over the emitted quantized constants.
                        # Vary channel scales/biases to catch wrong per-chunk DMA offsets.
                        graph = json.loads((root / 'qgraph/qgraph.json').read_text())
                        blob = (root / 'qgraph/qweights.bin').read_bytes()
                        def scalar(name, fmt, extra=0):
                            return struct.unpack_from(fmt, blob, graph['tensors'][name]['offset'] + extra)[0]
                        def rounded(x, shift):
                            return (x + (1 << (shift - 1))) >> shift if shift else x
                        quant = graph['ops'][0]
                        a = max(-128, min(127, rounded(scalar('embed', '<h') * quant['M'], quant['S'])))
                        expected = []
                        for n in range(vocab):
                            acc = 16 * a * scalar('head', '<b', n * 16) + scalar('bias', '<i', n * 4)
                            expected.append(max(-32768, min(32767, rounded(
                                acc * scalar('head.rq', '<i', n * 8), scalar('head.rq', '<i', n * 8 + 4)))))
                        winner = expected.index(max(expected))
                        prompt_len = min(rows, 15)
                        launches = 1 if rows == 16 else 2
                        reference = {'prompt_tokens': [0] * prompt_len, 'generated': [winner],
                                     'argmax_per_step': [winner] * launches,
                                     'logits_last_rows': [expected] * launches}
                        (root / 'reference.json').write_text(json.dumps(reference))
                        for seed in (None, 7):
                            cmd = [str(SIM), 'model.llbin', '--backend', 'func',
                                   '--prompt', tokenizer['itos'][0] * prompt_len, '--tokens', '1',
                                   '--verify', 'reference.json', '--quiet']
                            if seed: cmd += ['--interleave', str(seed)]
                            run = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
                            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                            self.assertIn('VERIFY: MATCH', run.stdout + run.stderr)


    def test_many_norm_constants_are_transient(self):
        dim, count, vocab = 512, 96, 17
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensors = [('embed', [vocab, dim]), ('head', [vocab, dim])]
            tensors += [(f'gamma{i}', [dim]) for i in range(count)]
            declarations, index, data = [], [], bytearray()
            for name, shape in tensors:
                declarations.append(f'llaccel.weight @{name} : tensor<{"x".join(map(str, shape))}xf32>')
                size = 1
                for d in shape: size *= d
                index.append({'name': name, 'shape': shape, 'offset': len(data)})
                data.extend(struct.pack('<f', 0.1) * size)
            ops = []
            previous = 'x'
            for i in range(count):
                ops.append(f'%h{i} = llaccel.rmsnorm %{previous}, @gamma{i} {{eps = 1.0e-5 : f64, llaccel.name = "h{i}"}} : tensor<?x512xf32>')
                previous = f'h{i}'
            module = """module attributes {llaccel.model = {dim = 512 : i64,
n_layers = 1 : i64, n_heads = 32 : i64, n_kv_heads = 1 : i64,
head_dim = 16 : i64, ffn = 512 : i64, vocab = 17 : i64,
max_seq = 16 : i64, rope_base = 10000.0 : f64, rms_eps = 1.0e-5 : f64}} {
""" + "\n".join(declarations) + """
func.func @forward(%x: tensor<?x512xf32> {llaccel.name = "x"}) -> tensor<?x17xf32> {
""" + "\n".join(ops) + f"""
%y = llaccel.linear %{previous}, @head {{llaccel.name = "logits"}} : tensor<?x512xf32> -> tensor<?x17xf32>
return %y : tensor<?x17xf32>
}}
}}"""
            (root / 'model.mlir').write_text(module)
            (root / 'weights.bin').write_bytes(data)
            (root / 'weights.json').write_text(json.dumps(index))
            (root / 'calib.json').write_text(json.dumps(dict.fromkeys(
                ['x', 'logits'] + [f'h{i}' for i in range(count)], 1.0)))
            p = subprocess.run([str(BIN / 'llaccel-compile'), 'model.mlir',
                '--weights', 'weights.bin', '--weights-json', 'weights.json',
                '--calib', 'calib.json', '-o', 'model.llbin', '--prefill-m', '1',
                '--schedule', 'overlap', '--sram-bytes', '65536', '--dump-qgraph', 'qgraph'],
                cwd=root, text=True, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            baseline = json.loads((root / 'qgraph/qgraph.json').read_text())['ops'][0]['C']
            calibration = json.loads((root / 'calib.json').read_text())
            calibration['x.rms_min'] = 0.0001
            (root / 'calib.json').write_text(json.dumps(calibration))
            cmd = [str(BIN / 'llaccel-compile'), 'model.mlir', '--weights', 'weights.bin',
                   '--weights-json', 'weights.json', '--calib', 'calib.json',
                   '-o', 'model.llbin', '--prefill-m', '1', '--dump-qgraph', 'qgraph']
            p = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            measured = json.loads((root / 'qgraph/qgraph.json').read_text())['ops'][0]['C']
            self.assertLess(measured, baseline)
            calibration['x.rms_min'] = 2.0
            (root / 'calib.json').write_text(json.dumps(calibration))
            p = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('rms_min', p.stderr)

    def test_residual_exponents_preserve_small_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = """module attributes {llaccel.model = {dim = 16 : i64,
n_layers = 1 : i64, n_heads = 1 : i64, n_kv_heads = 1 : i64,
head_dim = 16 : i64, ffn = 16 : i64, vocab = 16 : i64,
max_seq = 16 : i64, rope_base = 10000.0 : f64, rms_eps = 1.0e-5 : f64}} {
llaccel.weight @embed : tensor<16x16xf32>
llaccel.weight @early : tensor<16x16xf32>
llaccel.weight @late : tensor<16x16xf32>
llaccel.weight @head : tensor<16x16xf32>
func.func @forward(%x: tensor<?x16xf32> {llaccel.name = "x"}) -> tensor<?x16xf32> {
%o = llaccel.linear %x, @early {llaccel.name = "o"} : tensor<?x16xf32> -> tensor<?x16xf32>
%x1 = llaccel.add %x, %o {llaccel.name = "x1"} : tensor<?x16xf32>
%d = llaccel.linear %x1, @late {llaccel.name = "d"} : tensor<?x16xf32> -> tensor<?x16xf32>
%x2 = llaccel.add %x1, %d {llaccel.name = "x2"} : tensor<?x16xf32>
%lg = llaccel.linear %x2, @head {llaccel.name = "logits"} : tensor<?x16xf32> -> tensor<?x16xf32>
return %lg : tensor<?x16xf32>
}
}"""
            (root / 'model.mlir').write_text(module)
            (root / 'weights.bin').write_bytes(struct.pack('<f', 0.01) * 1024)
            (root / 'weights.json').write_text(json.dumps([
                {'name': name, 'shape': [16, 16], 'offset': i * 1024}
                for i, name in enumerate(['embed', 'early', 'late', 'head'])]))
            (root / 'calib.json').write_text(json.dumps({
                'x': 0.01, 'o': 0.01, 'x1': 0.02, 'd': 1024, 'x2': 1024, 'logits': 1.0}))
            for fusion in (False, True):
                cmd = [str(BIN / 'llaccel-compile'), 'model.mlir', '--weights', 'weights.bin',
                       '--weights-json', 'weights.json', '--calib', 'calib.json',
                       '-o', 'model.llbin', '--dump-qgraph', 'qgraph',
                       '--target', 'llaccel-v2', '--schedule', 'overlap']
                if fusion: cmd.append('--enable-fusion')
                p = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stderr)
                graph = json.loads((root / 'qgraph/qgraph.json').read_text())
                self.assertEqual(graph['model']['E_RES'], -15)
                self.assertEqual(graph['exps']['x1'], -15)
                self.assertEqual(graph['exps']['d'], -4)
                self.assertEqual(graph['exps']['x2'], -4)
                adds = [op for op in graph['ops'] if op['op'] == 'add']
                late = next(op for op in adds if op['out'] == 'x2')
                self.assertEqual((late['a'], late['b'], late['sh_b']), ('d', 'x1', 11))
                # Shifted residual ADD must remain explicit even with fusion enabled.
                self.assertEqual(len(adds), 1 if fusion else 2)
                embed = graph['tensors']['embed']
                blob = (root / 'qgraph/qweights.bin').read_bytes()
                self.assertEqual(struct.unpack_from('<h', blob, embed['offset'])[0], 328)


if __name__ == '__main__':
    unittest.main()
