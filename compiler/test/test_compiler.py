"""Portable compiler regressions; fixtures use only the Python standard library."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

BIN = Path(sys.argv.pop(1)).resolve()
MODEL = '''module attributes {llaccel.model = {dim = 16 : i64, n_layers = 1 : i64,
 n_heads = 1 : i64, n_kv_heads = 1 : i64, head_dim = 16 : i64,
 ffn = 16 : i64, vocab = 17 : i64, max_seq = 16 : i64,
 rope_base = 10000.0 : f64, rms_eps = 1.0e-5 : f64}} {
 llaccel.weight @embed : tensor<17x16xf32>
 llaccel.weight @head : tensor<17x16xf32>
 func.func @forward(%x: tensor<?x16xf32> {llaccel.name = "x"}) -> tensor<?x17xf32> {
 %y = llaccel.linear %x, @head {llaccel.name = "logits"} : tensor<?x16xf32> -> tensor<?x17xf32>
 return %y : tensor<?x17xf32>
 }
}'''

class CompilerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'model.mlir').write_text(MODEL)
        (self.root / 'weights.bin').write_bytes(struct.pack('<544f', *([0.01] * 544)))
        (self.root / 'weights.json').write_text(json.dumps([
            {'name': 'embed', 'shape': [17, 16], 'offset': 0},
            {'name': 'head', 'shape': [17, 16], 'offset': 1088}]))
        (self.root / 'calib.json').write_text('{"x": 1.0, "logits": 1.0}')
        (self.root / 'tokenizer.json').write_text('{}')

    def run_tool(self, tool, *args, success=True):
        p = subprocess.run([str(BIN / tool), *map(str, args)], cwd=self.root,
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0 if success else 1, p.stderr)
        return p

    def compile(self, *args, success=True):
        return self.run_tool('llaccel-compile', 'model.mlir', '--weights', 'weights.bin',
            '--weights-json', 'weights.json', '--calib', 'calib.json', '-o', 'model.llbin',
            *args, success=success)

    def test_explicit_quantization_bounds_preserve_raw_range(self):
        (self.root / 'calib.json').write_text(json.dumps({
            'x': 8.0, 'x.i8_absmax': 4.0, 'x.i16_absmax': 4.0,
            'logits': 8.0, 'logits.i16_absmax': 4.0}))
        self.compile('--dump-qgraph', 'qgraph')
        graph = json.loads((self.root / 'qgraph/qgraph.json').read_text())
        self.assertEqual(graph['exps']['x'], -12)
        self.assertAlmostEqual(graph['scales']['x.q'], 4.0 / 127)
        self.assertEqual(graph['model']['E_LOGIT'], -12)
        for invalid in [0.0, 9.0]:
            (self.root / 'calib.json').write_text(json.dumps({
                'x': 8.0, 'x.i8_absmax': invalid, 'logits': 8.0}))
            result = self.compile(success=False)
            self.assertIn('invalid selected quantization bound', result.stderr)

    def test_weight_syntax_roundtrip(self):
        p = self.run_tool('llaccel-opt', 'model.mlir')
        (self.root / 'roundtrip.mlir').write_text(p.stdout)
        self.run_tool('llaccel-opt', 'roundtrip.mlir')

    def test_compile_matrix_and_chunking(self):
        for target in ['v1', 'v2']:
            for schedule in ['inorder', 'overlap']:
                args = ['--target', 'llaccel-' + target, '--schedule', schedule,
                        '--weight-chunk-bytes', '256', '--dump-qgraph', 'qgraph']
                if target == 'v2': args += ['--enable-fusion']
                self.compile(*args)
                data = (self.root / 'model.llbin').read_bytes()
                magic, version, count = struct.unpack_from('<III', data)
                self.assertEqual((magic, version, count), (0x4E424C4C, 2, 4))
                programs = []
                for i in range(count):
                    kind, flags, off, size = struct.unpack_from('<IIQQ', data, 12 + i * 24)
                    self.assertLessEqual(off + size, len(data))
                    if kind == 2: programs.append(flags)
                self.assertEqual(sorted(programs), [1, 16])
                self.run_tool('llaccel-disasm', 'model.llbin')
                self.assertTrue((self.root / 'qgraph/tokenizer.json').is_file())

    def test_configurable_prefill_programs_and_metadata(self):
        for rows in [1, 4]:
            self.compile('--prefill-m', str(rows), '--dump-qgraph', 'qgraph')
            graph = json.loads((self.root / 'qgraph/qgraph.json').read_text())
            self.assertEqual(graph['model']['prefill_m'], rows)
            data = (self.root / 'model.llbin').read_bytes()
            count = struct.unpack_from('<I', data, 8)[0]
            shapes = [struct.unpack_from('<IIQQ', data, 12 + i * 24)[:2]
                      for i in range(count)]
            self.assertEqual(sorted(flags for kind, flags in shapes if kind == 2),
                             sorted(set([1, rows])))

    def test_rejects_prefill_past_context_storage(self):
        for rows in [0, 17, 3]:
            self.compile('--prefill-m', str(rows), success=False)
        (self.root / 'model.mlir').write_text(MODEL.replace('max_seq = 16', 'max_seq = 17'))
        result = self.compile(success=False)
        self.assertIn('max_seq must be divisible by prefill-m', result.stderr)
        self.compile('--prefill-m', '1')

    def test_rejects_invalid_configuration(self):
        self.compile('--enable-fusion', success=False)
        self.compile('--schedule', 'bad', success=False)
        self.compile('--weight-chunk-bytes', '1', success=False)
        self.compile('--sram-bytes', '256', success=False)

    def test_rejects_invalid_model_dimensions(self):
        (self.root / 'model.mlir').write_text(MODEL.replace('n_heads = 1', 'n_heads = 0'))
        p = self.compile(success=False)
        self.assertIn('invalid dimensions', p.stderr)

    def test_rejects_bad_weight_ranges_and_nonfinite_data(self):
        index = json.loads((self.root / 'weights.json').read_text())
        for offset in [-4, 1, 10**15]:
            index[0]['offset'] = offset
            (self.root / 'weights.json').write_text(json.dumps(index))
            self.compile(success=False)
        index[0]['offset'] = 0
        index[0]['shape'] = [2**62, 16]
        (self.root / 'weights.json').write_text(json.dumps(index))
        self.compile(success=False)
        (self.root / 'weights.bin').write_bytes(struct.pack('<f', float('nan')))
        self.compile(success=False)

    def test_rejects_bad_calibration(self):
        (self.root / 'calib.json').write_text('{"x": -1.0, "logits": 1.0}')
        self.compile(success=False)

    def test_rope_projection_reserves_rotated_headroom(self):
        model = MODEL.replace(' llaccel.weight @head',
            ' llaccel.weight @proj : tensor<16x16xf32>\n llaccel.weight @head')
        model = model.replace(' %y = llaccel.linear %x, @head',
            ' %q = llaccel.linear %x, @proj {llaccel.name = "q"} : tensor<?x16xf32> -> tensor<?x16xf32>\n'
            ' %r = llaccel.rope %q {heads = 1 : i64, llaccel.name = "qr"} : tensor<?x16xf32>\n'
            ' %y = llaccel.linear %r, @head')
        (self.root / 'model.mlir').write_text(model)
        binary = (self.root / 'weights.bin').read_bytes()
        (self.root / 'weights.bin').write_bytes(binary + struct.pack('<256f', *([0.01] * 256)))
        index = json.loads((self.root / 'weights.json').read_text())
        index.append({'name': 'proj', 'shape': [16, 16], 'offset': len(binary)})
        (self.root / 'weights.json').write_text(json.dumps(index))
        (self.root / 'calib.json').write_text('{"x": 1.0, "q": 1.0, "qr": 8.0, "logits": 1.0}')
        self.compile('--dump-qgraph', 'qgraph')
        graph = json.loads((self.root / 'qgraph/qgraph.json').read_text())
        self.assertEqual(graph['exps']['q'], -11)
        self.assertEqual(graph['exps']['qr'], -11)

    def test_disasm_rejects_truncation(self):
        self.compile()
        data = (self.root / 'model.llbin').read_bytes()
        for size in [0, 11, 12, 35, 100]:
            (self.root / 'truncated.llbin').write_bytes(data[:size])
            self.run_tool('llaccel-disasm', 'truncated.llbin', success=False)

if __name__ == '__main__':
    unittest.main()
