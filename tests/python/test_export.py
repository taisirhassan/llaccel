"""(c) exporter: op sequence / names / strict-syntax structural check, HF-style GQA variant, error reporting."""
from __future__ import annotations

import json
import re

import pytest
import torch
import torch.nn.functional as F

from llaccel import export as E
from llaccel.model import Attention, ModelConfig, TinyLlama, apply_rotary

LAYER_SEQ = ["rmsnorm:h", "linear:q", "linear:k", "linear:v", "rope:qr", "rope:kr", "attention:a", "linear:o", "add:x1",
             "rmsnorm:h2", "linear:g", "linear:u", "silu:sg", "mul:f", "linear:d", "add:x2"]

OP_RE = re.compile(r'^\s{4}%(?P<res>[A-Za-z_][\w]*) = llaccel\.(?P<op>\w+) (?P<operands>[^{]*?)\s*\{(?P<attrs>.*)\} : (?P<ty>.+)$')
WEIGHT_RE = re.compile(r"^  llaccel\.weight @(?P<name>\w+) : tensor<(?P<shape>[\dx]+)xf32>$")
HDR_RE = re.compile(r'^  func\.func @forward\(%x: tensor<\?x(\d+)xf32> \{llaccel\.name = "input"\}\) -> tensor<\?x(\d+)xf32> \{$')


def check_mlir(text: str) -> dict:
    """A structural mini-checker for DIALECT.md section-1 syntax (the real parser is the C++ compiler)."""
    lines = text.splitlines()
    assert lines[0].startswith("module attributes {llaccel.model = {")
    attrs = {}
    hdr = " ".join(lines[0:3])
    for k, v, ty in re.findall(r"(\w+) = ([-+\d.eE]+) : (i64|f64)", hdr):
        attrs[k] = int(v) if ty == "i64" else float(v)
        if ty == "f64":
            assert re.fullmatch(r"[-+]?\d+\.\d*([eE][-+]?\d+)?", v), f"f64 literal {v!r} needs a '.'"
    assert set(attrs) == {"dim", "n_layers", "n_heads", "n_kv_heads", "head_dim", "ffn", "vocab", "max_seq", "rope_base", "rms_eps"}
    weights, ops, names = {}, [], []
    defined = {"x"}
    i = 3
    while (m := WEIGHT_RE.match(lines[i])):
        weights[m["name"]] = [int(s) for s in m["shape"].split("x")]
        i += 1
    assert HDR_RE.match(lines[i]), lines[i]
    i += 1
    while not lines[i].startswith("    return "):
        m = OP_RE.match(lines[i])
        assert m, f"line does not parse as an llaccel op: {lines[i]!r}"
        res, op = m["res"], m["op"]
        assert res not in defined, f"SSA value %{res} redefined"
        nm = re.search(r'llaccel\.name = "([^"]+)"', m["attrs"])
        assert nm, f"missing llaccel.name on {lines[i]!r}"
        names.append(nm.group(1))
        for tok in [t.strip() for t in m["operands"].split(",") if t.strip()]:
            if tok.startswith("%"):
                assert tok[1:] in defined, f"use of undefined value {tok} in {lines[i]!r}"
            elif tok.startswith("@"):
                assert tok[1:] in weights, f"undeclared weight {tok} in {lines[i]!r}"
            else:
                raise AssertionError(f"bad operand {tok!r}")
        for a in re.findall(r"(\w+) = (\S+) : (i64|f64)", m["attrs"]):
            if a[2] == "i64":
                int(a[1])
        defined.add(res)
        ops.append((op, nm.group(1)))
        i += 1
    ret = re.fullmatch(r"    return %(\w+) : tensor<\?x(\d+)xf32>", lines[i])
    assert ret and ret.group(1) in defined
    assert lines[i + 1] == "  }" and lines[i + 2] == "}"
    assert len(set(names)) == len(names), "llaccel.name values must be unique"
    return {"attrs": attrs, "weights": weights, "ops": ops, "names": names}


def expected_ops(n_layers: int) -> list[tuple[str, str]]:
    out = []
    for i in range(n_layers):
        out += [(s.split(":")[0], f"l{i}.{s.split(':')[1]}") for s in LAYER_SEQ]
    return out + [("rmsnorm", "hn"), ("linear", "logits")]


@pytest.mark.parametrize("cfg", [ModelConfig(), ModelConfig(qkv_bias=True, n_layers=1)], ids=["default", "bias_1layer"])
def test_export_structure(cfg, tmp_path):
    torch.manual_seed(0)
    model = TinyLlama(cfg).eval()
    m = E.import_model(model, cfg)
    text = E.emit_mlir(m)
    info = check_mlir(text)
    assert info["ops"] == expected_ops(cfg.n_layers)
    assert info["attrs"]["n_layers"] == cfg.n_layers and info["attrs"]["rope_base"] == cfg.rope_base
    assert info["attrs"]["rms_eps"] == cfg.rms_eps and info["attrs"]["vocab"] == cfg.vocab
    # weights: [N][K] like nn.Linear.weight; bias only when configured
    assert info["weights"]["embed"] == [cfg.vocab, cfg.dim]
    assert info["weights"]["l0_wq"] == [cfg.dim, cfg.dim] and info["weights"]["l0_wk"] == [cfg.n_kv_heads * cfg.head_dim, cfg.dim]
    assert info["weights"]["l0_wd"] == [cfg.dim, cfg.ffn] and info["weights"]["lm_head"] == [cfg.vocab, cfg.dim]
    assert ("l0_bq" in info["weights"]) == cfg.qkv_bias
    assert ("l0_bo" not in info["weights"]) and ("lm_head_bias" not in info["weights"])
    q_line = next(l for l in text.splitlines() if 'llaccel.name = "l0.q"' in l)
    if cfg.qkv_bias:
        assert "@l0_wq, @l0_bq {" in q_line
    else:
        assert "@l0_wq {" in q_line
    # the calibration name set is exactly the set of op results + input
    assert set(m.node_names.values()) == set(info["names"]) | {"input"}
    # weights.bin / weights.json round-trip
    idx = E.write_weights(m, tmp_path)
    blob = (tmp_path / "weights.bin").read_bytes()
    sd = model.state_dict()
    import numpy as np
    for e in idx:
        n = int(np.prod(e["shape"]))
        arr = np.frombuffer(blob, dtype="<f4", count=n, offset=e["offset"]).reshape(e["shape"])
        if e["name"] == "l0_wq":
            assert np.array_equal(arr, sd["layers.0.self_attn.q_proj.weight"].numpy())
        if e["name"] == "embed":
            assert np.array_equal(arr, sd["embed_tokens.weight"].numpy())
    assert sorted(json.loads(json.dumps(idx))[0].keys()) == ["name", "offset", "shape"]


class HFStyleAttention(Attention):
    """HF `repeat_kv` (unsqueeze/expand/reshape) instead of repeat_interleave, mm/addmm-free but different plumbing."""

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        k = k[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)
        v = v[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(a.transpose(1, 2).contiguous().view(B, T, -1))


def test_hf_style_gqa_layout_is_recognized():
    cfg = ModelConfig(n_layers=1)
    torch.manual_seed(1)
    ref = TinyLlama(cfg).eval()
    alt = TinyLlama(cfg).eval()
    alt.load_state_dict(ref.state_dict())
    for blk in alt.layers:
        hf = HFStyleAttention(cfg)
        hf.load_state_dict(blk.self_attn.state_dict())
        blk.self_attn = hf
    tok = torch.randint(0, cfg.vocab, (1, 12))
    assert torch.allclose(ref(tok), alt(tok), atol=1e-6)
    assert E.emit_mlir(E.import_model(alt, cfg)) == E.emit_mlir(E.import_model(ref, cfg))


def test_wrong_head_split_is_rejected():
    class BadAttention(Attention):
        def forward(self, x, cos, sin):
            B, T, _ = x.shape
            # head split in the wrong order: columns are (D, H) instead of (H, D)
            q = self.q_proj(x).view(B, T, self.head_dim, self.n_heads).permute(0, 3, 1, 2)
            k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
            k, v = k.repeat_interleave(self.n_rep, 1), v.repeat_interleave(self.n_rep, 1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o_proj(a.transpose(1, 2).reshape(B, T, -1))

    cfg = ModelConfig(n_layers=1)
    m = TinyLlama(cfg).eval()
    m.layers[0].self_attn = BadAttention(cfg)
    with pytest.raises(E.ImportError_, match="standard head split"):
        E.import_model(m, cfg)


def test_unrecognized_op_reports_node():
    cfg = ModelConfig(n_layers=1)

    class Weird(TinyLlama):
        def forward(self, tokens):
            x = self.embed_tokens(tokens)
            x = torch.tanh(x)
            T = tokens.shape[1]
            for layer in self.layers:
                x = layer(x, self.rope_cos[:T], self.rope_sin[:T])
            return self.lm_head(self.norm(x))

    with pytest.raises(E.ImportError_) as ei:
        E.import_model(Weird(cfg).eval(), cfg)
    assert "unrecognized op" in str(ei.value) and "aten.tanh" in str(ei.value)


def test_mlir_float_literals():
    assert E.mlir_float(1e-05) == "1.0e-05" and E.mlir_float(10000.0) == "10000.0" and E.mlir_float(1e-6) == "1.0e-06"
    assert E.mlir_float(0.5) == "0.5"
