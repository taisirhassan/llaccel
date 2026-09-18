"""Activation calibration: per-tensor abs-max for every `llaccel.name`.

    uv run python -m llaccel.calibrate checkpoints/tiny.pt -o build/export/calib.json

Runs the *exported* graph (so the recorded tensors are exactly the values the
llaccel ops in model.mlir denote) with an FX interpreter over N sequences of
training data and records max |x| at every node that carries a `llaccel.name`
(including the embedding output `input`, the RoPE outputs, the attention output
and the logits).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.fx as fx

from .model import ModelConfig, load_checkpoint


def placeholder_args(ep: torch.export.ExportedProgram, user_args: tuple) -> list:
    """Positional args for `Interpreter(ep.graph_module).run(*args)` in placeholder order."""
    from torch.export.graph_signature import InputKind
    args, it = [], iter(user_args)
    for spec in ep.graph_signature.input_specs:
        if spec.kind == InputKind.USER_INPUT:
            args.append(next(it))
        elif spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
            t = ep.state_dict.get(spec.target)
            if t is None:
                t = ep.constants[spec.target]
            args.append(t)
        elif spec.kind == InputKind.CONSTANT_TENSOR:
            args.append(ep.constants[spec.target])
        else:
            raise RuntimeError(f"unsupported placeholder kind {spec.kind}")
    return args


class AbsMaxRecorder(fx.Interpreter):
    def __init__(self, gm: fx.GraphModule, names: dict[fx.Node, str], rms_names=(),
                 quant_samples_per_sequence: int = 0) -> None:
        super().__init__(gm)
        self.names = {n.name: nm for n, nm in names.items()}
        self.absmax: dict[str, float] = {nm: 0.0 for nm in self.names.values()}
        self.rms_min = {nm: float("inf") for nm in rms_names}
        self.rms_widths = dict(rms_names) if isinstance(rms_names, dict) else {}
        from .calibration_bounds import QuantizationSamples
        self.quant_samples = ({name: QuantizationSamples(quant_samples_per_sequence) for name in self.absmax}
                              if quant_samples_per_sequence else {})

    def run_node(self, n: fx.Node):
        out = super().run_node(n)
        nm = self.names.get(n.name)
        if nm is not None:
            value = float(out.detach().abs().max())
            if not np.isfinite(value):
                raise ValueError(f"nonfinite calibration activation {nm!r}")
            self.absmax[nm] = max(self.absmax[nm], value)
            if nm in self.quant_samples:
                self.quant_samples[nm].add(out.detach().cpu().numpy())
            if nm in self.rms_min:
                values = out.detach().double()
                if nm in self.rms_widths:
                    values = values.reshape(-1, self.rms_widths[nm])
                rms = values.square().mean(dim=-1).sqrt().min().item()
                self.rms_min[nm] = min(self.rms_min[nm], rms)
        return out


def calibrate(model, n_seqs: int = 64, seq_len: int = 256, data_dir: Path | None = None, seed: int = 0) -> dict[str, float]:
    """`model` is an `export.Model`. Returns {llaccel.name: absmax}."""
    from .data import load_corpus, sample_batch
    if n_seqs <= 0 or seq_len <= 0 or seq_len > model.cfg["max_seq"]:
        raise ValueError("calibration needs positive sequences and a length within model max_seq")
    train_ids, _, _ = load_corpus(data_dir)
    if len(train_ids) and (train_ids.min() < 0 or train_ids.max() >= model.cfg["vocab"]):
        raise ValueError("calibration corpus token IDs exceed model vocabulary; supply a matching corpus")
    rng = np.random.default_rng(seed)
    rec = AbsMaxRecorder(model.ep.graph_module, model.node_names)
    x, _ = sample_batch(train_ids, n_seqs, seq_len, rng)
    with torch.no_grad():
        for i in range(n_seqs):
            tokens = torch.from_numpy(x[i : i + 1])
            rec.run(*placeholder_args(model.ep, (tokens,)))
    missing = [k for k, v in rec.absmax.items() if v == 0.0]
    if missing:
        raise RuntimeError(f"calibration recorded zero abs-max for {missing}")
    return rec.absmax


def calibrate_tokens(model, sequences: list[list[int]], *, optimize_quantization: bool = False,
                     sample_budget: int = 32768) -> dict[str, float]:
    """Calibrate exported operations using IDs from the checkpoint's tokenizer."""
    if not sequences:
        raise ValueError("calibration requires at least one token sequence")
    if type(optimize_quantization) is not bool or type(sample_budget) is not int or sample_budget < 1:
        raise ValueError("invalid quantization calibration options")
    if optimize_quantization and len(sequences) > sample_budget:
        raise ValueError("sample_budget must cover at least one value per sequence")
    rms_names = {op.inputs[0].name: op.inputs[0].cols // op.attrs.get("heads", 1)
                 for op in model.ops if op.kind == "rmsnorm"}
    rec = AbsMaxRecorder(model.ep.graph_module, model.node_names, rms_names,
                        max(1, sample_budget // len(sequences)) if optimize_quantization else 0)
    with torch.no_grad():
        for ids in sequences:
            if not 2 <= len(ids) <= model.cfg["max_seq"]:
                raise ValueError("calibration sequence length must be in [2,max_seq]")
            if any(type(i) is not int or not 0 <= i < model.cfg["vocab"] for i in ids):
                raise ValueError("calibration token IDs must be integers within vocabulary")
            rec.run(*placeholder_args(model.ep, (torch.tensor([ids], dtype=torch.long),)))
    if any(not np.isfinite(v) for v in rec.absmax.values()):
        raise ValueError("nonfinite calibration activation")
    # Zero is valid for e.g. an all-zero projection; quantizer defines zero scale.
    bounds = {}
    if optimize_quantization:
        for name, samples in rec.quant_samples.items():
            for bits in (8, 16):
                selected = samples.select(rec.absmax[name], bits)
                bounds[name + f".i{bits}_absmax"] = selected['absmax']
                bounds[name + f".i{bits}_mse"] = selected['mse']
                bounds[name + f".i{bits}_unclipped_mse"] = selected['unclipped_mse']
    return rec.absmax | {name + ".rms_min": value for name, value in rec.rms_min.items()} | bounds


def main(argv=None) -> None:
    from .export import import_model
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("-o", "--out", default="build/export/calib.json")
    ap.add_argument("--seqs", type=int, default=64)
    ap.add_argument("--len", type=int, default=256)
    args = ap.parse_args(argv)
    model, ckpt = load_checkpoint(args.ckpt)
    cfg = ModelConfig.from_dict(ckpt["config"])
    m = import_model(model, cfg)
    calib = calibrate(m, n_seqs=args.seqs, seq_len=min(args.len, cfg.max_seq))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(calib, f, indent=1)
    print(f"wrote {args.out} ({len(calib)} tensors)")


if __name__ == "__main__":
    main()
