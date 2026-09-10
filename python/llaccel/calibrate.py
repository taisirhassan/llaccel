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
    def __init__(self, gm: fx.GraphModule, names: dict[fx.Node, str]) -> None:
        super().__init__(gm)
        self.names = {n.name: nm for n, nm in names.items()}
        self.absmax: dict[str, float] = {nm: 0.0 for nm in self.names.values()}

    def run_node(self, n: fx.Node):
        out = super().run_node(n)
        nm = self.names.get(n.name)
        if nm is not None:
            self.absmax[nm] = max(self.absmax[nm], float(out.detach().abs().max()))
        return out


def calibrate(model, n_seqs: int = 64, seq_len: int = 256, data_dir: Path | None = None, seed: int = 0) -> dict[str, float]:
    """`model` is an `export.Model`. Returns {llaccel.name: absmax}."""
    from .data import load_corpus, sample_batch
    train_ids, _, _ = load_corpus(data_dir)
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
