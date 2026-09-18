"""Accuracy / conformance checks.

Quantized (golden, integer) model vs the fp32 PyTorch model:

    uv run python -m llaccel.verify --ckpt checkpoints/tiny.pt --qgraph build/q/ --prompt "ROMEO:" --tokens 64

  The golden model generates greedily; the fp32 model is then run *teacher-forced*
  on the exact same token sequence (prompt + golden's generated tokens), so that
  at every step both models predict the next token from an identical context.
  Reported: per-step top-1 agreement rate, mean cosine similarity between the
  dequantized i16 logits and the fp32 logits, and both free-running strings.

Runtime output vs golden.json (bit-exact expectation):

    uv run python -m llaccel.verify --sim-json build/sim.json --golden build/golden.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .golden import GoldenModel


def fp32_logits_teacher_forced(model, tokens: list[int]) -> np.ndarray:
    """logits[t] = prediction after seeing tokens[:t+1] (one prefill pass, causal)."""
    with torch.no_grad():
        x = torch.tensor([tokens], dtype=torch.long)
        return model(x)[0].float().cpu().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compare_models(model, tok, qgraph_dir: Path, prompt: str, n_tokens: int) -> dict:
    g = GoldenModel(qgraph_dir)
    prompt_ids = tok.encode(prompt)
    rec = g.generate(prompt_ids, n_tokens)
    seq = prompt_ids + rec["generated"]
    fp = fp32_logits_teacher_forced(model, seq)
    vocab = g.vocab
    # steps: golden step j predicts token at position ctx_len_j; contexts are prompt prefixes (per chunk end) and decodes
    rows = []
    for j, step in enumerate(rec["steps"]):
        ctx_len = step["pos"] + step["valid_rows"]  # tokens seen so far
        gl = np.array(rec["logits_last_rows"][j][:vocab], dtype=np.float64) * 2.0 ** g.E_LOGIT
        fl = fp[ctx_len - 1]
        rows.append({"step": j, "kind": step["kind"], "ctx": ctx_len, "golden_top1": int(np.argmax(gl)),
                     "fp32_top1": int(np.argmax(fl)), "cos": cosine(gl, fl)})
    agree = float(np.mean([r["golden_top1"] == r["fp32_top1"] for r in rows]))
    mean_cos = float(np.mean([r["cos"] for r in rows]))
    with torch.no_grad():
        fp_free = model.generate(torch.tensor([prompt_ids], dtype=torch.long), n_tokens)[0, len(prompt_ids):].tolist()
    return {"rows": rows, "top1_agreement": agree, "mean_cosine": mean_cos,
            "golden_text": tok.decode(rec["generated"]), "fp32_text": tok.decode(fp_free),
            "golden_tokens": rec["generated"], "fp32_tokens": fp_free, "prompt": prompt, "n_steps": len(rows)}


def print_report(r: dict) -> None:
    print(f"{'step':>4} {'kind':>8} {'ctx':>4} {'golden':>6} {'fp32':>6} {'cos':>7}")
    for row in r["rows"]:
        mark = "" if row["golden_top1"] == row["fp32_top1"] else "  <-- differ"
        print(f"{row['step']:>4} {row['kind']:>8} {row['ctx']:>4} {row['golden_top1']:>6} {row['fp32_top1']:>6} {row['cos']:>7.4f}{mark}")
    print(f"steps={r['n_steps']}  top-1 agreement={r['top1_agreement']:.4f}  mean logit cosine={r['mean_cosine']:.4f}")
    print("---- golden (int) ----")
    print(r["prompt"] + r["golden_text"])
    print("---- fp32 ----")
    print(r["prompt"] + r["fp32_text"])


def compare_sim(sim_path: Path, golden_path: Path) -> tuple[bool, str]:
    with open(sim_path) as f:
        sim = json.load(f)
    with open(golden_path) as f:
        gold = json.load(f)
    for owner, record in (("sim", sim), ("golden", gold)):
        for key in ("argmax_per_step", "generated", "logits_last_rows"):
            if not isinstance(record.get(key), list):
                return False, f"MISMATCH: {owner}.json lacks an array {key!r}"
        if len(record["logits_last_rows"]) != len(record["argmax_per_step"]):
            return False, f"MISMATCH: {owner}.json logits/launch counts differ"
        if any(type(x) is not int for key in ("argmax_per_step", "generated") for x in record[key]):
            return False, f"MISMATCH: {owner}.json token IDs must be integers"
        if any(not isinstance(row, list) or not row or any(type(x) is not int for x in row)
               for row in record["logits_last_rows"]):
            return False, f"MISMATCH: {owner}.json logits must be nonempty integer rows"
    for key in ("argmax_per_step", "generated", "logits_last_rows"):
        a, b = sim[key], gold[key]
        if len(a) != len(b):
            return False, f"MISMATCH: {key} length sim={len(a)} golden={len(b)}"
        for i, (left, right) in enumerate(zip(a, b)):
            if left != right:
                return False, f"MISMATCH in {key} at step {i}: sim={left} golden={right}"

    return True, f"MATCH: {len(gold['generated'])} tokens, {len(gold['argmax_per_step'])} steps identical"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tiny.pt")
    ap.add_argument("--qgraph", default="build/q/")
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--sim-json", default=None)
    ap.add_argument("--golden", default="build/golden.json")
    ap.add_argument("--json", default=None, help="also write the report as JSON")
    args = ap.parse_args(argv)
    if args.sim_json:
        ok, msg = compare_sim(Path(args.sim_json), Path(args.golden))
        print(msg)
        return 0 if ok else 1
    from .data import CharTokenizer
    from .model import load_checkpoint
    model, ckpt = load_checkpoint(args.ckpt)
    tok = CharTokenizer(ckpt["tokenizer"]["itos"])
    r = compare_models(model, tok, Path(args.qgraph), args.prompt, args.tokens)
    print_report(r)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(r, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
