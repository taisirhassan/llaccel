"""Train TinyLlama on Tiny Shakespeare.

    uv run python -m llaccel.train --minutes 8 -o checkpoints/tiny.pt

Stops at `--steps` or when the `--minutes` budget is exhausted, whichever comes
first (cosine LR schedule is laid out over `--steps`). Saves
{state_dict, config, tokenizer, train_log} to the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_corpus, sample_batch
from .model import ModelConfig, TinyLlama


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, ids, batch, ctx, device, rng, iters=10) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = sample_batch(ids, batch, ctx, rng)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        logits = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
    model.train()
    return float(np.mean(losses))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="checkpoints/tiny.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default=None)
    ap.add_argument("--qkv-bias", action="store_true")
    ap.add_argument("--n-layers", type=int, default=4)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device or pick_device()
    train_ids, val_ids, tok = load_corpus()
    cfg = ModelConfig(vocab=tok.vocab_size, max_seq=args.ctx, qkv_bias=args.qkv_bias, n_layers=args.n_layers)
    model = TinyLlama(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params:,} config={cfg.to_dict()}")

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        frac = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * frac))

    log = []
    t0 = time.time()
    budget = args.minutes * 60.0
    step = 0
    train_loss = float("nan")
    model.train()
    while step < args.steps:
        if time.time() - t0 > budget:
            print(f"time budget of {args.minutes} min reached at step {step}")
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = sample_batch(train_ids, args.batch, args.ctx, rng)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_loss = loss.item()
        if step % 100 == 0 or step == args.steps - 1:
            val_loss = estimate_loss(model, val_ids, args.batch, args.ctx, device, rng)
            elapsed = time.time() - t0
            print(f"step {step:5d}  train {train_loss:.4f}  val {val_loss:.4f}  lr {lr_at(step):.2e}  {elapsed:6.1f}s", flush=True)
            log.append({"step": step, "train": train_loss, "val": val_loss, "time": elapsed})
        step += 1

    final_val = estimate_loss(model, val_ids, args.batch, args.ctx, device, rng, iters=20)
    final_train = estimate_loss(model, train_ids, args.batch, args.ctx, device, rng, iters=20)
    elapsed = time.time() - t0
    log.append({"step": step, "train": final_train, "val": final_val, "time": elapsed, "final": True})
    print(f"done: steps={step} train={final_train:.4f} val={final_val:.4f} time={elapsed:.1f}s device={device}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    torch.save(
        {"state_dict": model.state_dict(), "config": cfg.to_dict(), "tokenizer": tok.to_dict(),
         "train_log": log, "device": device, "steps": step},
        out,
    )
    with open(out.with_suffix(".log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print(f"saved {out}")

    prompt = torch.tensor([tok.encode("ROMEO:")], dtype=torch.long)
    sample = model.generate(prompt, 200)[0].tolist()
    print("---- sample ----")
    print(tok.decode(sample))


if __name__ == "__main__":
    main()
