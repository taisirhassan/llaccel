"""Tiny Shakespeare download + char-level tokenizer."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dataset(data_dir: Path | None = None) -> Path:
    data_dir = data_dir or repo_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "input.txt"
    if not path.exists():
        print(f"downloading {URL} -> {path}")
        urllib.request.urlretrieve(URL, path)
    return path


class CharTokenizer:
    def __init__(self, itos: list[str]) -> None:
        self.itos = list(itos)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @classmethod
    def from_json(cls, path: Path | str) -> "CharTokenizer":
        with open(path) as f:
            return cls(json.load(f)["itos"])

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    def save(self, path: Path | str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def load_corpus(data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, CharTokenizer]:
    """Returns (train_ids, val_ids, tokenizer); 90/10 split, int64 arrays."""
    text = ensure_dataset(data_dir).read_text()
    tok = CharTokenizer.from_text(text)
    ids = np.array(tok.encode(text), dtype=np.int64)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], tok


def sample_batch(ids: np.ndarray, batch: int, ctx: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    starts = rng.integers(0, len(ids) - ctx - 1, size=batch)
    x = np.stack([ids[s : s + ctx] for s in starts])
    y = np.stack([ids[s + 1 : s + ctx + 1] for s in starts])
    return x, y
