"""TinyLlama: a Llama/Qwen2-architecture decoder written the Hugging Face way.

The module is deliberately written with the exact float formulas that the FX
importer (python/llaccel/export.py) pattern-matches after `torch.export`:

* RMSNorm:   x * rsqrt(mean(x^2, -1) + eps) * weight
* RoPE:      HF `rotate_half` with inv_freq = base^(-arange(0, D, 2) / D)
* GQA:       repeat_interleave of the KV heads
* attention: F.scaled_dot_product_attention(..., is_causal=True)
* SwiGLU:    down(silu(gate(x)) * up(x))

`forward(tokens[B, T]) -> logits[B, T, V]` is the training / prefill path; the
integer golden model owns the KV-cache decode path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 32
    ffn: int = 384
    vocab: int = 65
    max_seq: int = 256
    rope_base: float = 10000.0
    rms_eps: float = 1e-5
    qkv_bias: bool = False

    def __post_init__(self) -> None:
        if self.n_heads * self.head_dim != self.dim:
            raise ValueError("n_heads * head_dim must equal dim")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even (rotate_half)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; cos/sin: [T, D] broadcast over batch and heads.
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        kv_dim = cfg.n_kv_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.dim, cfg.dim, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(cfg.dim, kv_dim, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(cfg.dim, kv_dim, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(a)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.ffn, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.ffn, bias=False)
        self.down_proj = nn.Linear(cfg.ffn, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim, cfg.rms_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.dim, cfg.rms_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TinyLlama(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        D = cfg.head_dim
        inv_freq = cfg.rope_base ** (-torch.arange(0, D, 2, dtype=torch.float32) / D)
        t = torch.arange(cfg.max_seq, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [max_seq, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_seq, D]
        self.register_buffer("rope_cos", emb.cos(), persistent=False)
        self.register_buffer("rope_sin", emb.sin(), persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        T = tokens.shape[1]
        x = self.embed_tokens(tokens)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, n_new: int) -> torch.Tensor:
        """Greedy decoding through the full prefill path (no KV cache)."""
        for _ in range(n_new):
            ctx = tokens[:, -self.cfg.max_seq :]
            logits = self(ctx)[:, -1, :]
            nxt = logits.argmax(-1, keepdim=True)
            tokens = torch.cat((tokens, nxt), dim=1)
        return tokens


def load_checkpoint(path: str, device: str = "cpu") -> tuple[TinyLlama, dict]:
    """Returns (model in eval mode on `device`, checkpoint dict incl. tokenizer)."""
    ckpt = torch.load(path, map_location="cpu")
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = TinyLlama(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
