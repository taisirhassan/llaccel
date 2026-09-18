"""static RoPE contracts for device-resident tables."""
from __future__ import annotations

import math
import torch


def rope_config_from_hf(raw: dict, kind: str) -> dict:
    legacy, modern = raw.get('rope_scaling'), raw.get('rope_parameters')
    if any(p is not None and not isinstance(p, dict) for p in (legacy, modern)):
        raise ValueError('RoPE parameters must be an object')
    if legacy and modern:
        left = dict(legacy)
        left['rope_type'] = left.pop('type', left.get('rope_type', 'default'))
        if any(modern.get(k) != v for k, v in left.items()):
            raise ValueError('conflicting rope_scaling and rope_parameters')
    params = modern if modern is not None else legacy
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError('RoPE parameters must be an object')
    p = dict(params)
    if 'type' in p and 'rope_type' in p and p['type'] != p['rope_type']:
        raise ValueError('conflicting RoPE type aliases')
    p['rope_type'] = p.pop('type', p.get('rope_type', 'default'))
    p.setdefault('rope_theta', raw.get('rope_theta', 10000.0 if kind == 'llama' else 1000000.0))
    mode = p['rope_type']
    allowed = {'rope_type', 'rope_theta'}
    if mode in ('linear', 'llama3', 'yarn'):
        allowed.add('factor')
    elif mode != 'default':
        raise ValueError(f'unsupported RoPE type {mode!r}; supported: default, linear, llama3, yarn')
    if mode == 'llama3':
        allowed.update(('low_freq_factor', 'high_freq_factor', 'original_max_position_embeddings'))
    if mode == 'yarn':
        allowed.update(('original_max_position_embeddings', 'attention_factor', 'beta_fast', 'beta_slow', 'truncate'))
        p.setdefault('original_max_position_embeddings', raw.get('max_position_embeddings'))
        p.setdefault('attention_factor', 1 + 0.1 * math.log(p['factor']) if isinstance(p.get('factor'), (int, float)) and p['factor'] > 0 else 1)
        p.setdefault('beta_fast', 32.0)
        p.setdefault('beta_slow', 1.0)
        p.setdefault('truncate', True)
    if set(p) - allowed:
        raise ValueError(f'unsupported RoPE fields: {sorted(set(p) - allowed)}')
    for name in allowed - {'rope_type', 'original_max_position_embeddings', 'truncate'}:
        x = p.get(name)
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
            raise ValueError(f'RoPE {name} must be positive and finite')
    if mode != 'default' and p['factor'] < 1:
        raise ValueError('RoPE factor must be >= 1')
    if mode == 'yarn':
        if p['rope_theta'] <= 1 or p['beta_fast'] <= p['beta_slow']:
            raise ValueError('YaRN requires theta > 1 and beta_fast > beta_slow')
        if type(p['truncate']) is not bool:
            raise ValueError('YaRN truncate must be boolean')
        if p['attention_factor'] > 32767 / 16384:
            raise ValueError('YaRN attention_factor exceeds device Q1.14 table range')
    if mode == 'llama3':
        if p['high_freq_factor'] <= p['low_freq_factor']:
            raise ValueError('RoPE high_freq_factor must exceed low_freq_factor')
    if mode in ('llama3', 'yarn'):
        n = p.get('original_max_position_embeddings')
        if type(n) is not int or n <= 0:
            raise ValueError('RoPE original_max_position_embeddings must be a positive integer')
    return p


def build_rope_tables(cfg) -> tuple[torch.Tensor, torch.Tensor]:
    p = cfg.rope_scaling or {'rope_type': 'default', 'rope_theta': cfg.rope_base}
    d = cfg.head_dim
    inv = p['rope_theta'] ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    amplitude = 1.0
    if p['rope_type'] == 'linear':
        inv = inv / p['factor']
    elif p['rope_type'] == 'llama3':
        wave = 2 * math.pi / inv
        low_wave = p['original_max_position_embeddings'] / p['low_freq_factor']
        high_wave = p['original_max_position_embeddings'] / p['high_freq_factor']
        scaled = torch.where(wave > low_wave, inv / p['factor'], inv)
        smooth = (p['original_max_position_embeddings'] / wave - p['low_freq_factor']) / (p['high_freq_factor'] - p['low_freq_factor'])
        interpolated = (1 - smooth) * scaled / p['factor'] + smooth * scaled
        inv = torch.where((wave >= high_wave) & (wave <= low_wave), interpolated, scaled)
    elif p['rope_type'] == 'yarn':
        def correction(rotations):
            return d * math.log(p['original_max_position_embeddings'] / (rotations * 2 * math.pi)) / (2 * math.log(p['rope_theta']))
        low, high = correction(p['beta_fast']), correction(p['beta_slow'])
        if p['truncate']:
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, d - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(d // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
        frequency = p['rope_theta'] ** (torch.arange(0, d, 2, dtype=torch.float32) / d)
        inv = (1 / (p['factor'] * frequency)) * ramp + (1 / frequency) * (1 - ramp)
        amplitude = p['attention_factor']
    elif p['rope_type'] != 'default':
        raise ValueError(f"unsupported RoPE type {p['rope_type']}")
    angles = torch.outer(torch.arange(cfg.max_seq, dtype=torch.float32), inv)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos() * amplitude, angles.sin() * amplitude
