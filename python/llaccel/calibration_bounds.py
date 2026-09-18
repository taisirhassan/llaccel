"""Calibration-only clipping selection with bounded, weighted activation samples.

Raw abs-max remains separate for range/overflow checks. Bounds are explicitly
named i8_absmax/i16_absmax and minimize reconstructed activation squared error;
they are not percentile claims or a guarantee of end-to-end language quality.
"""
from __future__ import annotations

import math
import numpy as np


class QuantizationSamples:
    def __init__(self, values_per_sequence: int = 512):
        if type(values_per_sequence) is not int or values_per_sequence < 1:
            raise ValueError('values_per_sequence must be positive')
        self.per_sequence = values_per_sequence
        self.parts = []

    def add(self, values):
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        if not len(flat) or not np.isfinite(flat).all():
            raise ValueError('calibration samples must be nonempty and finite')
        n = min(self.per_sequence, len(flat))
        indices = np.linspace(0, len(flat) - 1, n, dtype=np.int64)
        sample = flat[indices].copy()
        # Preserve each sequence's extrema even if uniform sampling misses it.
        # Its unit weight represents one actual element, while each regularly
        # sampled element represents its proportional share of the activation.
        extreme = flat[np.argmax(np.abs(flat))]
        self.parts.append((sample, len(flat) / n, np.float32(extreme)))

    def select(self, absmax: float, bits: int):
        if not math.isfinite(absmax) or absmax < 0 or bits not in (8, 16) or not self.parts:
            raise ValueError('invalid calibration range, bit width, or empty samples')
        values = np.concatenate([p[0] for p in self.parts]).astype(np.float64)
        weights = np.concatenate([np.full(len(p[0]), p[1]) for p in self.parts])
        values = np.concatenate([values, np.asarray([p[2] for p in self.parts])])
        weights = np.concatenate([weights, np.ones(len(self.parts))])
        if absmax == 0:
            return {'absmax': 0.0, 'mse': 0.0, 'unclipped_mse': 0.0, 'sampled_values': len(values)}
        qmax = (1 << (bits - 1)) - 1
        if bits == 8:
            bounds = [absmax * fraction for fraction in (1, .999, .995, .99, .98, .95, .9, .85, .8, .7, .6, .5)]
            candidates = [(bound, bound / qmax) for bound in bounds]
        else:
            exponent = max(math.ceil(math.log2(absmax) - math.log2(qmax)), -15)
            candidates = [(min(absmax, qmax * 2.0**e), 2.0**e)
                          for e in range(exponent, max(exponent - 4, -15) - 1, -1)]
        records = []
        for bound, step in candidates:
            quantized = np.clip(np.floor(values / step + .5), -qmax - 1, qmax) * step
            mse = float(np.average((quantized - values) ** 2, weights=weights))
            records.append((mse, bound))
        mse, bound = min(records, key=lambda record: record[0])
        return {'absmax': float(bound), 'mse': mse, 'unclipped_mse': records[0][0],
                'sampled_values': len(values)}
