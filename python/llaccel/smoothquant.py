"""Offline SmoothQuant for the existing RMSNorm -> linear projection groups.

The equivalent reparameterization is gamma' = gamma/s and W'[:,j] = W[:,j]*s[j].
It adds no graph operators or device instructions. Calibration quality and alpha
still determine whether subsequent integer quantization improves model quality.
Algorithm reference: https://github.com/mit-han-lab/smoothquant
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .model import TinyLlama


@torch.no_grad()
def smooth_model(model: TinyLlama, sequences: list[list[int]], alpha: float = 0.5,
                 *, smooth_lm_head: bool = True, smooth_values: bool = True,
                 smooth_ffn: bool = True, row_chunk: int = 1024,
                 auto_alpha: bool = False, sample_rows: int = 128) -> dict:
    """Smooth an fp32 TinyLlama in place and return calibration/scale metadata.

    Q/K/V share one scale vector and gate/up another. The optional final head
    group defaults on; tied embeddings are cloned before changing the head, so
    input embeddings remain exactly unchanged. Weight reductions use row chunks
    to bound temporary memory for large-vocabulary heads. All calibration and
    scale validation completes before model parameters are changed.

    By default value/output and FFN product/down groups are also balanced:
    V rows (and bias) are divided by s and O columns multiplied by the same s,
    repeated over GQA query heads. Up rows are divided by s and down columns
    multiplied by s; the gate and SiLU are unchanged. These exact linear
    reparameterizations commute with the RMSNorm groups. With auto_alpha=True,
    a bounded calibration sample selects each group alpha by weighted linear
    quantization error; the caller alpha is included among the candidates.
    This experimental local proxy is not an end-to-end quality guarantee.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be finite and in [0,1]')
    if type(sample_rows) is not int or sample_rows < 1:
        raise ValueError('sample_rows must be positive')
    if type(auto_alpha) is not bool:
        raise ValueError('auto_alpha must be boolean')
    if type(row_chunk) is not int or row_chunk <= 0:
        raise ValueError('row_chunk must be a positive integer')
    for flag in (smooth_lm_head, smooth_values, smooth_ffn):
        if type(flag) is not bool:
            raise ValueError('smoothing options must be boolean')
    if not isinstance(model, TinyLlama):
        raise TypeError('smooth_model requires TinyLlama')
    parameters = list(model.parameters())
    device = model.embed_tokens.weight.device
    if any(p.dtype != torch.float32 or p.device != device for p in parameters):
        raise ValueError('smoothing requires fp32 parameters on one device')
    if not sequences:
        raise ValueError('smoothing requires at least one calibration sequence')
    for ids in sequences:
        if not 1 <= len(ids) <= model.cfg.max_seq:
            raise ValueError('calibration sequence length must be in [1,max_seq]')
        if any(type(i) is not int or not 0 <= i < model.cfg.vocab for i in ids):
            raise ValueError('calibration token IDs must be integers within vocabulary')

    groups = []
    for i, layer in enumerate(model.layers):
        groups.append((f'layers.{i}.input_layernorm', layer.input_layernorm,
                       [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]))
        groups.append((f'layers.{i}.post_attention_layernorm', layer.post_attention_layernorm,
                       [layer.mlp.gate_proj, layer.mlp.up_proj]))
    if smooth_lm_head:
        groups.append(('norm', model.norm, [model.lm_head]))
    maxima = {name: torch.zeros(model.cfg.dim, device=device) for name, _, _ in groups}
    transfers = []
    for i, layer in enumerate(model.layers):
        if smooth_values:
            name = f'layers.{i}.value_output'
            transfers.append((name, layer.self_attn.v_proj, layer.self_attn.o_proj,
                              model.cfg.n_heads // model.cfg.n_kv_heads))
            maxima[name] = torch.zeros(model.cfg.n_kv_heads * model.cfg.head_dim, device=device)
        if smooth_ffn:
            name = f'layers.{i}.mlp_product_down'
            transfers.append((name, layer.mlp.up_proj, layer.mlp.down_proj, 1))
            maxima[name] = torch.zeros(model.cfg.ffn, device=device)
    samples = {name: [] for name in maxima}
    sample_counts = {name: 0 for name in maxima}
    hooks = []
    sampled_sequence_count = min(len(sequences), (sample_rows + 1) // 2)
    sampled_sequence_indices = ({0} if sampled_sequence_count == 1 else
        {round(i * (len(sequences) - 1) / (sampled_sequence_count - 1))
         for i in range(sampled_sequence_count)})
    collect_samples = False

    def recorder(name):
        def hook(_module, _inputs, output):
            channel_max = output.detach().reshape(-1, maxima[name].numel()).abs().amax(dim=0)
            if not torch.isfinite(channel_max).all().item():
                raise ValueError(f'nonfinite calibration activation: {name}')
            maxima[name] = torch.maximum(maxima[name], channel_max)
            if auto_alpha and collect_samples and sample_counts[name] < sample_rows:
                flat = output.detach().reshape(-1, maxima[name].numel())
                # Two evenly spaced rows per sequence spread a fixed sample
                # budget over the calibration corpus, without RNG dependence.
                count = min(2, len(flat), sample_rows - sample_counts[name])
                selected = torch.linspace(0, len(flat) - 1, count, device=device).long()
                samples[name].append(flat[selected].clone())
                sample_counts[name] += count
        return hook

    training = {module: module.training for module in model.modules()}
    try:
        model.eval()
        for name, norm, _ in groups:
            hooks.append(norm.register_forward_hook(recorder(name)))
        for name, source, target, _repeat in transfers:
            if name.endswith('.value_output'):
                hooks.append(source.register_forward_hook(recorder(name)))
            else:
                def prehook(module, inputs, name=name):
                    recorder(name)(module, (), inputs[0])
                hooks.append(target.register_forward_pre_hook(prehook))
        for sequence_index, ids in enumerate(sequences):
            collect_samples = sequence_index in sampled_sequence_indices
            # Skip projecting all calibration positions to the large vocabulary;
            # only the hidden states feeding projection groups are needed.
            x = model.embed_tokens(torch.tensor([ids], dtype=torch.long, device=device))
            cos, sin = model.rope_cos[:len(ids)], model.rope_sin[:len(ids)]
            for layer in model.layers:
                x = layer(x, cos, sin)
            model.norm(x)
    finally:
        for hook in hooks:
            hook.remove()
        for module, flag in training.items():
            module.training = flag

    plans, statistics = [], []
    minimum = 1e-5
    limit = torch.finfo(torch.float32).max

    selected_alphas, objectives = {}, {}
    candidates = sorted(set([0.0, 0.25, 0.5, 0.75, 0.9, 1.0, float(alpha)]))

    def quantization_objective(name, scales, linears, repeat):
        x = torch.cat(samples[name])
        if repeat > 1:
            x = x.reshape(len(x), model.cfg.n_kv_heads, model.cfg.head_dim).repeat_interleave(repeat, dim=1).reshape(len(x), -1)
            scales = scales.reshape(model.cfg.n_kv_heads, model.cfg.head_dim).repeat_interleave(repeat, dim=0).flatten()
            maxima_x = maxima[name].reshape(model.cfg.n_kv_heads, model.cfg.head_dim).repeat_interleave(repeat, dim=0).flatten()
        else:
            maxima_x = maxima[name]
        smoothed = x / scales
        step = (maxima_x / scales).max().clamp_min(1e-30) / 127
        quantized = (smoothed / step).round().clamp(-128, 127) * step
        activation_error = (quantized - smoothed).square().mean(dim=0)
        activation_energy = smoothed.square().mean(dim=0)
        error = 0.0
        for linear in linears:
            for chunk in linear.weight.split(row_chunk, dim=0):
                weight = chunk * scales.unsqueeze(0)
                weight_step = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / 127
                qweight = (weight / weight_step).round().clamp(-128, 127) * weight_step
                # Calibration-weighted first-order output MSE: the activation
                # and weight errors are evaluated separately, with channel
                # energies as sensitivity. No held-out data participates.
                error += float((weight.square() * activation_error.unsqueeze(0)).double().sum())
                error += float(((qweight - weight).square() * activation_energy.unsqueeze(0)).double().sum())
        return error

    def scales_for(name, column_max, linears, repeat=1):
        act_max = maxima[name]
        scores = []
        best = None
        for candidate in (candidates if auto_alpha else [float(alpha)]):
            logs = candidate * act_max.double().clamp_min(minimum).log()
            logs -= (1 - candidate) * column_max.double().clamp_min(minimum).log()
            scales = logs.exp().clamp_min(minimum).float()
            if not torch.isfinite(scales).all().item() or not (scales > 0).all().item():
                raise ValueError(f'nonfinite or nonpositive smoothing scale: {name}')
            error = quantization_objective(name, scales, linears, repeat) if auto_alpha else 0.0
            if not math.isfinite(error):
                raise ValueError(f'nonfinite calibration objective: {name}')
            scores.append({'alpha': candidate, 'weighted_error': error})
            if best is None or error < best[0]:
                best = (error, candidate, scales)
        selected_alphas[name] = best[1]
        objectives[name] = scores
        return best[2]

    def report(name, scales, column_max):
        act_max = maxima[name]
        return {'group': name, 'selected_alpha': selected_alphas[name],
            'alpha_objective': objectives[name], 'scale_min': scales.min().item(),
            'scale_max': scales.max().item(), 'scales': scales.cpu().tolist(),
            'activation_max_before': act_max.max().item(),
            'activation_max_after': (act_max / scales).max().item(),
            'weight_max_before': column_max.max().item(),
            'weight_max_after': (column_max.double() * scales.double()).max().item()}

    def check_range(tensor, name):
        if not torch.isfinite(tensor).all().item() or (tensor.abs() > limit).any().item():
            raise ValueError(f'smoothing would overflow fp32 parameters: {name}')

    input_scales = {}
    for name, norm, linears in groups:
        column_max = torch.zeros(model.cfg.dim, device=device)
        for linear in linears:
            for chunk in linear.weight.split(row_chunk, dim=0):
                if not torch.isfinite(chunk).all().item():
                    raise ValueError(f'nonfinite weight in {name}')
                column_max = torch.maximum(column_max, chunk.abs().amax(dim=0))
        scales = scales_for(name, column_max, linears)
        new_gamma = norm.weight.double() / scales.double()
        new_column_max = column_max.double() * scales.double()
        check_range(new_gamma, name)
        check_range(new_column_max, name)
        plans.append((norm, linears, scales))
        for linear in linears:
            input_scales[linear] = scales
        statistics.append(report(name, scales, column_max))

    transfer_plans = []
    for name, source, target, repeat in transfers:
        column_max = torch.zeros(target.in_features, device=device)
        for chunk in target.weight.split(row_chunk, dim=0):
            if not torch.isfinite(chunk).all().item():
                raise ValueError(f'nonfinite weight in {name}')
            column_max = torch.maximum(column_max, chunk.abs().amax(dim=0))
        if name.endswith('.value_output'):
            # o_proj input order is [KV head, repeated Q head, head channel].
            grouped_max = column_max.reshape(model.cfg.n_kv_heads, repeat, model.cfg.head_dim).amax(dim=1).flatten()
            scales = scales_for(name, grouped_max, [target], repeat)
            target_scales = scales.reshape(model.cfg.n_kv_heads, 1, model.cfg.head_dim).expand(-1, repeat, -1).reshape(-1)
        else:
            grouped_max = column_max
            scales = scales_for(name, grouped_max, [target])
            target_scales = scales
        check_range(column_max.double() * target_scales.double(), name)
        # Check the actual composition with already-planned input balancing.
        # No model mutation is made until every transfer is validated.
        for offset in range(0, source.out_features, row_chunk):
            chunk = source.weight[offset:offset + row_chunk].double()
            chunk = chunk * input_scales[source].double().unsqueeze(0)
            chunk = chunk / scales[offset:offset + row_chunk].double().unsqueeze(1)
            check_range(chunk, name)
        if source.bias is not None:
            check_range(source.bias.double() / scales.double(), name)
        transfer_plans.append((source, target, scales, target_scales))
        statistics.append(report(name, scales, grouped_max))
    tied = model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
    if smooth_lm_head and tied:
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone(),
                                            requires_grad=model.lm_head.weight.requires_grad)
    for norm, linears, scales in plans:
        norm.weight.div_(scales)
        for linear in linears:
            for chunk in linear.weight.split(row_chunk, dim=0):
                chunk.mul_(scales.unsqueeze(0))
    for source, target, scales, target_scales in transfer_plans:
        for offset in range(0, source.out_features, row_chunk):
            source.weight[offset:offset + row_chunk].div_(scales[offset:offset + row_chunk].unsqueeze(1))
        if source.bias is not None:
            source.bias.div_(scales)
        for chunk in target.weight.split(row_chunk, dim=0):
            chunk.mul_(target_scales.unsqueeze(0))
    return {'algorithm': 'SmoothQuant equivalent RMSNorm, value/output and FFN reparameterization',
            'alpha': float(alpha), 'auto_alpha': auto_alpha,
            'sample_rows_limit': sample_rows, 'calibration_sequences': len(sequences),
            'calibration_tokens': sum(map(len, sequences)), 'smooth_lm_head': smooth_lm_head,
            'untied_lm_head': bool(smooth_lm_head and tied), 'minimum_scale': minimum,
            'smooth_values': smooth_values, 'smooth_ffn': smooth_ffn,
            'groups': statistics}
