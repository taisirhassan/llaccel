"""configuration preflight; does not validate checkpoints or execution."""
from .hf_import import config_from_hf


def inspect_config(raw: dict, context: int = 32) -> dict:
    report = {
        'model_type': raw.get('model_type'), 'compiled_context': context,
        'status': 'UNSUPPORTED_CONFIG', 'reasons': [],
        'checkpoint_validated': False, 'compiled': False, 'executed': False,
        'hardware_validated': False,
        'scope': 'Configuration preflight only; weights, tokenizer, allocation and numerical conformance must still be checked.',
    }
    try:
        cfg = config_from_hf(raw, context)
    except (ValueError, TypeError, KeyError) as error:
        report['reasons'].append(str(error))
        return report
    report['config'] = cfg.to_dict()
    report['status'] = 'CONFIG_COMPATIBLE'
    limits = []
    if cfg.head_dim not in (16, 32, 64, 128, 256):
        limits.append('Compiler head dimensions are 16, 32, 64, 128 or 256.')
    if any(n % 16 for n in (cfg.dim, cfg.ffn, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim)):
        limits.append('Projection widths must fit 16-element compiler tiles.')
    if cfg.n_heads > 255 or cfg.n_kv_heads > 255:
        limits.append('Head counts exceed the 255-head ISA limit.')
    rope_bytes = context * cfg.head_dim * 2
    report['resident_rope_bytes'] = rope_bytes
    if rope_bytes >= 1 << 20:
        limits.append('The resident RoPE table exhausts the 1 MiB SRAM before activation allocation.')
    report['compiler_preflight_reasons'] = limits
    if limits:
        report['status'] = 'CONFIG_SUPPORTED_CAPACITY_BLOCKED'
    report['rtl_geometry_compatible'] = cfg.head_dim in (16, 32, 64, 128, 256)
    report['rtl_implementation_status'] = 'UNVERIFIED_SOURCE_EXTENSION'
    report['rtl_note'] = ('RTL source implements head dimensions through 256. Existing binaries must be rebuilt; no new RTL builds, simulations, lint or physical tests were run.')
    return report
