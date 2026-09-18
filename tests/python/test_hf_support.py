from llaccel.hf_support import inspect_config


def config(**changes):
    return dict(model_type='llama', hidden_size=64, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                intermediate_size=128, vocab_size=37, max_position_embeddings=8192) | changes


def test_preflight_never_claims_execution():
    report = inspect_config(config())
    assert report['status'] == 'CONFIG_COMPATIBLE'
    assert not report['checkpoint_validated'] and not report['compiled']
    assert not report['executed'] and not report['hardware_validated']


def test_extended_rtl_geometry_is_not_validation():
    report = inspect_config(config(model_type='qwen3', head_dim=128))
    assert report['status'] == 'CONFIG_COMPATIBLE'
    assert report['rtl_geometry_compatible']
    assert report['rtl_implementation_status'] == 'UNVERIFIED_SOURCE_EXTENSION'
    assert not report['hardware_validated']


def test_context_capacity_is_not_import_success():
    report = inspect_config(config(head_dim=256), 4096)
    assert report['status'] == 'CONFIG_SUPPORTED_CAPACITY_BLOCKED'
    assert report['resident_rope_bytes'] == 2 * 1024 * 1024
    assert any('SRAM' in reason for reason in report['compiler_preflight_reasons'])


def test_reject_other_families_and_active_window():
    for changes in ({'model_type': 'qwen3_moe'}, {'model_type': 'llama4'}, {'use_sliding_window': True}):
        assert inspect_config(config(**changes))['status'] == 'UNSUPPORTED_CONFIG'
