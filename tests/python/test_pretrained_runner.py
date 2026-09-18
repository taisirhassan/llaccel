"""Small provenance/verification fixtures for the public checkpoint runner."""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'regress_pretrained', Path(__file__).resolve().parents[2] / 'scripts/regress_pretrained.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def export_case(tmp_path):
    checkpoint, exported = tmp_path / 'checkpoint', tmp_path / 'export'
    checkpoint.mkdir()
    exported.mkdir()
    config = {'model_type': 'llama', 'hidden_size': 32}
    write(checkpoint / 'config.json', config)
    for name in ['model.mlir', 'weights.bin', 'weights.json', 'calib.json', 'hf-tokenizer/tokenizer.json']:
        path = exported / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'fixture')
    (checkpoint / 'model.safetensors').write_bytes(b'checkpoint fixture')
    sources = {'model.safetensors': runner.sha256(checkpoint / 'model.safetensors')}
    metadata = {'context': 32, 'calibration_sequences': 64, 'calibration_sha256': 'calibration-hash',
                'source': str(checkpoint), 'hf_config': config, 'checkpoint_files': sources,
                'float_import_validation': [{'length': 3, 'max_abs_error': 1e-6}],
                'smoothquant': {'alpha': 0.5, 'smooth_values': True,
                                'smooth_ffn': True, 'smooth_lm_head': True}}
    metadata['export_files_sha256'] = {name:runner.sha256(exported/name) for name in ['model.mlir','weights.bin','weights.json','calib.json','hf-tokenizer/tokenizer.json']}
    write(exported / 'hf-import.json', metadata)
    args = argparse.Namespace(context=32, calibration_sequences=64)
    def validate():
        return runner.validate_export(exported, checkpoint, args, 'calibration-hash', sources)
    return exported, metadata, validate


def test_valid_export(export_case):
    _, metadata, validate = export_case
    assert validate() == metadata


@pytest.mark.parametrize(('field', 'value'), [
    ('context', 16), ('calibration_sequences', 8), ('calibration_sha256', 'stale'),
    ('source', '/different/checkpoint'), ('hf_config', {}),
    ('checkpoint_files', {'model.safetensors': 'stale'}), ('checkpoint_files', {}),
    ('float_import_validation', 'not run'), ('float_import_validation', []),
    ('smoothquant', {'alpha': 0.3}),
])
def test_stale_export_rejected(export_case, field, value):
    exported, metadata, validate = export_case
    metadata[field] = value
    write(exported / 'hf-import.json', metadata)
    with pytest.raises(ValueError):
        validate()


@pytest.mark.parametrize('feature', ['smooth_values', 'smooth_ffn', 'smooth_lm_head'])
@pytest.mark.parametrize('value', [None, False, 1, 'true'])
def test_old_or_disabled_smoothing_rejected(export_case, feature, value):
    exported, metadata, validate = export_case
    if value is None:
        metadata['smoothquant'].pop(feature)
    else:
        metadata['smoothquant'][feature] = value
    write(exported / 'hf-import.json', metadata)
    with pytest.raises(ValueError, match=feature):
        validate()


@pytest.mark.parametrize('filename', ['model.mlir', 'weights.bin', 'weights.json', 'calib.json', 'hf-tokenizer/tokenizer.json'])
def test_missing_export_artifact_rejected(export_case, filename):
    exported, _, validate = export_case
    (exported / filename).unlink()
    with pytest.raises(ValueError, match='incomplete export'):
        validate()


@pytest.fixture
def run_case(tmp_path):
    directory = tmp_path / 'run'
    directory.mkdir()
    image, simulator = tmp_path / 'model.llbin', tmp_path / 'simulator'
    image.write_bytes(b'compiled model')
    simulator.write_bytes(b'simulator binary')
    golden = {'prompt_tokens': [5], 'generated': [2], 'argmax_per_step': [2, 1],
              'logits_last_rows': [[0, 1, 2], [-1, 4, 2]]}
    actual = {**golden, 'steps': [{'pos': 0, 'M': 1}, {'pos': 1, 'M': 1}],
              'perf_total': {'cycles': 100}}
    write(directory / 'golden.json', golden)
    for backend in ['func', 'rtl']:
        write(directory / f'{backend}.json', actual)
        (directory / f'{backend}.log').write_text('VERIFY: MATCH (2 launches, 1 generated tokens compared)\n')
    report = {'backends': ['func', 'rtl'], 'exact_integer_verification': 'MATCH',
              'image_sha256': runner.sha256(image), 'simulator_sha256': runner.sha256(simulator),
              'fp32_teacher_forced': {'rows': [{'cosine': 0.99, 'top1_match': False}]}}
    write(directory / 'report.json', report)
    return directory, actual, report, lambda: runner.collect_run(tmp_path, image, simulator)


def test_collect_preserves_quality_and_performance(run_case):
    _, _, report, collect = run_case
    result = collect()
    assert result['verification'] == report
    assert result['performance']['rtl']['launches'] == 2
    assert result['performance']['func']['total'] == {'cycles': 100}
    # Exact integer agreement must not imply original-float top-1 agreement.
    assert not result['verification']['fp32_teacher_forced']['rows'][0]['top1_match']


@pytest.mark.parametrize(('field', 'value'), [
    ('backends', ['func']), ('exact_integer_verification', 'MISMATCH'),
    ('fp32_teacher_forced', {'rows': []}), ('image_sha256', 'stale'),
    ('simulator_sha256', 'stale'),
])
def test_incomplete_or_stale_run_rejected(run_case, field, value):
    directory, _, report, collect = run_case
    report[field] = value
    write(directory / 'report.json', report)
    with pytest.raises(ValueError):
        collect()


@pytest.mark.parametrize('backend', ['func', 'rtl'])
@pytest.mark.parametrize('field', ['prompt_tokens', 'generated', 'argmax_per_step', 'logits_last_rows'])
def test_complete_trace_required(run_case, backend, field):
    directory, actual, _, collect = run_case
    actual[field] = actual[field][:-1]
    write(directory / f'{backend}.json', actual)
    with pytest.raises(ValueError, match=field):
        collect()


@pytest.mark.parametrize('backend', ['func', 'rtl'])
def test_positive_backend_verification_required(run_case, backend):
    directory, _, _, collect = run_case
    (directory / f'{backend}.log').write_text('execution interrupted')
    with pytest.raises(ValueError, match='positive simulator verification'):
        collect()

@pytest.mark.parametrize('filename', ['model.mlir','weights.bin','weights.json','calib.json','hf-tokenizer/tokenizer.json'])
def test_modified_export_artifact_rejected(export_case, filename):
    exported, _, validate = export_case
    (exported / filename).write_bytes(b'changed after export')
    with pytest.raises(ValueError, match='artifact hash'):
        validate()


def test_export_reuse_requires_requested_sampling(export_case):
    exported, metadata, _ = export_case
    args = argparse.Namespace(context=32, calibration_sequences=64, calibration_sampling='uniform-windows')
    checkpoint = Path(metadata['source'])
    metadata['calibration_selection'] = {'sampling': 'uniform-windows'}
    write(exported/'hf-import.json', metadata)
    runner.validate_export(exported, checkpoint, args, 'calibration-hash', metadata['checkpoint_files'])
    metadata['calibration_selection']['sampling'] = 'line-prefix'
    write(exported/'hf-import.json', metadata)
    with pytest.raises(ValueError, match='sampling differs'):
        runner.validate_export(exported, checkpoint, args, 'calibration-hash', metadata['checkpoint_files'])
