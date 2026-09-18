"""Optional reporting changes must preserve timing and fail on upstream drift."""
import importlib.util
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('native_orfs', Path(__file__).resolve().parents[2] / 'synth/run_native_orfs.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

SOURCE = '''  puts "Report metrics stage $stage, $when..."
  report_checks -path_delay max
  report_puts "$when report_power"
  report_power >> $filename
  report_power_metric
  # TODO these only work to stdout
  report_design_area
'''


def test_profile_retains_timing_area_and_saves_cts_before_reports():
    result = driver.metrics_without_power(SOURCE)
    assert 'report_checks -path_delay max' in result
    assert 'report_design_area' in result
    assert 'report_power' not in result
    assert result.index('orfs_write_db') < result.index('Report metrics stage')
    assert '4_cts_before_metrics.sdc' in result


def test_profile_refuses_missing_or_duplicate_upstream_markers():
    for text in [SOURCE.replace('report_power"', 'new_command"'), SOURCE + SOURCE]:
        with pytest.raises(ValueError, match='structure changed'):
            driver.metrics_without_power(text)


@pytest.mark.parametrize('fail_segments', [False, True])
def test_route_checkpoint_ready_only_after_all_writes(tmp_path, fail_segments):
    tcl = shutil.which('tclsh')
    if not tcl:
        pytest.skip('tclsh unavailable')
    source = '''  if { ![do_global_route $res_aware $use_cugr] } {
    return
  }
'''
    prefix = '''set res_aware ""
set use_cugr ""
set ::env(RESULTS_DIR) $::env(CHECKPOINT_TEST_DIR)
proc do_global_route {args} {return 1}
proc save_test_file {path} {set f [open $path w]; puts $f complete; close $f}
proc orfs_write_db {path} {save_test_file $path}
proc orfs_write_sdc {path} {save_test_file $path}
proc write_global_route_segments {path} {
  if {$::env(FAIL_SEGMENTS)} {error "simulated segment write failure"}
  save_test_file $path
}
'''
    ready = tmp_path / '5_before_repair.ready'
    ready.write_text('stale marker')
    script = tmp_path / 'checkpoint.tcl'
    script.write_text(prefix + driver.global_route_with_checkpoint(source, 'a' * 64))
    result = subprocess.run([tcl, str(script)], capture_output=True, text=True,
                            env={**os.environ, 'CHECKPOINT_TEST_DIR': str(tmp_path),
                                 'FAIL_SEGMENTS': str(int(fail_segments))})
    if fail_segments:
        assert result.returncode != 0
        assert not ready.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert ready.read_text().strip() == 'a' * 64
        for suffix in ['odb', 'sdc', 'segments']:
            assert (tmp_path / f'5_before_repair.{suffix}').read_text() == 'complete\n'


def test_route_checkpoint_refuses_upstream_drift():
    with pytest.raises(ValueError, match='structure changed'):
        driver.global_route_with_checkpoint('changed upstream script', 'a' * 64)


def test_resume_uses_raw_segments_and_preserves_remaining_flow():
    source = '''load_design 4_cts.odb 4_cts.sdc
  log_cmd pin_access {*}$additional_args

  if { ![do_global_route $res_aware $use_cugr] } {
    return
  }
  repair_design_helper
  report_metrics 5 "global route"
'''
    result = driver.global_route_resume(source)
    assert 'load_design 5_before_repair.odb 5_before_repair.sdc' in result
    assert 'read_global_route_segments' in result
    assert 'pin_access' not in result
    assert '[do_global_route' not in result
    assert 'repair_design_helper' in result
    assert 'report_metrics 5' in result


@pytest.mark.parametrize('corrupt', [None, 'segments', 'cts', 'provenance'])
def test_resume_verifies_artifacts_and_upstream_inputs(tmp_path, corrupt):
    mapped = tmp_path/'mapped.v'
    sdc = tmp_path/'input.sdc'
    cts = tmp_path/'4_cts.odb'
    for path in [mapped, sdc, cts]:
        path.write_text(path.name)
    inputs = {name: driver.digest(path) for name, path in [
        ('mapped_sha256', mapped), ('sdc_sha256', sdc), ('cts_sha256', cts)]}
    files = {}
    for suffix in ['odb', 'sdc', 'segments']:
        path = tmp_path/f'5_before_repair.{suffix}'
        path.write_text(suffix)
        files[str(path)] = {'sha256': driver.digest(path), 'bytes': path.stat().st_size}
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    (tmp_path/'5_before_repair.ready').write_text(key)
    record = tmp_path/'record.json'
    record.write_text(json.dumps({'checkpoint_key': key,
        'profile': {'global_route_checkpoint': {'key': key, 'inputs': inputs}}, 'files': files}))
    if corrupt:
        if corrupt == 'provenance':
            value = json.loads(record.read_text())
            value['profile']['global_route_checkpoint']['inputs']['unexpected'] = 'change'
            record.write_text(json.dumps(value))
        else:
            (cts if corrupt == 'cts' else tmp_path/'5_before_repair.segments').write_text('corruption')
        with pytest.raises(ValueError, match='changed'):
            driver.verify_route_checkpoint(record, tmp_path, mapped, sdc)
    else:
        assert driver.verify_route_checkpoint(record, tmp_path, mapped, sdc) == key


def test_detailed_route_profile_preserves_final_checks():
    source = '''source_step_tcl PRE DETAIL_ROUTE
  log_cmd detailed_route {*}$all_args
  check_antennas
  if { ![design_is_routed] } {error "unrouted nets"}
'''
    result = driver.detail_route_profile(source, 2, True)
    assert 'set_thread_count 2' in result
    assert 'lappend all_args -no_pin_access' in result
    assert 'log_cmd detailed_route {*}$all_args' in result
    assert 'check_antennas' in result
    assert 'design_is_routed' in result
    with pytest.raises(ValueError):
        driver.detail_route_profile(source, 0, True)


def test_unchanged_generated_script_preserves_make_timestamp(tmp_path):
    path = tmp_path/'script.tcl'
    path.write_text('unchanged')
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    driver.write_if_changed(path, 'unchanged')
    assert path.stat().st_mtime_ns == 1_000_000_000
    driver.write_if_changed(path, 'updated')
    assert path.read_text() == 'updated'
