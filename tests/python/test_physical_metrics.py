"""Physical report parsing must distinguish logical storage from mapped area."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('collect_ppa', Path(__file__).resolve().parents[2] / 'scripts/collect_ppa.py')
ppa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppa)


def test_setup_paths_exclude_hold():
    text = '''==========================================================================
finish report_checks -path_delay min
Startpoint: hold_start
Endpoint: hold_end
  -3.0 slack (VIOLATED)
==========================================================================
finish report_checks -path_delay max
Startpoint: rst_n
Endpoint: pc_reg (recovery check against rising-edge clock clk)
  1.22 slack (MET)
Startpoint: u_dma/data_reg (rising edge-triggered flip-flop)
Endpoint: u_gemm/result_reg (rising edge-triggered flip-flop)
  -0.25 slack (VIOLATED)
==========================================================================
finish report_checks -unconstrained
Startpoint: reset
Endpoint: out
'''
    paths = ppa.critical_paths(text)
    assert len(paths) == 1
    assert paths[0]['startpoint'].startswith('u_dma/')
    assert paths[0]['slack_ns'] == -0.25
    assert 'hold_start' not in paths[0]['report']


def test_logical_memory_bits_are_not_macro_area(tmp_path):
    def memory(width, depth, writes):
        return {'type': '$mem_v2', 'parameters': {
            'WIDTH': bin(width)[2:], 'SIZE': bin(depth)[2:], 'WR_PORTS': bin(writes)[2:]}}
    path = tmp_path / 'mem.json'
    path.write_text(json.dumps({'modules': {'core': {'cells': {
        'rom': memory(16, 256, 0), 'fifo': memory(512, 32, 1)}}}}))
    result = ppa.memory_accounting(path)
    assert result['mutable_bits'] == 16384
    assert result['rom_bits'] == 4096
    assert 'not physical macro area' in result['note']


def test_absent_finish_never_claims_area_or_completion(tmp_path):
    result = ppa.collect('v1', tmp_path)
    assert result['physical_complete'] is False
    assert result['design_area_um2'] is None
    assert result['critical_setup_paths'] == []
    assert result['sram_estimate'] is None
    assert result['external_sram_contract']['bytes'] == 1048576
    assert result['external_sram_contract']['ports_per_bank'] == 1
    assert 'DRAM controller and PHY' in result['excluded_components']


def test_new_yosys_stat_count_and_area(tmp_path):
    reports = tmp_path / 'reports/nangate45/llaccel_core_v1/base'
    reports.mkdir(parents=True)
    (reports / 'synth_stat.txt').write_text("""  1267036 1.82E+06  1267036 1.82E+06 cells
   Chip area for module '\\llaccel_core': 1824750.157996
""")
    result = ppa.collect('v1', tmp_path)
    assert result['synth_cells'] == 1267036
    assert result['synth_area_um2'] == 1824750.157996
    assert result['physical_complete'] is False


def test_openroad_um_squared_area_and_declared_profile(tmp_path):
    logs = tmp_path / 'logs/nangate45/llaccel_core_v1/base'
    logs.mkdir(parents=True)
    (logs / '6_finish.log').write_text('Design area 1824750.158 um^2 35% utilization.\n')
    (tmp_path / 'physical-profile.json').write_text(json.dumps({'repair_max_iterations': 500, 'hook_sha256': 'abc'}))
    result = ppa.collect('v1', tmp_path)
    assert result['design_area_um2'] == 1824750.158
    assert result['utilization_pct'] == 35
    assert result['optimization_profile']['repair_max_iterations'] == 500
    assert result['physical_complete'] is False


def test_omitted_power_profile_rejects_stale_power_value(tmp_path):
    reports = tmp_path / 'reports/nangate45/llaccel_core_v1/base'
    reports.mkdir(parents=True)
    (reports / '6_finish.rpt').write_text('Total 1 2 3 6\n')
    (tmp_path / 'physical-profile.json').write_text(json.dumps({'vectorless_power': False}))
    result = ppa.collect('v1', tmp_path)
    assert result['total_power_w'] is None
    assert 'total_power_w' not in result['sources']
    assert result['power_method'].startswith('Not measured:')


def test_reset_removal_does_not_become_register_hold_failure(tmp_path):
    reports = tmp_path / 'reports/nangate45/llaccel_core_v1/base'
    reports.mkdir(parents=True)
    (reports / '6_finish.rpt').write_text('''==========================================================================
finish report_checks -path_delay min
Startpoint: rst_n
Endpoint: perf (removal check against rising-edge clock clk)
  -0.33 slack (VIOLATED)
==========================================================================
finish report_checks -path_delay min reg to reg
Startpoint: d3
Endpoint: d4
  0.08 slack (MET)
==========================================================================
hold violation count 57
setup violation count 5412
max slew violation count 0
max fanout violation count 0
max cap violation count 6
''')
    result = ppa.collect('v1', tmp_path)
    assert result['worst_register_hold_slack_ns'] == 0.08
    assert result['critical_min_paths'][0]['check_kind'] == 'removal'
    assert result['critical_min_paths'][0]['slack_ns'] == -0.33
    assert result['hold_violation_count'] == 57
    assert result['setup_violation_count'] == 5412
    assert result['max_cap_violation_count'] == 6
    assert result['max_slew_violation_count'] == 0
