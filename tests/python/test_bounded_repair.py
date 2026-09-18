"""The optional physical diagnostic cap preserves arguments and stricter limits."""
import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / 'synth/orfs/bounded_repair.tcl'


def tcl(source):
    executable = shutil.which('tclsh')
    if not executable:
        pytest.skip('tclsh unavailable')
    return subprocess.run([executable], input=source, text=True, capture_output=True, check=True)


def test_cap_preserves_args_and_is_idempotent():
    result = tcl(f'''
set ::env(LLACCEL_REPAIR_MAX_ITERATIONS) 500
proc repair_timing {{args}} {{puts "RESULT:$args"}}
source {{{HOOK}}}
source {{{HOOK}}}
repair_timing -setup -setup_margin 0 -repair_tns 100
repair_timing -hold -max_iterations 20
repair_timing -setup -max_iterations -1
repair_timing -setup -max_iterations 2000
''')
    rows = [line for line in result.stdout.splitlines() if line.startswith('RESULT:')]
    assert rows == [
        'RESULT:-setup -setup_margin 0 -repair_tns 100 -max_iterations 500',
        'RESULT:-hold -max_iterations 20',
        'RESULT:-setup -max_iterations 500',
        'RESULT:-setup -max_iterations 500',
    ]


def test_invalid_cap_rejected():
    result = tcl(f'''
set ::env(LLACCEL_REPAIR_MAX_ITERATIONS) 0
if {{[catch {{source {{{HOOK}}}}} error]}} {{puts "ERROR:$error"}}
''')
    assert 'ERROR:LLACCEL_REPAIR_MAX_ITERATIONS must be a positive integer' in result.stdout
