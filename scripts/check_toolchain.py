#!/usr/bin/env python3
"""Verify the exact, locally validated macOS toolchain; install nothing.

Run inside the locked environment: uv run --frozen python scripts/check_toolchain.py.
This detects version drift, not binary provenance; see docs/REPRODUCIBILITY.md.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
expected = json.loads((root / 'toolchain.json').read_text())
failures = []
observed = {'platform': platform.system(), 'architecture': platform.machine(),
            'python': platform.python_version(), 'tools': {}, 'packages': {}, 'homebrew_libraries': {}}
commands = {
    'clang': [os.environ.get('CXX', '/opt/homebrew/opt/llvm/bin/clang++'), '--version'],
    'llvm-config': ['/opt/homebrew/opt/llvm/bin/llvm-config', '--version'],
    'verilator': ['verilator', '--version'], 'cmake': ['cmake', '--version'],
    'ninja': ['ninja', '--version'], 'uv': ['uv', '--version'],
}
for name, command in commands.items():
    try:
        result = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        observed['tools'][name] = re.search(r'\d+\.\d+(?:\.\d+)?', result).group()
    except (OSError, subprocess.CalledProcessError, AttributeError) as exc:
        failures.append(f'{name}: unavailable ({exc})')
for name in expected['packages']:
    try: observed['packages'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: failures.append(f'package missing: {name}')
for name in expected['homebrew_libraries']:
    try:
        result = subprocess.check_output(['brew', 'list', '--versions', name], text=True).split()
        observed['homebrew_libraries'][name] = result[-1] if result else 'missing'
    except (OSError, subprocess.CalledProcessError) as exc:
        failures.append(f'{name}: unavailable ({exc})')
for category, want in expected.items():
    have = observed.get(category)
    if isinstance(want, dict):
        for name, version in want.items():
            if have.get(name) != version:
                failures.append(f'{category}.{name}: expected {version}, found {have.get(name)}')
    elif have != want:
        failures.append(f'{category}: expected {want}, found {have}')
print(json.dumps(observed, indent=2))
for message in failures: print(message, file=sys.stderr)
raise SystemExit(bool(failures))
