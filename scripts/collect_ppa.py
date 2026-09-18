"""Collect source-attributed ORFS PPA; absent measurements remain null.

    python3 scripts/collect_ppa.py v1 --base build/orfs/baseline --require-finish

Power is vectorless unless the run separately records activity annotation.
External SRAM is outside the synthesized core and excluded from these numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _read(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def _source(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def critical_paths(text: str, delay: str = "max") -> list[dict]:
    """Keep maximum and minimum path reports separate, including reset checks."""
    if delay not in {"max", "min"}:
        raise ValueError('path delay must be max or min')
    paths = []
    for section in re.split(r"^={10,}\s*$", text, flags=re.M):
        if not re.search(rf"report_checks -path_delay {delay}(?:\s|$)", section):
            continue
        for block in re.split(r"(?=^Startpoint:)", section, flags=re.M)[1:]:
            # OpenSTA's max-delay report can include asynchronous reset
            # recovery checks alongside datapath setup checks.
            if delay == "max" and re.search(r"\brecovery check\b", block, re.I):
                continue
            def field(pattern):
                match = re.search(pattern, block, re.M)
                return match.group(1).strip() if match else None
            slack = field(rf"^\s*({NUMBER})\s+slack(?:\s|$)")
            paths.append({'startpoint': field(r"^Startpoint:\s*(.+)$"),
                          'endpoint': field(r"^Endpoint:\s*(.+)$"),
                          'check_kind': ('removal' if re.search(r'\bremoval check\b', block, re.I)
                                         else 'setup' if delay == 'max' else 'hold'),
                          'register_to_register': bool(re.search(r'report_checks[^\n]*reg to reg', section)),
                          'slack_ns': float(slack) if slack is not None else None,
                          'report': block.strip()})
    return paths


def memory_accounting(path: Path) -> dict | None:
    """Count pre-mapping bits, not physical SRAM or gate area."""
    if not path.is_file():
        return None
    design = json.loads(path.read_text())
    memories = []
    for module, content in design.get('modules', {}).items():
        for name, cell in content.get('cells', {}).items():
            if not cell.get('type', '').startswith('$mem'):
                continue
            params = cell['parameters']
            def parameter(key):
                value = params[key]
                return int(value, 2) if isinstance(value, str) else int(value)
            width, depth = parameter('WIDTH'), parameter('SIZE')
            memories.append({'module': module, 'name': name, 'width_bits': width,
                             'depth': depth, 'bits': width * depth,
                             'readonly': parameter('WR_PORTS') == 0})
    return {'source': _source(path), 'boundary': 'core internal inferred memories before mapping',
            'mutable_bits': sum(m['bits'] for m in memories if not m['readonly']),
            'rom_bits': sum(m['bits'] for m in memories if m['readonly']),
            'note': 'ROM bits become logic; mutable memories become registers in this flow. These bit counts are not physical macro area.',
            'memories': memories}


def collect(variant: str, base: Path | None = None, stage: str = "finish") -> dict:
    base = (base or ROOT / "build/orfs").resolve()
    nick = f"llaccel_core_{variant}"
    relative = Path("nangate45") / nick / "base"
    logs, reports, results = (base / sub / relative for sub in ("logs", "reports", "results"))
    prefixes = {"finish": "6", "route": "5", "cts": "4", "place": "3", "floorplan": "2", "synth": "1"}
    prefix = prefixes[stage]
    # Keep stages separate: an absent finish report must not become a placement
    # result silently. Within the chosen stage, prefer the final report/log.
    report_paths = sorted(reports.glob(prefix + "_*.rpt"), reverse=True)
    preferred = reports / f"{prefix}_{stage}.rpt"
    report_paths = ([preferred] if preferred.is_file() else []) + [p for p in report_paths if p != preferred]
    paths = report_paths + sorted(logs.glob(prefix + "_*.log"), reverse=True)
    texts = [(p, _read(p)) for p in paths]
    sources: dict[str, str] = {}

    def find(name: str, pattern: str, cast=float, candidates=None):
        for path, text in texts if candidates is None else candidates:
            match = re.search(pattern, text, re.M)
            if match:
                value = cast(match.group(1))
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                sources[name] = _source(path)
                return value
        return None

    out = {"variant": variant, "platform": "nangate45", "stage": stage,
           "run_directory": _source(base), "external_sram_included": False,
           "physical_boundary": "llaccel_core: command processor, engines, SRAM crossbar, DRAM arbiter, counters and mapped internal arrays",
           "excluded_components": ["external 1 MiB SRAM macros", "external DRAM storage", "DRAM controller and PHY"]}
    period = _read(results / "clock_period.txt").strip()
    out["clock_period_ns"] = float(period) if re.fullmatch(NUMBER, period) else None
    if out["clock_period_ns"] is not None:
        sources["clock_period_ns"] = _source(results / "clock_period.txt")
    out["wns_ns"] = find("wns_ns", rf"^\s*wns(?:\s+max)?\s+({NUMBER})\s*$")
    out["worst_setup_slack_ns"] = find("worst_setup_slack_ns", rf"^\s*worst slack(?:\s+max)?\s+({NUMBER})\s*$")
    out["tns_ns"] = find("tns_ns", rf"^\s*tns(?:\s+max)?\s+({NUMBER})\s*$")
    for label in ['setup', 'hold', 'max slew', 'max fanout', 'max cap']:
        name = label.replace(' ', '_') + '_violation_count'
        out[name] = find(name, rf'^\s*{label} violation count\s+(\d+)\s*$', int)
    out['hold_count_scope'] = 'OpenSTA minimum-delay checks; may include asynchronous-reset removal checks.'
    out["design_area_um2"] = find("design_area_um2", rf"Design area\s+({NUMBER})\s*u(?:m)?\^2")
    out["utilization_pct"] = find("utilization_pct", rf"Design area\s+{NUMBER}\s*u(?:m)?\^2\s+({NUMBER})%")
    out["total_power_w"] = find("total_power_w", rf"^\s*Total\s+{NUMBER}\s+{NUMBER}\s+{NUMBER}\s+({NUMBER})(?:\s|$)")
    out["instance_count"] = find("instance_count", r"(?:instance count|Instances:)\s+(\d+)", int)
    # Synthesis has its own source even when collecting final routed metrics.
    synth_paths = sorted(reports.glob("*stat*")) + sorted(logs.glob("1_*.log"), reverse=True)
    synth_texts = [(p, _read(p)) for p in synth_paths if p.is_file()]
    out["synth_cells"] = find("synth_cells", r"Number of cells:\s+(\d+)", int, synth_texts)
    if out["synth_cells"] is None:
        out["synth_cells"] = find("synth_cells", r"^\s*(\d+)\s+cells\s*$", int, synth_texts)
    if out["synth_cells"] is None:
        # New Yosys stat tables include total/local counts and areas.
        out["synth_cells"] = find("synth_cells", rf"^\s*(\d+)\s+(?:{NUMBER}|-)\s+\d+\s+(?:{NUMBER}|-)\s+cells\s*$", int, synth_texts)
    out["synth_area_um2"] = find("synth_area_um2", rf"Chip area for (?:top )?module[^\n]*?:\s+({NUMBER})", candidates=synth_texts)
    # Unlike deriving 'fmax' from clipped WNS, this comes from STA's clock report.
    out["clock_min_period_ns"] = find("clock_min_period_ns", rf"^\s*clk\s+period_min\s*=\s*({NUMBER})")
    minimum = out["clock_min_period_ns"]
    out["fmax_mhz_est"] = 1000.0 / minimum if minimum is not None and minimum > 0 else None
    if minimum is not None:
        sources["fmax_mhz_est"] = sources["clock_min_period_ns"]
    out["timing_source"] = sources.get("worst_setup_slack_ns", sources.get("wns_ns"))
    out["physical_complete"] = all((results / f"6_final.{ext}").is_file() for ext in ("odb", "def", "v")) and (reports / "6_finish.rpt").is_file()
    out["power_method"] = "Vectorless OpenSTA; default input activity 0.1 transitions per minimum clock period, duty 0.5; no VCD/SAIF. External SRAM excluded."
    out["sources"] = sources
    out["critical_setup_paths"] = []
    for path, text in texts:
        extracted = critical_paths(text)
        if extracted:
            out["critical_setup_paths"] = extracted
            sources["critical_setup_paths"] = _source(path)
            break
    out['critical_min_paths'] = []
    for path, text in texts:
        extracted = critical_paths(text, 'min')
        if extracted:
            out['critical_min_paths'] = extracted
            sources['critical_min_paths'] = _source(path)
            break
    register_hold = [p['slack_ns'] for p in out['critical_min_paths']
                     if p['register_to_register'] and p['check_kind'] == 'hold' and p['slack_ns'] is not None]
    out['worst_register_hold_slack_ns'] = min(register_hold) if register_hold else None
    if register_hold:
        sources['worst_register_hold_slack_ns'] = sources['critical_min_paths']
    out["internal_memory_accounting"] = memory_accounting(results / "mem.json")
    out["external_sram_contract"] = {"bytes": 1 << 20, "banks": 16,
        "bank_word_bits": 128, "words_per_bank": 4096,
        "ports_per_bank": 1, "port_type": "single shared read/write port",
        "write_enable_granularity_bytes": 1, "behavioral_read_latency_cycles": 1,
        "contract_sources": ["rtl/sram_bank.sv", "rtl/llaccel_core.sv"],
        "physical_macro_implementation": None,
        "note": "Architectural storage interface only; external SRAM area, timing, leakage and dynamic power are unmeasured."}
    profile_path = base / "physical-profile.json"
    out["optimization_profile"] = json.loads(profile_path.read_text()) if profile_path.is_file() else None
    if out["optimization_profile"] and out["optimization_profile"].get("vectorless_power") is False:
        out["total_power_w"] = None
        sources.pop("total_power_w", None)
        out["power_method"] = "Not measured: optional vectorless power calculation omitted from this physical profile."
    if profile_path.is_file():
        sources["optimization_profile"] = _source(profile_path)
    out["sram_estimate"] = None  # No SRAM area is inferred or added to core area.
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("variant", choices=["v1", "v2"])
    ap.add_argument("--base", type=Path, default=ROOT / "build/orfs")
    ap.add_argument("--stage", choices=["synth", "floorplan", "place", "cts", "route", "finish"], default="finish")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--require-finish", action="store_true")
    args = ap.parse_args()
    result = collect(args.variant, args.base, args.stage)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered)
    if args.require_finish and not result["physical_complete"]:
        ap.exit(1, "Physical finish artifacts are incomplete; no completed PPA claim.\n")


if __name__ == "__main__":
    main()
