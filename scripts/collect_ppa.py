"""Collect PPA numbers from an ORFS run into JSON (and a Markdown row).

    uv run python scripts/collect_ppa.py v1 [--stage finish]

Reads build/orfs/{logs,reports}/nangate45/llaccel_core_<v>/base/ and prints:
  clock period (from SDC), worst slack / achieved fmax, cell count, stdcell area,
  utilization, total power (if the finish report exists), plus an SRAM-macro area
  *estimate* scaled from the platform's fakeram45 LEF (labelled as an estimate).
Nothing here fabricates a number: any missing report yields null.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(p: Path) -> str:
    return p.read_text(errors="replace") if p.exists() else ""


def _grep(text: str, pattern: str, group: int = 1, cast=float):
    m = re.search(pattern, text, re.M)
    if not m:
        return None
    try:
        return cast(m.group(group))
    except ValueError:
        return None


def collect(variant: str) -> dict:
    base = ROOT / "build" / "orfs"
    nick = f"llaccel_core_{variant}"
    logs = base / "logs" / "nangate45" / nick / "base"
    reports = base / "reports" / "nangate45" / nick / "base"
    out: dict = {"variant": variant, "platform": "nangate45"}
    sdc = _read(ROOT / "synth" / "orfs" / "constraint.sdc")
    out["clock_period_ns"] = _grep(sdc, r"set clk_period\s+([0-9.]+)")

    synth_stat = _read(reports / "synth_stat.txt")
    out["synth_cells"] = _grep(synth_stat, r"Number of cells:\s+(\d+)", cast=int)
    out["synth_area_um2"] = _grep(synth_stat, r"Chip area for (?:top )?module.*?:\s+([0-9.]+)")

    # Timing/area/power: prefer the final report, fall back to the latest stage log.
    candidates = [reports / "6_finish.rpt", logs / "6_report.log", logs / "6_1_fill.log", logs / "5_2_route.log",
                  logs / "4_1_cts.log", logs / "3_5_place_dp.log", logs / "3_4_place_resized.log"]
    text = ""
    for c in candidates:
        t = _read(c)
        if t:
            text = t
            out["timing_source"] = str(c.relative_to(ROOT))
            break
    if not text:
        for c in sorted(logs.glob("*.log"), reverse=True):
            t = _read(c)
            if "wns" in t or "slack" in t:
                text, out["timing_source"] = t, str(c.relative_to(ROOT))
                break
    wns = _grep(text, r"^wns\s+(-?[0-9.]+)") or _grep(text, r"worst slack (?:max|)\s*(-?[0-9.]+)")
    tns = _grep(text, r"^tns\s+(-?[0-9.]+)")
    out["wns_ns"] = wns
    out["tns_ns"] = tns
    if wns is not None and out["clock_period_ns"]:
        out["fmax_mhz_est"] = round(1000.0 / (out["clock_period_ns"] - min(wns, 0.0)), 1)
    out["design_area_um2"] = _grep(text, r"Design area\s+([0-9.]+)\s*u\^2")
    out["utilization_pct"] = _grep(text, r"Design area\s+[0-9.]+\s*u\^2\s+([0-9.]+)%")
    out["total_power_w"] = _grep(text, r"^Total\s+[0-9.e+-]+\s+[0-9.e+-]+\s+[0-9.e+-]+\s+([0-9.e+-]+)")
    out["instance_count"] = _grep(text, r"instance count\s+(\d+)", cast=int) or _grep(text, r"Instances:\s+(\d+)", cast=int)

    # SRAM estimate from the platform fakeram LEF (area per bit scaled to 1 MiB).
    lef = _read(ROOT / "build" / "orfs" / "platform_lef" / "fakeram45_2048x39.lef")
    size = re.search(r"SIZE\s+([0-9.]+)\s+BY\s+([0-9.]+)", lef)
    if size:
        w, h = float(size.group(1)), float(size.group(2))
        bits = 2048 * 39
        out["sram_estimate"] = {
            "method": "fakeram45_2048x39 LEF area per bit × 8,388,608 bits (1 MiB), no periphery correction",
            "macro_um2": w * h,
            "um2_per_bit": w * h / bits,
            "total_um2": w * h / bits * 8 * 1024 * 1024,
        }
    else:
        out["sram_estimate"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["v1", "v2"])
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    d = collect(a.variant)
    print(json.dumps(d, indent=2))
    if a.json:
        a.json.write_text(json.dumps(d, indent=2))
    missing = [k for k, v in d.items() if v is None]
    if missing:
        print("missing (report not found / pattern not matched):", ", ".join(missing), file=sys.stderr)


if __name__ == "__main__":
    main()
