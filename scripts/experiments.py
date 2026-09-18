"""Build the co-design results table from measured runtime outputs.

    uv run python scripts/experiments.py --build build --out docs/RESULTS.md

Inputs (all produced by `make sim-rtl` / `make sim-func` / `make synth`):
  build/rtl-<variant>.json   per-launch perf counters from the Verilated RTL
  build/func-<variant>.json  functional-simulator traffic model (for cross-check)
  build/ppa-<v>.json         OpenROAD numbers (optional)
Every number in the output comes from those files; missing inputs print "n/a".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANTS = ["v1-inorder", "v1-overlap", "v2-inorder", "v2-overlap"]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def decode_stats(res: dict) -> dict:
    """Average actual decode launches and sum prefill, including configurable M=1."""
    prompt_length = len(res["prompt_tokens"])
    dec = [s for s in res["steps"] if s["pos"] >= prompt_length]
    pre = [s for s in res["steps"] if s["pos"] < prompt_length]
    keys = dec[0]["perf"].keys() if dec else pre[0]["perf"].keys()
    out = {"decode_launches": len(dec), "prefill_launches": len(pre)}
    for k in keys:
        out[f"dec_{k}"] = sum(s["perf"][k] for s in dec) / len(dec) if dec else None
        out[f"pre_{k}"] = sum(s["perf"][k] for s in pre) if pre else None
    return out


def pct(a, b):
    if a is None or b in (None, 0):
        return "n/a"
    return f"{100.0 * (a - b) / b:+.1f}%"


def fmt(v, digits=0):
    if v is None:
        return "n/a"
    return f"{v:,.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=Path("build"))
    ap.add_argument("--out", type=Path, default=Path("docs/RESULTS.md"))
    a = ap.parse_args()

    rtl = {v: load(a.build / f"rtl-{v}.json") for v in VARIANTS}
    func = {v: load(a.build / f"func-{v}.json") for v in VARIANTS}
    ppa = {v: load(a.build / f"ppa-{v}.json") for v in ("v1", "v2")}
    st = {v: decode_stats(r) for v, r in rtl.items() if r}

    lines = ["# Measured results", "", "All numbers below are read from files produced by the flow (`make sim-rtl`, `make synth`);",
             "nothing is estimated unless a row says so. Model: TinyLlama (see docs/PLAN.md).", ""]

    lines += ["## Decode: average per token (M=1 launches, Verilated RTL)", "",
              "| variant | cycles/token | GEMM MAC util | SRAM rd+wr bytes | DRAM rd+wr bytes | CP wait stalls | DMA busy |",
              "|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        s = st.get(v)
        if not s:
            lines.append(f"| {v} | n/a | | | | | |")
            continue
        cyc = s["dec_cycles"]
        lines.append(
            f"| {v} | {fmt(cyc)} | {fmt(100*s['dec_gemm_mac_cycles']/cyc, 1)}% | "
            f"{fmt(s['dec_sram_rd_bytes'] + s['dec_sram_wr_bytes'])} | {fmt(s['dec_dram_rd_bytes'] + s['dec_dram_wr_bytes'])} | "
            f"{fmt(100*s['dec_cp_stall_wait']/cyc, 1)}% | {fmt(100*s['dec_dma_busy']/cyc, 1)}% |")

    lines += ["", "## Prefill: total over all prefill chunks (M=16 launches)", "",
              "| variant | cycles | GEMM MAC util | SRAM bytes | DRAM bytes |", "|---|---|---|---|---|"]
    for v in VARIANTS:
        s = st.get(v)
        if not s or s["pre_cycles"] is None:
            lines.append(f"| {v} | n/a | | | |")
            continue
        cyc = s["pre_cycles"]
        lines.append(f"| {v} | {fmt(cyc)} | {fmt(100*s['pre_gemm_mac_cycles']/cyc, 1)}% | "
                     f"{fmt(s['pre_sram_rd_bytes'] + s['pre_sram_wr_bytes'])} | {fmt(s['pre_dram_rd_bytes'] + s['pre_dram_wr_bytes'])} |")

    lines += ["", "## Co-design deltas (decode, per token)", ""]
    if "v1-overlap" in st and "v2-overlap" in st:
        b, f2 = st["v1-overlap"], st["v2-overlap"]
        lines += ["| metric | v1-overlap | v2-overlap (epilogue fusion) | delta |", "|---|---|---|---|",
                  f"| cycles/token | {fmt(b['dec_cycles'])} | {fmt(f2['dec_cycles'])} | {pct(f2['dec_cycles'], b['dec_cycles'])} |",
                  f"| SRAM bytes | {fmt(b['dec_sram_rd_bytes']+b['dec_sram_wr_bytes'])} | {fmt(f2['dec_sram_rd_bytes']+f2['dec_sram_wr_bytes'])} | {pct(f2['dec_sram_rd_bytes']+f2['dec_sram_wr_bytes'], b['dec_sram_rd_bytes']+b['dec_sram_wr_bytes'])} |",
                  f"| DRAM bytes | {fmt(b['dec_dram_rd_bytes']+b['dec_dram_wr_bytes'])} | {fmt(f2['dec_dram_rd_bytes']+f2['dec_dram_wr_bytes'])} | {pct(f2['dec_dram_rd_bytes']+f2['dec_dram_wr_bytes'], b['dec_dram_rd_bytes']+b['dec_dram_wr_bytes'])} |",
                  f"| vec engine busy cycles | {fmt(b['dec_vec_busy'])} | {fmt(f2['dec_vec_busy'])} | {pct(f2['dec_vec_busy'], b['dec_vec_busy'])} |",
                  f"| instructions issued | {fmt(b['dec_instr_issued'])} | {fmt(f2['dec_instr_issued'])} | {pct(f2['dec_instr_issued'], b['dec_instr_issued'])} |"]
    if "v1-inorder" in st and "v1-overlap" in st:
        b, o = st["v1-inorder"], st["v1-overlap"]
        lines += ["", "| metric | v1-inorder | v1-overlap (DMA/compute overlap) | delta |", "|---|---|---|---|",
                  f"| cycles/token | {fmt(b['dec_cycles'])} | {fmt(o['dec_cycles'])} | {pct(o['dec_cycles'], b['dec_cycles'])} |",
                  f"| CP wait stalls | {fmt(b['dec_cp_stall_wait'])} | {fmt(o['dec_cp_stall_wait'])} | {pct(o['dec_cp_stall_wait'], b['dec_cp_stall_wait'])} |",
                  f"| DMA DRAM wait cycles | {fmt(b['dec_dma_dram_wait'])} | {fmt(o['dec_dma_dram_wait'])} | {pct(o['dec_dma_dram_wait'], b['dec_dma_dram_wait'])} |"]

    lines += ["", "## Functional-model vs RTL traffic cross-check (decode, per token)", "",
              "| variant | SRAM bytes (func model) | SRAM bytes (RTL) | DRAM bytes (func) | DRAM bytes (RTL) |", "|---|---|---|---|---|"]
    for v in VARIANTS:
        fs = decode_stats(func[v]) if func.get(v) else None
        rs = st.get(v)
        if not fs or not rs:
            lines.append(f"| {v} | n/a | n/a | n/a | n/a |")
            continue
        lines.append(f"| {v} | {fmt(fs['dec_sram_rd_bytes']+fs['dec_sram_wr_bytes'])} | {fmt(rs['dec_sram_rd_bytes']+rs['dec_sram_wr_bytes'])} | "
                     f"{fmt(fs['dec_dram_rd_bytes']+fs['dec_dram_wr_bytes'])} | {fmt(rs['dec_dram_rd_bytes']+rs['dec_dram_wr_bytes'])} |")

    lines += ["", "## Physical design (OpenROAD, Nangate45, llaccel_core = everything except SRAM bank storage)", "",
              "| variant | clock (ns) | WNS (ns) | cells (synth) | design area (µm²) | utilization | total power (W) | source |", "|---|---|---|---|---|---|---|---|"]
    for v in ("v1", "v2"):
        p = ppa.get(v)
        if not p:
            lines.append(f"| {v} | n/a | | | | | | not run |")
            continue
        lines.append(f"| {v} | {fmt(p.get('clock_period_ns'), 2)} | {fmt(p.get('wns_ns'), 3)} | {fmt(p.get('synth_cells'))} | "
                     f"{fmt(p.get('design_area_um2'), 1)} | {fmt(p.get('utilization_pct'), 1)}% | {fmt(p.get('total_power_w'), 4)} | {p.get('timing_source', 'n/a')} |")
    if ppa.get("v1") and ppa.get("v2") and ppa["v1"].get("design_area_um2") and ppa["v2"].get("design_area_um2"):
        lines.append("")
        lines.append(f"Area delta v2 vs v1: {pct(ppa['v2']['design_area_um2'], ppa['v1']['design_area_um2'])}")
    se = (ppa.get("v1") or {}).get("sram_estimate")
    if se:
        lines += ["", f"SRAM (1 MiB, 16 banks) area *estimate*: {se['total_um2']:,.0f} µm² — {se['method']}."]

    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
