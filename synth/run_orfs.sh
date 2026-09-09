#!/usr/bin/env bash
# Physical-design evaluation of llaccel_core on Nangate45 with OpenROAD-flow-scripts.
#   synth/run_orfs.sh v1|v2 [stage]      stage: synth|floorplan|place|cts|route|finish (default finish)
# Step 1 (host): local Yosys + slang elaborates the SystemVerilog into a Verilog-2005 netlist.
# Step 2 (docker, amd64 image under Rosetta): ORFS synth -> floorplan -> place -> CTS -> route -> reports.
# Results land in build/orfs/{logs,reports,results}/nangate45/llaccel_core_vN/base/.
set -euo pipefail
cd "$(dirname "$0")/.."
variant="${1:-v1}"
stage="${2:-finish}"
case "$variant" in
  v1) fusion=0 ;;
  v2) fusion=1 ;;
  *) echo "variant must be v1 or v2" >&2; exit 2 ;;
esac
mkdir -p build/synth build/orfs
echo "== [$variant] elaborating SystemVerilog -> build/synth/llaccel_core_${variant}.v"
FUSION=$fusion OUT=build/synth/llaccel_core_${variant}.v yosys -q -l build/synth/elab_${variant}.log -s synth/yosys/elab_core.ys
grep -E "Number of cells|Estimated number of transistors" build/synth/elab_${variant}.log | tail -2 || true
echo "== [$variant] ORFS nangate45 through '$stage' (this runs an amd64 container under emulation; expect tens of minutes)"
docker run --rm --platform linux/amd64 \
  -v "$PWD":/work \
  -e WORK_HOME=/work/build/orfs \
  openroad/orfs:latest \
  bash -lc "source /OpenROAD-flow-scripts/env.sh >/dev/null 2>&1 || true; cd /OpenROAD-flow-scripts/flow && \
            make DESIGN_CONFIG=/work/synth/orfs/config_${variant}.mk WORK_HOME=/work/build/orfs $stage 2>&1 | tail -40"
echo "== [$variant] done. Reports: build/orfs/reports/nangate45/llaccel_core_${variant}/base/"
