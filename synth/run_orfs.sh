#!/usr/bin/env bash
# Physical-design evaluation of llaccel_core on Nangate45 with OpenROAD-flow-scripts.
#   synth/run_orfs.sh v1|v2 [stage]      stage: synth|floorplan|place|cts|route|finish (default finish)
# Step 1 (host): local Yosys + slang elaborates the SystemVerilog into a Verilog-2005 netlist.
# Step 2 (docker, amd64 image under host emulation): ORFS synth -> floorplan -> place -> CTS -> route -> reports.
# Results land in build/orfs/{logs,reports,results}/nangate45/llaccel_core_vN/base/.
set -euo pipefail
cd "$(dirname "$0")/.."
variant="${1:-v1}"
stage="${2:-finish}"
image="${ORFS_IMAGE:-openroad/orfs@sha256:d4598d07ce4dbdbed3c1baaa98860f983c305d1141ce3fdc4a31da788c32db61}"
case "$variant" in
  v1|v2) ;;
  *) echo "variant must be v1 or v2" >&2; exit 2 ;;
esac
case "$stage" in synth|floorplan|place|cts|route|finish) ;; *) echo "invalid ORFS stage" >&2; exit 2 ;; esac
snapshot="${ORFS_SNAPSHOT:-}"
if [[ -n "$snapshot" ]]; then
  [[ "$snapshot" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid snapshot name" >&2; exit 2; }
  netlist_dir="build/synth/$snapshot"
  work_dir="build/orfs/$snapshot"
  sdc_file="$netlist_dir/constraint.sdc"
  [[ -f "$netlist_dir/llaccel_core_${variant}.v" ]] || { echo "missing snapshot netlist" >&2; exit 2; }
else
  netlist_dir=build/synth
  work_dir=build/orfs
  sdc_file=synth/orfs/constraint.sdc
  synth/run_yosys.sh elab "$variant"
fi
mkdir -p "$work_dir"
echo "== [$variant] ORFS through '$stage', image $image, netlist $netlist_dir"
docker run --rm --platform linux/amd64 \
  -v "$PWD":/work \
  -e WORK_HOME="/work/$work_dir" \
  -e LLACCEL_VARIANT="$variant" -e LLACCEL_STAGE="$stage" \
  -e LLACCEL_SDC="/work/$sdc_file" \
  -e LLACCEL_NETLIST="/work/$netlist_dir/llaccel_core_${variant}.v" \
  "$image" \
  bash -lc 'set -eo pipefail
    source /OpenROAD-flow-scripts/env.sh
    cd /OpenROAD-flow-scripts/flow
    make DESIGN_CONFIG="/work/synth/orfs/config_${LLACCEL_VARIANT}.mk" \
      VERILOG_FILES="$LLACCEL_NETLIST" SDC_FILE="$LLACCEL_SDC" WORK_HOME="$WORK_HOME" NUM_CORES=3 "$LLACCEL_STAGE" 2>&1 \
      | tee "$WORK_HOME/run_${LLACCEL_VARIANT}_${LLACCEL_STAGE}.log" | tail -40'
echo "== [$variant] done. Reports: $work_dir/reports/nangate45/llaccel_core_${variant}/base/"
