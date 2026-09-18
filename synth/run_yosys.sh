#!/usr/bin/env bash
# Render constrained template parameters explicitly; Yosys .ys files do not
# expand shell environment variables. Outputs and logs stay under build/synth.
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-elab}"
variant="${2:-v1}"
case "$mode" in elab|stat) ;; *) echo "mode must be elab or stat" >&2; exit 2 ;; esac
case "$variant" in
  v1) fusion=0 ;;
  v2) fusion=1 ;;
  *) echo "variant must be v1 or v2" >&2; exit 2 ;;
esac
dma_depth="${DMA_DEPTH:-16}"
case "$dma_depth" in 1|2|4|8|16|32) ;; *) echo "DMA_DEPTH must be a power of two from 1 through 32" >&2; exit 2 ;; esac
mkdir -p build/synth
out="build/synth/llaccel_core_${variant}.v"
script="build/synth/${mode}_${variant}.ys"
sed -e "s/@FUSION@/$fusion/g" -e "s/@DMA_DEPTH@/$dma_depth/g" -e "s|@OUT@|$out|g" "synth/yosys/${mode}_core.ys" > "$script"
yosys -Q -T -l "build/synth/${mode}_${variant}.log" -s "$script"
# Verify the handoff using the Verilog frontend used by downstream synthesis.
# This catches internal cells that the backend cannot render as expressions.
if [[ "$mode" == elab ]]; then
  yosys -Q -T -l "build/synth/readback_${variant}.log" -p \
    "read_verilog $out; hierarchy -check -top llaccel_core; proc; check -assert"
fi
