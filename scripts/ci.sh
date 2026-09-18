#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CXX="${CXX:-/opt/homebrew/opt/llvm/bin/clang++}"
export CC="${CC:-/opt/homebrew/opt/llvm/bin/clang}"
ci_build="${LLACCEL_CI_BUILD:-build/ci}"
uv sync --frozen --extra hf
mkdir -p "$ci_build"
uv run --frozen --extra hf python scripts/check_toolchain.py > "$ci_build/toolchain-observed.json"
cmake -S . -B "$ci_build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" -DLLACCEL_COMPILER=ON -DLLACCEL_RTL=ON
cmake --build "$ci_build" -j 3
ctest --test-dir "$ci_build" --output-on-failure
uv run --frozen --extra hf pytest -q tests/python
make -C tb all CXX="$CXX"
make -C tb dma DMA_DEPTH=32 CXX="$CXX"
uv run --frozen --extra hf python scripts/regress_models.py --build "$ci_build" --out "$ci_build/model-regressions"
uv run --frozen --extra hf python scripts/regress_prefill.py --build "$ci_build" --out "$ci_build/prefill-regressions"
uv run --frozen --extra hf python scripts/regress_hf.py --build "$ci_build" --out "$ci_build/hf-regressions"
