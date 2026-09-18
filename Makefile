# llaccel top-level driver. Every target is idempotent; `make all` runs the
# whole chain: train -> export -> compile (v1/v2 x inorder/overlap) -> golden ->
# func-sim + RTL verification -> perf matrix. `make synth` runs the ASIC flow.
SHELL := /bin/bash
.DEFAULT_GOAL := help

ifeq ($(origin CXX),default)
CXX := /opt/homebrew/opt/llvm/bin/clang++
endif
ifeq ($(origin CC),default)
CC := /opt/homebrew/opt/llvm/bin/clang
endif
CXX ?= /opt/homebrew/opt/llvm/bin/clang++
CC ?= /opt/homebrew/opt/llvm/bin/clang
BUILD    ?= build
CKPT     ?= checkpoints/tiny.pt
EXPORT   ?= $(BUILD)/export
PROMPT   ?= ROMEO:
TOKENS   ?= 64
JOBS     ?= 11

COMPILE  := $(BUILD)/cmake/compiler/bin/llaccel-compile
SIM      := $(BUILD)/cmake/llaccel-sim
DISASM   := $(BUILD)/cmake/compiler/bin/llaccel-disasm

help:
	@echo "targets: setup train export build compile golden sim-func sim-rtl verify experiments synth test all clean"

setup:  ## python env + LUT headers
	uv sync
	uv run python python/llaccel/luts.py

train:  ## train the tiny Llama-architecture model (MPS), ~minutes
	uv run python -m llaccel.train --out $(CKPT)

export: ## torch.export -> llaccel MLIR + weights + calibration
	uv run python -m llaccel.export $(CKPT) -o $(EXPORT)

build:  ## C++: compiler (MLIR), runtime, Verilated RTL (v1 and v2)
	cmake -S . -B $(BUILD)/cmake -G Ninja -DCMAKE_CXX_COMPILER=$(CXX) -DCMAKE_C_COMPILER=$(CC) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD)/cmake -j $(JOBS)

# compile matrix: v1-inorder, v1-overlap, v2-overlap (v2-inorder for completeness)
VARIANTS := v1-inorder v1-overlap v2-inorder v2-overlap
define COMPILE_RULE
$(BUILD)/$(1).llbin: $(EXPORT)/model.mlir $(EXPORT)/weights.bin $(EXPORT)/weights.json $(EXPORT)/calib.json $(COMPILE)
	$(COMPILE) $(EXPORT)/model.mlir --weights $(EXPORT)/weights.bin --weights-json $(EXPORT)/weights.json \
	  --calib $(EXPORT)/calib.json --target llaccel-$(word 1,$(subst -, ,$(1))) \
	  $(if $(findstring v2,$(1)),--enable-fusion,) --schedule $(word 2,$(subst -, ,$(1))) \
	  --dump-qgraph $(BUILD)/q-$(1) --print-stats -o $$@
	cp $(EXPORT)/tokenizer.json $(BUILD)/tokenizer.json
endef
$(foreach v,$(VARIANTS),$(eval $(call COMPILE_RULE,$(v))))

compile: $(addprefix $(BUILD)/,$(addsuffix .llbin,$(VARIANTS)))  ## compile all variants

golden: compile  ## numpy golden trace for v1 and v2 quantized graphs (fusion changes the op list, not the numbers)
	uv run python -m llaccel.golden --qgraph $(BUILD)/q-v1-overlap --prompt "$(PROMPT)" --tokens $(TOKENS) -o $(BUILD)/golden-v1.json
	uv run python -m llaccel.golden --qgraph $(BUILD)/q-v2-overlap --prompt "$(PROMPT)" --tokens $(TOKENS) -o $(BUILD)/golden-v2.json

sim-func: golden  ## functional ISA simulator, all variants, verified against golden (+ random interleaving hazard check)
	@for v in $(VARIANTS); do g=$${v%%-*}; \
	  $(SIM) $(BUILD)/$$v.llbin --backend func --prompt "$(PROMPT)" --tokens $(TOKENS) --verify $(BUILD)/golden-$$g.json --out $(BUILD)/func-$$v.json --quiet || exit 1; \
	  $(SIM) $(BUILD)/$$v.llbin --backend func --interleave 7 --prompt "$(PROMPT)" --tokens $(TOKENS) --verify $(BUILD)/golden-$$g.json --quiet || exit 1; \
	done

sim-rtl: golden  ## Verilated RTL, all variants, verified against golden; perf counters to build/rtl-*.json
	@for v in $(VARIANTS); do g=$${v%%-*}; \
	  $(SIM) $(BUILD)/$$v.llbin --backend rtl --prompt "$(PROMPT)" --tokens $(TOKENS) --verify $(BUILD)/golden-$$g.json --out $(BUILD)/rtl-$$v.json || exit 1; \
	done

verify: sim-func sim-rtl  ## everything bit-exact + quantization accuracy vs fp32
	uv run python -m llaccel.verify --ckpt $(CKPT) --qgraph $(BUILD)/q-v1-overlap --prompt "$(PROMPT)" --tokens $(TOKENS)

experiments: ## co-design matrix table -> docs/RESULTS.md (needs sim-rtl outputs)
	uv run python scripts/experiments.py --build $(BUILD) --out docs/RESULTS.md

synth:  ## ASIC flow: local yosys elaboration + OpenROAD nangate45 (docker), v1 and v2
	synth/run_orfs.sh v1
	synth/run_orfs.sh v2
	uv run python scripts/collect_ppa.py v1 --json $(BUILD)/ppa-v1.json
	uv run python scripts/collect_ppa.py v2 --json $(BUILD)/ppa-v2.json

test: build  ## unit tests: python, C++ numerics, RTL engine testbenches, compiler tests
	uv run pytest -q tests/python
	ctest --test-dir $(BUILD)/cmake --output-on-failure
	$(MAKE) -C tb all

all: setup train export build compile golden sim-func sim-rtl verify experiments

clean:
	rm -rf $(BUILD)

.PHONY: help setup train export build compile golden sim-func sim-rtl verify experiments synth test all clean
