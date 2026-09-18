# tb/Makefile — Verilator testbenches for the llaccel RTL.
#   make            everything (gemm dma xbar system, plus vec / attn if present)
#   make gemm       gemm_engine + xbar + banks, EPILOGUE_FUSION = 0 and 1
#   make dma        dma_engine + xbar + banks + DRAM model
#   make xbar       sram_xbar + banks, directed
#   make system     llaccel_top programs through the DRAM model
#   make lint       the three RTL gates (verilator --lint-only, slang, yosys+slang)
#   make vec/attn   delegated to tb/vec/Makefile and tb/attn/Makefile when they exist
# Build outputs go to build/tb/<test>. Run with SEED=<n> to change the random seed.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

ROOT      := $(abspath ..)
BUILD     := $(ROOT)/build/tb
VERILATOR ?= verilator
SLANG     ?= slang
YOSYS     ?= yosys
JOBS      ?= 3
SEED      ?= 1
DMA_DEPTH ?= 16

VFLAGS := --cc --exe --build --assert -j $(JOBS) -Wall -Wno-fatal \
          -CFLAGS "-std=c++23 -O1 -I$(ROOT)/include -I$(ROOT)/tb/common"
# Engine-level builds pull in the whole llaccel_pkg; its device-wide parameters
# and the tied-off crossbar ports are expected to be unused there.
UNIT   := -I$(ROOT)/rtl -Wno-UNUSEDPARAM -Wno-UNUSEDSIGNAL -Wno-PINCONNECTEMPTY
INC    := -I$(ROOT)/rtl
PKGS   := rtl/llaccel_pkg.sv rtl/llaccel_luts_pkg.sv
XBAR   := rtl/sram_xbar.sv rtl/sram_bank.sv rtl/sync_fifo.sv
COMMON := $(ROOT)/tb/common/tb_common.h $(ROOT)/include/llaccel/numerics.h $(ROOT)/include/llaccel/isa.h \
          $(ROOT)/include/llaccel/dram_model.h

# Integration validation always uses the real engines; a lint failure is a failure.
SYS_FILELIST := rtl/filelist.f
SYS_NOTE := real vec/attn engines
SYS_RTL := $(wildcard $(ROOT)/rtl/*.sv)

.PHONY: all gemm dma xbar system vec attn lint clean build-gemm build-dma build-xbar build-system

all: gemm dma xbar system vec attn dram-arb cp-halt

# ---- gemm ------------------------------------------------------------------------------
GEMM_SRC := $(addprefix $(ROOT)/,$(PKGS) rtl/gemm_engine.sv $(XBAR)) $(ROOT)/tb/gemm/tb_gemm_top.sv $(ROOT)/tb/gemm/tb_gemm.cpp
$(BUILD)/gemm_f0/Vtb_gemm_top: $(GEMM_SRC) $(COMMON)
	@mkdir -p $(BUILD)/gemm_f0
	cd $(ROOT) && $(VERILATOR) $(VFLAGS) $(UNIT) -GEPILOGUE_FUSION=0 --top-module tb_gemm_top --Mdir $(BUILD)/gemm_f0 \
	  $(PKGS) rtl/gemm_engine.sv $(XBAR) $(ROOT)/tb/gemm/tb_gemm_top.sv $(ROOT)/tb/gemm/tb_gemm.cpp
$(BUILD)/gemm_f1/Vtb_gemm_top: $(GEMM_SRC) $(COMMON)
	@mkdir -p $(BUILD)/gemm_f1
	cd $(ROOT) && $(VERILATOR) $(VFLAGS) $(UNIT) -GEPILOGUE_FUSION=1 --top-module tb_gemm_top --Mdir $(BUILD)/gemm_f1 \
	  $(PKGS) rtl/gemm_engine.sv $(XBAR) $(ROOT)/tb/gemm/tb_gemm_top.sv $(ROOT)/tb/gemm/tb_gemm.cpp
build-gemm: $(BUILD)/gemm_f0/Vtb_gemm_top $(BUILD)/gemm_f1/Vtb_gemm_top
gemm: build-gemm
	$(BUILD)/gemm_f0/Vtb_gemm_top +seed=$(SEED)
	$(BUILD)/gemm_f1/Vtb_gemm_top +seed=$(SEED)

# ---- dma -------------------------------------------------------------------------------
DMA_SRC := $(addprefix $(ROOT)/,$(PKGS) rtl/dma_engine.sv $(XBAR)) $(ROOT)/tb/dma/tb_dma_top.sv $(ROOT)/tb/dma/tb_dma.cpp
$(BUILD)/dma$(DMA_DEPTH)/Vtb_dma_top: $(DMA_SRC) $(COMMON)
	@mkdir -p $(BUILD)/dma$(DMA_DEPTH)
	cd $(ROOT) && $(VERILATOR) $(VFLAGS) $(UNIT) --top-module tb_dma_top -GMAX_OUTSTANDING=$(DMA_DEPTH) --Mdir $(BUILD)/dma$(DMA_DEPTH) \
	  $(PKGS) rtl/dma_engine.sv $(XBAR) $(ROOT)/tb/dma/tb_dma_top.sv $(ROOT)/tb/dma/tb_dma.cpp
build-dma: $(BUILD)/dma$(DMA_DEPTH)/Vtb_dma_top
dma: build-dma
	$(BUILD)/dma$(DMA_DEPTH)/Vtb_dma_top +seed=$(SEED)

# ---- xbar ------------------------------------------------------------------------------
XBAR_SRC := $(addprefix $(ROOT)/,$(PKGS) $(XBAR)) $(ROOT)/tb/xbar/tb_xbar_top.sv $(ROOT)/tb/xbar/tb_xbar.cpp
$(BUILD)/xbar/Vtb_xbar_top: $(XBAR_SRC) $(COMMON)
	@mkdir -p $(BUILD)/xbar
	cd $(ROOT) && $(VERILATOR) $(VFLAGS) $(UNIT) --top-module tb_xbar_top --Mdir $(BUILD)/xbar \
	  $(PKGS) $(XBAR) $(ROOT)/tb/xbar/tb_xbar_top.sv $(ROOT)/tb/xbar/tb_xbar.cpp
build-xbar: $(BUILD)/xbar/Vtb_xbar_top
xbar: build-xbar
	$(BUILD)/xbar/Vtb_xbar_top +seed=$(SEED)

# ---- system ----------------------------------------------------------------------------
$(BUILD)/system_f0/Vllaccel_top: $(SYS_RTL) $(ROOT)/$(SYS_FILELIST) $(ROOT)/tb/system/tb_system.cpp $(COMMON)
	@mkdir -p $(BUILD)/system_f0
	@echo "system testbench: $(SYS_NOTE)"
	cd $(ROOT) && $(VERILATOR) $(VFLAGS) -GEPILOGUE_FUSION=0 --top-module llaccel_top \
	  --Mdir $(BUILD)/system_f0 -f $(SYS_FILELIST) tb/system/tb_system.cpp
$(BUILD)/system_f1/Vllaccel_top: $(SYS_RTL) $(ROOT)/$(SYS_FILELIST) $(ROOT)/tb/system/tb_system.cpp $(COMMON)
	@mkdir -p $(BUILD)/system_f1
	@echo "system testbench: $(SYS_NOTE)"
	cd $(ROOT) && $(VERILATOR) $(VFLAGS) -GEPILOGUE_FUSION=1 --top-module llaccel_top \
	  --Mdir $(BUILD)/system_f1 -f $(SYS_FILELIST) tb/system/tb_system.cpp
build-system: $(BUILD)/system_f0/Vllaccel_top $(BUILD)/system_f1/Vllaccel_top
system: build-system
	$(BUILD)/system_f0/Vllaccel_top +seed=$(SEED)
	$(BUILD)/system_f1/Vllaccel_top +seed=$(SEED)

# ---- other engineers' testbenches -------------------------------------------------------------
vec:
	$(MAKE) -C $(ROOT)/tb/vec SEED=$(SEED) JOBS=$(JOBS)
attn:
	$(MAKE) -C $(ROOT)/tb/attn SEED=$(SEED) JOBS=$(JOBS)

cp-halt:
	$(MAKE) -C $(ROOT)/tb/cp_halt JOBS=$(JOBS)

dram-arb:
	$(MAKE) -C $(ROOT)/tb/dram_arb JOBS=$(JOBS)

# ---- gates ----------------------------------------------------------------------------------
lint:
	cd $(ROOT) && $(VERILATOR) --lint-only -Wall -f rtl/filelist.f --top-module llaccel_top
	cd $(ROOT) && $(VERILATOR) --lint-only -Wall -f rtl/filelist_core_stubs.f --top-module llaccel_top
	cd $(ROOT) && $(SLANG) -f rtl/filelist.f --top llaccel_top
	cd $(ROOT) && $(YOSYS) -q -p "plugin -i slang; read_slang -f rtl/filelist.f --top llaccel_core; hierarchy -top llaccel_core; proc; opt_clean; check -assert; tee -o /dev/stdout stat" | grep -E "cells$$"
	cd $(ROOT) && $(YOSYS) -q -p "plugin -i slang; read_slang -f rtl/filelist.f --top llaccel_core -G EPILOGUE_FUSION=1; hierarchy -top llaccel_core; proc; opt_clean; check -assert; tee -o /dev/stdout stat" | grep -E "cells$$"

clean:
	rm -rf $(BUILD)/gemm_f0 $(BUILD)/gemm_f1 $(BUILD)/dma $(BUILD)/xbar $(BUILD)/system_f0 $(BUILD)/system_f1
