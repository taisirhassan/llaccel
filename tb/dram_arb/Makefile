SHELL := /bin/bash
ROOT := $(abspath ../..)
JOBS ?= 2
SEED ?= 1
VERILATOR ?= verilator
DEPTHS := 1 4 64
.PHONY: all
all: $(foreach D,$(DEPTHS),$(ROOT)/build/tb/dram_arb$(D)/Vdram_arb)
	@for depth in $(DEPTHS); do $(ROOT)/build/tb/dram_arb$$depth/Vdram_arb $(SEED); done

define build_rule
$(ROOT)/build/tb/dram_arb$(1)/Vdram_arb: $(ROOT)/rtl/dram_arb.sv $(ROOT)/rtl/sync_fifo.sv $(ROOT)/rtl/llaccel_pkg.sv $(ROOT)/tb/dram_arb/tb_dram_arb.cpp
	@mkdir -p $(ROOT)/build/tb/dram_arb$(1)
	cd $(ROOT) && $(VERILATOR) --cc --exe --build --assert -j $(JOBS) -Wall -Wno-UNUSEDPARAM -Wno-UNUSEDSIGNAL -Irtl --top-module dram_arb -GTAG_DEPTH=$(1) --Mdir $(ROOT)/build/tb/dram_arb$(1) -CFLAGS "-std=c++20 -O1 -DTEST_TAG_DEPTH=$(1)" rtl/llaccel_pkg.sv rtl/sync_fifo.sv rtl/dram_arb.sv $(ROOT)/tb/dram_arb/tb_dram_arb.cpp
endef
$(foreach D,$(DEPTHS),$(eval $(call build_rule,$(D))))
