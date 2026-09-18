# tb/attn/Makefile — Verilator unit test for attn_engine (ATTN + KV_WRITE).
#   make            build + run
#   make lint       verilator --lint-only -Wall on the attn RTL (see ../vec/llaccel_lint.vlt for the waivers)
ROOT      := $(abspath ../..)
BUILD     := $(ROOT)/build/tb/attn
VERILATOR ?= verilator
SEED      ?= 1
JOBS      ?= 3

RTL := $(ROOT)/rtl/llaccel_pkg.sv $(ROOT)/rtl/llaccel_luts_pkg.sv $(ROOT)/rtl/sync_fifo.sv $(ROOT)/rtl/udiv.sv $(ROOT)/rtl/attn_engine.sv
STUB := $(ROOT)/tb/common/sram_stub.sv
LINT_CFG := $(ROOT)/tb/vec/llaccel_lint.vlt
VFLAGS := --cc --exe --build -j $(JOBS) --assert -Wall -Wno-fatal -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL \
          -I$(ROOT)/rtl -CFLAGS "-std=c++20 -O2 -I$(ROOT)/include -I$(ROOT)/tb/common"

.PHONY: all run build lint clean

all: run

build: $(BUILD)/attn/Vtb_attn_top

$(BUILD)/attn/Vtb_attn_top: $(RTL) $(STUB) tb_attn_top.sv tb_attn.cpp $(ROOT)/include/llaccel/numerics.h $(ROOT)/include/llaccel/isa.h
	@mkdir -p $(BUILD)/attn
	$(VERILATOR) $(VFLAGS) --top-module tb_attn_top --Mdir $(BUILD)/attn $(RTL) $(STUB) tb_attn_top.sv tb_attn.cpp

run: build
	$(BUILD)/attn/Vtb_attn_top +seed=$(SEED)

lint:
	$(VERILATOR) --lint-only -Wall $(LINT_CFG) $(ROOT)/rtl/llaccel_pkg.sv $(ROOT)/rtl/llaccel_luts_pkg.sv $(ROOT)/rtl/isqrt.sv \
	  $(ROOT)/rtl/sync_fifo.sv $(ROOT)/rtl/udiv.sv $(ROOT)/rtl/vec_engine.sv $(ROOT)/rtl/attn_engine.sv --top-module attn_engine -I$(ROOT)/rtl

clean:
	rm -rf $(BUILD)
