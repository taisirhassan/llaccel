# tb/vec/Makefile — Verilator unit tests for vec_engine, isqrt, udiv.
#   make            build + run both tests
#   make lint       verilator --lint-only -Wall on the vec RTL
ROOT      := $(abspath ../..)
BUILD     := $(ROOT)/build/tb/vec
VERILATOR ?= verilator
SEED      ?= 1

RTL := $(ROOT)/rtl/llaccel_pkg.sv $(ROOT)/rtl/isqrt.sv $(ROOT)/rtl/udiv.sv $(ROOT)/rtl/vec_engine.sv
STUB := $(ROOT)/tb/common/sram_stub.sv
VFLAGS := --cc --exe --build -j 0 --assert -Wall -Wno-fatal -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL \
          -I$(ROOT)/rtl -CFLAGS "-std=c++20 -O2 -I$(ROOT)/include"

.PHONY: all run build lint clean

all: run

build: $(BUILD)/vec/Vtb_vec_top $(BUILD)/math/Vtb_math_top

$(BUILD)/vec/Vtb_vec_top: $(RTL) $(STUB) tb_vec_top.sv tb_vec.cpp $(ROOT)/include/llaccel/numerics.h $(ROOT)/include/llaccel/isa.h
	@mkdir -p $(BUILD)/vec
	$(VERILATOR) $(VFLAGS) --top-module tb_vec_top --Mdir $(BUILD)/vec $(RTL) $(STUB) tb_vec_top.sv tb_vec.cpp

$(BUILD)/math/Vtb_math_top: $(ROOT)/rtl/isqrt.sv $(ROOT)/rtl/udiv.sv tb_math_top.sv tb_math.cpp
	@mkdir -p $(BUILD)/math
	$(VERILATOR) $(VFLAGS) --top-module tb_math_top --Mdir $(BUILD)/math $(ROOT)/rtl/isqrt.sv $(ROOT)/rtl/udiv.sv tb_math_top.sv tb_math.cpp

run: build
	$(BUILD)/math/Vtb_math_top
	$(BUILD)/vec/Vtb_vec_top +seed=$(SEED)

lint:
	$(VERILATOR) --lint-only -Wall $(RTL) --top-module vec_engine

clean:
	rm -rf $(BUILD)
