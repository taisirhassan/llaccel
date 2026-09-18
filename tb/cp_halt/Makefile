ROOT := $(abspath ../..)
BUILD := $(ROOT)/build/tb/cp_halt
VERILATOR ?= verilator
JOBS ?= 2
.PHONY: all run
all: run
run: $(BUILD)/Vtb_cp_halt_top
	$(BUILD)/Vtb_cp_halt_top
$(BUILD)/Vtb_cp_halt_top: tb_cp_halt_top.sv tb_cp_halt.cpp $(ROOT)/rtl/cmd_proc.sv $(ROOT)/rtl/dram_arb.sv $(ROOT)/rtl/sync_fifo.sv $(ROOT)/rtl/llaccel_pkg.sv
	@mkdir -p $(BUILD)
	$(VERILATOR) --cc --exe --build --assert -j $(JOBS) -Wall -Wno-fatal -Wno-PINMISSING -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -I$(ROOT)/rtl --top-module tb_cp_halt_top --Mdir $(BUILD) -CFLAGS "-std=c++20 -O2" $(ROOT)/rtl/llaccel_pkg.sv $(ROOT)/rtl/sync_fifo.sv $(ROOT)/rtl/cmd_proc.sv $(ROOT)/rtl/dram_arb.sv tb_cp_halt_top.sv tb_cp_halt.cpp
