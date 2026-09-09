// .llbin container reader (docs/ISA.md "Binary container").
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "llaccel/isa.h"

namespace llaccel {

struct Program {
  uint32_t M = 0;
  uint32_t pc = 0;  // DRAM address of the first instruction
  std::vector<Instr> instrs;  // copy of the PROGRAM section (for disassembly / func-sim tracing)
};

struct Llbin {
  std::vector<uint8_t> dramImage;
  std::vector<Program> programs;
  nlohmann::json meta;

  static Llbin load(const std::string& path);
  const Program& programForM(uint32_t M) const;
};

}  // namespace llaccel
