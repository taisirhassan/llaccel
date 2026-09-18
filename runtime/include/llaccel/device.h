// backend-neutral device interface used by the host loop.
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "llaccel/isa.h"

namespace llaccel {

using PerfCounters = std::array<uint64_t, kNumPerf>;

class Device {
 public:
  virtual ~Device() = default;
  virtual std::string name() const = 0;
  // conservative default for unknown/older backends. Concrete backends override it.
  virtual uint32_t maxAttentionHeadDim() const { return 64; }
  // DRAM access from the host side.
  virtual void dramWrite(uint64_t addr, const void* src, uint64_t n) = 0;
  virtual void dramRead(uint64_t addr, void* dst, uint64_t n) const = 0;
  virtual uint64_t dramSize() const = 0;
  // run a program until HALT. Returns the perf counters of this launch.
  virtual PerfCounters run(uint32_t pc, uint32_t pos) = 0;
  // debug: copy of SRAM (func-sim: exact; RTL: read through the bank arrays).
  virtual std::vector<uint8_t> sramSnapshot() const = 0;
};

struct DeviceOptions {
  uint64_t dramBytes = 0;   // 0 = size to the image, rounded up
  uint32_t dramLatency = 100;
  bool trace = false;       // func-sim: print each instruction
  uint32_t interleaveSeed = 0;  // func-sim: 0 = sequential, else random legal engine interleaving
  bool epilogueFusion = false;  // RTL variant to instantiate (v1 = false, v2 = true)
  std::string fstPath;      // RTL: waveform output (empty = none)
  uint64_t maxCycles = 2'000'000'000ull;
};

std::unique_ptr<Device> makeFuncSim(const DeviceOptions& opt);
std::unique_ptr<Device> makeRtlSim(const DeviceOptions& opt);  // throws if built without RTL support

}  // namespace llaccel
