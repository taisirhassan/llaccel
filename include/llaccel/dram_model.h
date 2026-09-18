// Cycle-level DRAM model shared by the RTL backend, the Verilator testbenches
// and (for traffic accounting) the functional simulator.
// Semantics (docs/ARCH.md): one 64-byte request accepted per cycle when ready;
// read data returned `latency` cycles later, in order, at most one response per
// cycle; up to `maxOutstanding` reads in flight; writes complete on acceptance.
#pragma once
#include <cstdint>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <vector>

namespace llaccel {

class DramModel {
 public:
  explicit DramModel(uint64_t bytes, uint32_t latency = 100, uint32_t maxOutstanding = 32)
      : mem_(bytes, 0), latency_(latency), maxOut_(maxOutstanding) {}

  uint64_t size() const { return mem_.size(); }
  uint8_t* data() { return mem_.data(); }
  const uint8_t* data() const { return mem_.data(); }

  void write(uint64_t addr, const void* src, uint64_t n) { check(addr, n); std::memcpy(mem_.data() + addr, src, n); }
  void read(uint64_t addr, void* dst, uint64_t n) const { check(addr, n); std::memcpy(dst, mem_.data() + addr, n); }

  // ---- cycle interface ----
  bool reqReady() const { return pending_.size() < maxOut_; }

  // Called once per cycle (before tick) with the request signals; returns true if accepted.
  bool request(bool valid, bool we, uint32_t addr, const uint8_t* wdata, const uint8_t* wstrb) {
    if (!valid || !reqReady()) return false;
    if (addr % 64 != 0) throw std::runtime_error("DRAM request not 64-B aligned");
    check(addr, 64);
    if (we) {
      for (int i = 0; i < 64; ++i)
        if (wstrb[i / 8] >> (i % 8) & 1) mem_[addr + i] = wdata[i];
      wrBytes_ += 64;
    } else {
      Pending p;
      p.readyAt = cycle_ + latency_;
      std::memcpy(p.data, mem_.data() + addr, 64);
      pending_.push_back(p);
      rdBytes_ += 64;
    }
    return true;
  }

  // Advance one cycle; returns true and fills rdata if a read response is delivered this cycle.
  bool tick(uint8_t* rdata) {
    ++cycle_;
    if (!pending_.empty() && pending_.front().readyAt <= cycle_) {
      std::memcpy(rdata, pending_.front().data, 64);
      pending_.pop_front();
      return true;
    }
    return false;
  }

  uint64_t cycle() const { return cycle_; }
  uint64_t readBytes() const { return rdBytes_; }
  uint64_t writeBytes() const { return wrBytes_; }
  void resetStats() { rdBytes_ = wrBytes_ = 0; }

 private:
  struct Pending { uint64_t readyAt; uint8_t data[64]; };
  void check(uint64_t addr, uint64_t n) const {
    if (addr > mem_.size() || n > mem_.size() - addr) throw std::runtime_error("DRAM access out of range");
  }
  std::vector<uint8_t> mem_;
  uint32_t latency_, maxOut_;
  std::deque<Pending> pending_;
  uint64_t cycle_ = 0, rdBytes_ = 0, wrBytes_ = 0;
};

}  // namespace llaccel
