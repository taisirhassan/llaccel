// Verilated-RTL backend, templated on the Verilator top class so the v1 and v2
// hardware variants (EPILOGUE_FUSION = 0 / 1) can both be linked into one binary.
#pragma once
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "verilated.h"
#include "verilated_fst_c.h"

#include "llaccel/device.h"
#include "llaccel/dram_model.h"

namespace llaccel {

template <class Top>
class RtlSim final : public Device {
 public:
  explicit RtlSim(const DeviceOptions& opt, const char* variant)
      : opt_(opt), variant_(variant), dram_(std::max<uint64_t>(opt.dramBytes, 1u << 20), opt.dramLatency) {
    ctx_ = std::make_unique<VerilatedContext>();
    ctx_->traceEverOn(!opt.fstPath.empty());
    top_ = std::make_unique<Top>(ctx_.get());
    if (!opt.fstPath.empty()) {
      fst_ = std::make_unique<VerilatedFstC>();
      top_->trace(fst_.get(), 99);
      fst_->open(opt.fstPath.c_str());
    }
    top_->clk = 0;
    top_->rst_n = 0;
    top_->start = 0;
    top_->pc_start = 0;
    top_->pos = 0;
    top_->dram_req_ready = 1;
    top_->dram_rsp_valid = 0;
    for (int i = 0; i < 5; ++i) cycle();
    top_->rst_n = 1;
    for (int i = 0; i < 2; ++i) cycle();
  }
  ~RtlSim() override {
    if (fst_) fst_->close();
    top_->final();
  }
  std::string name() const override { return std::string("Verilated RTL (") + variant_ + ")"; }
  void dramWrite(uint64_t addr, const void* src, uint64_t n) override { dram_.write(addr, src, n); }
  void dramRead(uint64_t addr, void* dst, uint64_t n) const override { dram_.read(addr, dst, n); }
  uint64_t dramSize() const override { return dram_.size(); }
  std::vector<uint8_t> sramSnapshot() const override { return {}; }

  PerfCounters run(uint32_t pc, uint32_t pos) override {
    top_->pc_start = pc;
    top_->pos = pos;
    top_->start = 1;
    cycle();
    top_->start = 0;
    uint64_t n = 0;
    while (!top_->done) {
      cycle();
      if (++n > opt_.maxCycles) throw std::runtime_error("RTL: exceeded max cycles without done");
    }
    PerfCounters p{};
    for (uint32_t i = 0; i < kNumPerf; ++i) p[i] = top_->perf[i];
    return p;
  }

 private:
  // One clock cycle: rising edge, then service the DRAM model, then falling edge.
  void cycle() {
    top_->clk = 1;
    top_->eval();
    // Request handshake as seen by the DUT at this edge: valid && ready (ready was driven last cycle).
    // wstrb is 64 bits -> a plain uint64_t in Verilator; wdata is 512 bits -> VlWide.
    uint64_t wstrb = top_->dram_req_wstrb;
    bool accepted = dram_.request(top_->dram_req_valid && top_->dram_req_ready, top_->dram_req_we, top_->dram_req_addr,
                                  reinterpret_cast<const uint8_t*>(top_->dram_req_wdata.data()),
                                  reinterpret_cast<const uint8_t*>(&wstrb));
    (void)accepted;
    uint8_t rdata[64];
    bool rsp = dram_.tick(rdata);
    top_->dram_rsp_valid = rsp;
    if (rsp) std::memcpy(top_->dram_rsp_rdata.data(), rdata, 64);
    top_->dram_req_ready = dram_.reqReady();
    top_->clk = 0;
    top_->eval();
    if (fst_) { fst_->dump(ctx_->time()); ctx_->timeInc(1); }
  }

  DeviceOptions opt_;
  const char* variant_;
  DramModel dram_;
  std::unique_ptr<VerilatedContext> ctx_;
  std::unique_ptr<Top> top_;
  std::unique_ptr<VerilatedFstC> fst_;
};

}  // namespace llaccel
