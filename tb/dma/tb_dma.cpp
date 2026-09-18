// tb_dma.cpp — randomized 2-D DMA_LOAD / DMA_STORE tests against llaccel::DramModel.
//
// Each case: random rows x row_bytes (16-B multiples) with independent strides;
// the DRAM row address is 16-B aligned and usually NOT 64-B aligned, so rows
// start and end in the middle of DRAM beats; SRAM addresses are 16-B aligned.
// Random DRAM latency and a random bank hog on a higher-priority crossbar port.
// After the instruction retires the whole SRAM (loads) or the whole DRAM
// (stores) is compared against a reference image, so stray writes are caught.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "Vtb_dma_top.h"
#include "Vtb_dma_top___024root.h"
#include "tb_common.h"
#include "verilated.h"

using namespace llaccel;
using namespace tb;

namespace {

constexpr uint64_t kDramBytes = 256 * 1024;
constexpr uint64_t kTimeout = 1'000'000;

struct Tb {
  Vtb_dma_top top;
  SramImage sram;
  DramModel* dram = nullptr;
  uint64_t cycle = 0;

  Tb() { TB_COLLECT_BANKS(top, tb_dma_top, sram); }

  template <class F>
  void tick(F& drv) {
    drv.preEdge(top);
    top.clk = 1; top.eval();
    drv.postEdge(top);
    top.clk = 0; top.eval();
    ++cycle;
  }
  void reset() {
    top.rst_n = 0; top.instr_valid = 0; top.hog_pct = 0; top.dram_req_ready = 1; top.dram_rsp_valid = 0; top.clk = 0;
    for (int i = 0; i < 16; ++i) top.instr_flat[i] = 0;
    for (int i = 0; i < 4; ++i) { top.clk = 1; top.eval(); top.clk = 0; top.eval(); }
    top.rst_n = 1;
    top.clk = 1; top.eval(); top.clk = 0; top.eval();
  }
  // Runs one instruction with the given DRAM model; returns cycles from acceptance to done_pulse or -1.
  long run(const Instr& in, DramModel& d, uint8_t hog, uint64_t* stalls, uint64_t* dramWait) {
    DramDriver<Vtb_dma_top> drv(d);
    drv.init(top);
    for (int i = 0; i < 16; ++i) top.instr_flat[i] = in.w[i];
    top.hog_pct = hog;
    top.instr_valid = 1;
    top.eval();
    uint64_t guard = 0;
    while (!top.instr_ready) { tick(drv); if (++guard > 1000) { std::printf("  engine never became ready\n"); return -1; } }
    tick(drv);  // acceptance edge
    top.instr_valid = 0;
    top.eval();
    if (!top.busy) { std::printf("  busy not asserted after accept\n"); return -1; }
    long cycles = 0;
    uint64_t st = 0, dw = 0;
    while (!top.done_pulse) {
      st += top.perf_sram_stall;
      dw += top.perf_dram_wait;
      tick(drv);
      ++cycles;
      if (cycles > long(kTimeout)) { std::printf("  TIMEOUT\n"); return -1; }
    }
    if (top.done_sig_sem != in.signalSem()) { std::printf("  done_sig_sem mismatch\n"); return -1; }
    if (!top.busy) { std::printf("  busy low in the done_pulse cycle\n"); return -1; }
    tick(drv);
    if (top.done_pulse) { std::printf("  done_pulse longer than one cycle\n"); return -1; }
    if (top.busy) { std::printf("  busy still high after done\n"); return -1; }
    top.hog_pct = 0;
    // let any hog read in flight and the DRAM model settle
    for (int i = 0; i < 4; ++i) tick(drv);
    *stalls = st; *dramWait = dw;
    return cycles;
  }
};

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  uint64_t seed = seedFromArgs(argc, argv, 1);
  int ncases = int(argValue(argc, argv, "cases", 120));
  Rng rng(seed);
  Tb tb;
  tb.reset();
  Report rep;
  std::printf("tb_dma: seed=%llu cases=%d\n", (unsigned long long)seed, ncases);

  std::vector<uint8_t> sramImg(SramImage::kBytes), dramImg(kDramBytes);
  uint64_t unaligned = 0, totalCycles = 0, totalBytes = 0, totalBeats = 0;
  for (int i = 0; i < ncases; ++i) {
    bool load = rng.coin();
    uint32_t rows = uint32_t(rng.range(1, 8));
    uint32_t rowBytes = uint32_t(rng.range(1, 24)) * 16;
    if (i == 0) { rows = 4; rowBytes = 16; }             // single-chunk rows
    if (i == 1) { rows = 1; rowBytes = 16 * 64; }         // one long row (1 KiB)
    if (rng.coin(10)) rowBytes = 64 * uint32_t(rng.range(1, 6));   // some fully 64-B rows
    uint32_t dramStride = rowBytes + uint32_t(rng.range(0, 8)) * 16;
    uint32_t sramStride = rowBytes + uint32_t(rng.range(0, 8)) * 16;
    if (rng.coin(20)) { dramStride = rowBytes; sramStride = rowBytes; }   // contiguous
    // DRAM row base: 16-B aligned, 75 % of the time not 64-B aligned
    uint32_t dramAddr = uint32_t(rng.range(0, int64_t((kDramBytes - 64 * 1024) / 16))) * 16;
    if (rng.coin(25)) dramAddr &= ~63u;
    uint32_t sramAddr = uint32_t(rng.range(0, int64_t((SramImage::kBytes - 64 * 1024) / 16))) * 16;
    static const uint32_t lat[] = {1, 3, 20, 100};
    uint32_t latency = rng.pick(lat);
    uint8_t hog = uint8_t(rng.coin(60) ? rng.range(0, 60) : 0);

    rng.fill(sramImg.data(), sramImg.size());
    rng.fill(dramImg.data(), dramImg.size());
    tb.sram.fill(sramImg);
    DramModel dram(kDramBytes, latency);
    dram.write(0, dramImg.data(), dramImg.size());

    std::vector<uint8_t> expectSram = sramImg, expectDram = dramImg;
    uint64_t beats = 0;
    for (uint32_t r = 0; r < rows; ++r) {
      uint64_t d = uint64_t(dramAddr) + uint64_t(r) * dramStride, s = uint64_t(sramAddr) + uint64_t(r) * sramStride;
      if (load) std::memcpy(&expectSram[s], &dramImg[d], rowBytes);
      else      std::memcpy(&expectDram[d], &sramImg[s], rowBytes);
      beats += rowBeats(d, rowBytes);
    }
    Instr in = load ? mkDmaLoad(sramAddr, dramAddr, rows, rowBytes, dramStride, sramStride)
                    : mkDmaStore(sramAddr, dramAddr, rows, rowBytes, sramStride, dramStride);
    in.setSignal(uint8_t(rng.range(0, 31)));

    uint64_t st = 0, dw = 0;
    long cyc = tb.run(in, dram, hog, &st, &dw);
    bool ok = cyc >= 0;
    char what[200];
    std::snprintf(what, sizeof what, "case %d %s rows=%u row_bytes=%u dram=%06x(+%u) sram=%06x(+%u) lat=%u hog=%u",
                  i, load ? "LOAD" : "STORE", rows, rowBytes, dramAddr, dramStride, sramAddr, sramStride, latency, hog);
    if (ok) {
      auto gotSram = tb.sram.snapshot();
      long d = firstDiff(gotSram.data(), expectSram.data(), gotSram.size());
      if (d >= 0) { ok = false; std::printf("  SRAM mismatch at %06lx: got %02x expected %02x\n", d, gotSram[d], expectSram[d]); }
      long e = firstDiff(dram.data(), expectDram.data(), kDramBytes);
      if (e >= 0) { ok = false; std::printf("  DRAM mismatch at %06lx: got %02x expected %02x\n", e, dram.data()[e], expectDram[e]); }
      uint64_t modelBytes = load ? dram.readBytes() : dram.writeBytes();
      if (modelBytes != beats * 64) {
        ok = false;
        std::printf("  DRAM beats: model moved %llu bytes, expected %llu (%llu beats)\n", (unsigned long long)modelBytes,
                    (unsigned long long)(beats * 64), (unsigned long long)beats);
      }
      if ((load ? dram.writeBytes() : dram.readBytes()) != 0) { ok = false; std::printf("  unexpected DRAM traffic in the other direction\n"); }
      totalCycles += uint64_t(cyc);
      totalBytes += uint64_t(rows) * rowBytes;
      totalBeats += beats;
    }
    rep.tally(ok, what);
    if (!ok) std::printf("  (%s)\n", what);
    if (dramAddr & 63) ++unaligned;
  }
  std::printf("tb_dma: %llu cases with non-64-B-aligned DRAM rows; %llu bytes in %llu beats over %llu cycles\n",
              (unsigned long long)unaligned, (unsigned long long)totalBytes, (unsigned long long)totalBeats,
              (unsigned long long)totalCycles);
  tb.top.final();
  return rep.finish("tb_dma");
}
