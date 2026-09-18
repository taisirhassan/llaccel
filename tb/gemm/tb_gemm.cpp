// tb_gemm.cpp — randomized, bit-exact tests of gemm_engine against numerics.h.
//
// Each case: random M/N/K, random i8 A and W (W stored tiled per ISA.md), random
// per-channel rq {M in [2^30, 2^31), S in [24, 40]}, optional bias, i8 or i16
// output; with EPILOGUE_FUSION=1 every epilogue mode incl. random aux rows and
// SiLU params (Mi = 2^30, Si in [26, 34], sh_out in [12, 20]). Operand base
// addresses are random 16-B aligned offsets (W 256-B aligned) so the engine's
// line-split paths for rq / bias / aux / out are exercised. The whole 1 MiB is
// compared against a reference image (random snapshot + reference output), so
// stray writes are caught too. gemm_mac_cycles must equal NT*KT*M.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "Vtb_gemm_top.h"
#include "Vtb_gemm_top___024root.h"
#include "tb_common.h"
#include "verilated.h"

using namespace llaccel;
using namespace tb;

namespace {

constexpr uint64_t kTimeout = 2'000'000;

struct Tb {
  Vtb_gemm_top top;
  SramImage sram;
  uint64_t cycle = 0, dbgCycles = 0;

  Tb() { TB_COLLECT_BANKS(top, tb_gemm_top, sram); }

  void tick() {
    top.dbg = cycle < dbgCycles;
    top.clk = 0; top.eval();
    top.clk = 1; top.eval();
    ++cycle;
  }
  void reset() {
    top.rst_n = 0; top.instr_valid = 0; top.dbg = 0; top.deny_pct = 0;
    for (int i = 0; i < 16; ++i) top.instr_flat[i] = 0;
    for (int i = 0; i < 4; ++i) tick();
    top.rst_n = 1;
    tick();
  }
  // Runs one instruction; returns cycles from acceptance to done_pulse (or -1 on timeout / protocol error).
  long run(const Instr& in, uint64_t* macCycles, uint64_t* epCycles, uint64_t* stalls) {
    for (int i = 0; i < 16; ++i) top.instr_flat[i] = in.w[i];
    top.instr_valid = 1;
    top.eval();
    uint64_t guard = 0;
    while (!top.instr_ready) { tick(); if (++guard > 1000) { std::printf("  engine never became ready\n"); return -1; } }
    tick();  // acceptance edge
    top.instr_valid = 0;
    top.eval();
    if (!top.busy) { std::printf("  busy not asserted after accept\n"); return -1; }
    long cycles = 0;
    uint64_t mac = 0, ep = 0, st = 0;
    while (!top.done_pulse) {
      mac += top.perf_mac_cycle;
      ep += top.perf_epilogue_cycle;
      st += top.perf_sram_stall;
      tick();
      ++cycles;
      if (cycles > long(kTimeout)) { std::printf("  TIMEOUT\n"); return -1; }
    }
    if (top.done_sig_sem != in.signalSem()) { std::printf("  done_sig_sem mismatch\n"); return -1; }
    if (!top.busy) { std::printf("  busy low in the done_pulse cycle\n"); return -1; }
    tick();
    if (top.done_pulse) { std::printf("  done_pulse longer than one cycle\n"); return -1; }
    if (top.busy) { std::printf("  busy still high after done\n"); return -1; }
    *macCycles = mac; *epCycles = ep; *stalls = st;
    return cycles;
  }
};

struct Case {
  GemmParams p;
  uint32_t aAddr, wAddr, outAddr, rqAddr, biasAddr, auxAddr;
  std::vector<int8_t> A, W;
  std::vector<num::RqEntry> rq;
  std::vector<int32_t> bias;
  std::vector<int16_t> aux;
};

Case randomCase(Rng& rng, bool fusion, int idx) {
  static const uint32_t Ns[] = {16, 32, 48, 64, 128};
  static const uint32_t Ks[] = {16, 32, 64, 128, 256};
  Case c;
  c.p.M = uint32_t(rng.range(1, 16));
  c.p.N = rng.pick(Ns);
  c.p.K = rng.pick(Ks);
  if (idx == 0) { c.p.M = 16; c.p.N = 128; c.p.K = 256; }        // largest shape
  if (idx == 1) { c.p.M = 1; c.p.N = 16; c.p.K = 16; }           // smallest shape
  c.p.hasBias = rng.coin();
  c.p.outI8 = rng.coin(35);
  c.p.mode = Epilogue::NONE;
  if (fusion && !c.p.outI8) c.p.mode = Epilogue(rng.range(0, 3));
  c.p.auxShift = uint8_t(rng.range(8, 20));
  c.p.siluMi = 1u << 30;
  c.p.siluSi = uint32_t(rng.range(26, 34));
  c.p.siluSh = uint32_t(rng.range(12, 20));

  c.A.resize(size_t(c.p.M) * c.p.K);
  c.W.resize(size_t(c.p.N) * c.p.K);
  for (auto& v : c.A) v = int8_t(rng.u64());
  for (auto& v : c.W) v = int8_t(rng.u64());
  c.rq.resize(c.p.N);
  for (auto& e : c.rq) { e.M = int32_t((1u << 30) + (rng.u64() % (1u << 30))); e.S = int32_t(rng.range(24, 40)); }
  c.bias.resize(c.p.N);
  for (auto& v : c.bias) v = int32_t(rng.range(-(1 << 22), (1 << 22)));
  if (rng.coin(10)) for (auto& v : c.bias) v = int32_t(rng.u64());   // full i32 range (acc + bias exceeds i32)
  c.aux.resize(size_t(c.p.M) * c.p.N);
  for (auto& v : c.aux) v = int16_t(rng.u64());

  // 16-B aligned random offsets inside disjoint 64 KiB regions; W 256-B aligned.
  auto off16 = [&](uint32_t maxBytes) { return uint32_t(rng.range(0, int64_t((0x10000 - maxBytes) / 16))) * 16; };
  c.aAddr    = 0x00000 + off16(16 * 256);
  c.wAddr    = 0x10000 + uint32_t(rng.range(0, (0x10000 - 8 * 16 * 256) / 256)) * 256;
  c.rqAddr   = 0x20000 + off16(128 * 8);
  c.biasAddr = 0x30000 + off16(128 * 4);
  c.auxAddr  = 0x40000 + off16(16 * 128 * 2);
  c.outAddr  = 0x50000 + off16(16 * 128 * 2);
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  uint64_t seed = seedFromArgs(argc, argv, 1);
  int ncases = int(argValue(argc, argv, "cases", 250));
  uint64_t dbgCycles = argValue(argc, argv, "dbg", 0);   // trace the first N cycles of case 0
  Rng rng(seed);
  Tb tb;
  tb.dbgCycles = dbgCycles;
  tb.top.eval();
  const bool fusion = tb.top.fusion;   // elaborated EPILOGUE_FUSION
  std::printf("tb_gemm: EPILOGUE_FUSION=%d seed=%llu cases=%d\n", int(fusion), (unsigned long long)seed, ncases);
  tb.reset();
  Report rep;
  uint64_t totalCycles = 0, totalMac = 0, splitCases = 0;
  int modeCount[4] = {0, 0, 0, 0}, i8Count = 0;

  std::vector<uint8_t> img(SramImage::kBytes);
  for (int i = 0; i < 2 * ncases; ++i) {
    tb.top.deny_pct = (i < ncases) ? 0 : 30;
    Case c = randomCase(rng, fusion, i % ncases);
    rng.fill(img.data(), img.size());
    // place operands
    std::memcpy(&img[c.aAddr], c.A.data(), c.A.size());
    auto wt = tileWeights(c.W.data(), c.p.N, c.p.K);
    std::memcpy(&img[c.wAddr], wt.data(), wt.size());
    std::memcpy(&img[c.rqAddr], c.rq.data(), c.rq.size() * 8);
    std::memcpy(&img[c.biasAddr], c.bias.data(), c.bias.size() * 4);
    std::memcpy(&img[c.auxAddr], c.aux.data(), c.aux.size() * 2);
    tb.sram.fill(img);

    std::vector<uint8_t> expect = img;
    auto ref = refGemm(c.p, c.A.data(), c.W.data(), c.rq.data(), c.p.hasBias ? c.bias.data() : nullptr,
                       (c.p.mode == Epilogue::RESADD || c.p.mode == Epilogue::MUL) ? c.aux.data() : nullptr);
    std::memcpy(&expect[c.outAddr], ref.data(), ref.size());

    Instr in = mkGemm(c.p, c.aAddr, c.wAddr, c.outAddr, c.rqAddr, c.biasAddr, c.auxAddr);
    in.setSignal(uint8_t(rng.range(0, 31)));
    uint64_t mac = 0, ep = 0, st = 0;
    long cyc = tb.run(in, &mac, &ep, &st);
    bool ok = cyc >= 0;
    char what[256];
    std::snprintf(what, sizeof what, "case %d M=%u N=%u K=%u bias=%d i8=%d mode=%d a=%05x w=%05x rq=%05x bias=%05x aux=%05x out=%05x",
                  i, c.p.M, c.p.N, c.p.K, int(c.p.hasBias), int(c.p.outI8), int(c.p.mode), c.aAddr, c.wAddr, c.rqAddr,
                  c.biasAddr, c.auxAddr, c.outAddr);
    if (ok) {
      auto got = tb.sram.snapshot();
      long d = firstDiff(got.data(), expect.data(), got.size());
      if (d >= 0) {
        ok = false;
        std::printf("  first mismatch at SRAM %06lx: got %02x expected %02x (out region %05x..%05zx)\n", d, got[d], expect[d],
                    c.outAddr, size_t(c.outAddr) + ref.size());
      }
      uint64_t NT = c.p.N / 16, KT = c.p.K / 16;
      if (mac != NT * KT * c.p.M) {
        ok = false;
        std::printf("  gemm_mac_cycles %llu != NT*KT*M %llu\n", (unsigned long long)mac, (unsigned long long)(NT * KT * c.p.M));
      }
      if (ep < c.p.M * NT) { ok = false; std::printf("  epilogue cycles %llu < NT*M\n", (unsigned long long)ep); }
      totalCycles += uint64_t(cyc);
      totalMac += mac;
    }
    rep.tally(ok, what);
    if (!ok) std::printf("  (%s)\n", what);
    modeCount[int(c.p.mode)]++;
    i8Count += c.p.outI8;
    if ((c.rqAddr & 63) || (c.biasAddr & 63) || (c.auxAddr & 31) || (c.outAddr & 31)) ++splitCases;
  }
  std::printf("tb_gemm: modes NONE=%d RESADD=%d SILU=%d MUL=%d, i8 outputs=%d, cases with unaligned rq/bias/aux/out=%llu\n",
              modeCount[0], modeCount[1], modeCount[2], modeCount[3], i8Count, (unsigned long long)splitCases);
  std::printf("tb_gemm: total cycles=%llu mac cycles=%llu (MAC utilization %.1f%%)\n", (unsigned long long)totalCycles,
              (unsigned long long)totalMac, totalCycles ? 100.0 * double(totalMac) / double(totalCycles) : 0.0);
  tb.top.final();
  return rep.finish(fusion ? "tb_gemm[EPILOGUE_FUSION=1]" : "tb_gemm[EPILOGUE_FUSION=0]");
}
