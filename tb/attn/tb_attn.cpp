// tb_attn.cpp — randomized, bit-exact tests of attn_engine (ATTN and KV_WRITE)
// against numerics.h::attention_head.
//
// SRAM and DRAM use separate fixture images. ATTN compares the whole SRAM
// against the reference and requires all DRAM unchanged; KV_WRITE requires
// unchanged SRAM and compares the whole dynamically sized DRAM image, including
// head padding, future slots and guard bytes.
// Random cases run at grant-denial probability 0 and 30 %; directed cases cover
// all-equal scores, the p = 0 path (huge score gap), a single key (T = 1) and a
// timing probe (D = 32, T = 64).
#include <algorithm>
#include <memory>
#include <fstream>
#include <iostream>
#include "tb_common.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "Vtb_attn_top.h"
#include "Vtb_attn_top___024root.h"
#include "llaccel/isa.h"
#include "llaccel/numerics.h"
#include "verilated.h"

using namespace llaccel;

namespace {

constexpr uint32_t kMem = kSramBytes;
// SRAM contains queries/output/source only. K/V use separately sized DRAM fixtures.
constexpr uint32_t kRegQ = 0x00000, kRegQSize = 0x20000;
constexpr uint32_t kRegK = 0x20000, kRegKSize = 0x40000;
constexpr uint32_t kRegO = 0xA0000, kRegOSize = 0x20000;
constexpr uint32_t kRegS = 0xC0000, kRegSSize = 0x20000;
constexpr int kNCases = 100;
// Wide D256/M16/T4096 cases serialize 64-lane MAC segments and DRAM pieces.
// Keep a generous bounded guard; actual runtime remains unmeasured.
constexpr uint64_t kTimeout = 100'000'000;

struct Tb {
  Vtb_attn_top top;
  uint64_t cycle = 0;
  bool profile = false;
  uint64_t readBytes = 0, writeBytes = 0;
  std::vector<uint8_t> dramImage, dramAfter;
  tb::DramDriver<Vtb_attn_top>* driver = nullptr;

  uint8_t* mem() { return reinterpret_cast<uint8_t*>(&top.rootp->tb_attn_top__DOT__u_sram__DOT__mem[0]); }

  void tick() {
    if (driver) driver->preEdge(top); else {top.clk = 0; top.eval();}
    top.clk = 1; top.eval();
    if (driver) driver->postEdge(top);
    top.clk = 0; top.eval();
    ++cycle;
  }
  void reset() {
    top.dram_req_ready=0; top.dram_rsp_valid=0;
    top.rst_n = 0; top.instr_valid = 0; top.pos = 0; top.deny_pct = 0;
    for (int i = 0; i < 16; ++i) top.instr_flat[i] = 0;
    for (int i = 0; i < 4; ++i) tick();
    top.rst_n = 1;
    tick();
  }
  // Runs one instruction; returns cycles from acceptance to done_pulse (or -1 on timeout / protocol error).
  long run(const Instr& in, uint32_t pos, uint8_t deny, uint64_t* stall_cycles = nullptr, uint64_t* mac_cycles = nullptr) {
    uint32_t dramBase = pos>256 ? 0x1000000 : 0;
    DramModel dram(uint64_t(dramImage.size())+dramBase, profile ? 100 : 13 + (pos % 101), profile ? 32 : 1 + (pos % 32));
    dram.write(dramBase,dramImage.data(),dramImage.size());
    std::vector<uint8_t> before(mem(),mem()+kMem);
    tb::DramDriver<Vtb_attn_top> drv(dram); drv.init(top); driver=&drv;
    struct ClearDriver { Tb& tb; ~ClearDriver() { tb.driver = nullptr; } } clearDriver{*this};
    for (int i = 0; i < 16; ++i) top.instr_flat[i] = in.w[i];
    if(in.op()==Op::ATTN) { top.instr_flat[4]+=dramBase; top.instr_flat[5]+=dramBase; }
    else if(in.op()==Op::KV_WRITE) top.instr_flat[3]+=dramBase;
    top.pos = pos;
    top.deny_pct = deny;
    top.instr_valid = 1;
    top.eval();
    uint64_t guard = 0;
    while (!top.instr_ready) { tick(); if (++guard > 1000) { std::printf("  engine never became ready\n"); return -1; } }
    tick();  // acceptance edge
    top.instr_valid = 0;
    top.eval();
    if (!top.busy) { std::printf("  busy not asserted after accept\n"); return -1; }
    long cycles = 0;
    uint64_t stalls = 0, macs = 0;
    while (!top.done_pulse) {
      if (top.perf_sram_stall) ++stalls;
      if (top.perf_mac_cycles) ++macs;
      tick();
      ++cycles;
      if (cycles > long(kTimeout)) { std::printf("  TIMEOUT\n"); return -1; }
    }
    if (top.done_sig_sem != in.signalSem()) { std::printf("  done_sig_sem mismatch\n"); return -1; }
    if (!top.busy) { std::printf("  busy low in the done_pulse cycle\n"); return -1; }
    tick();
    if (top.done_pulse) { std::printf("  done_pulse longer than one cycle\n"); return -1; }
    if (top.busy) { std::printf("  busy still high after done_pulse\n"); return -1; }
    if (stall_cycles) *stall_cycles = stalls;
    if (mac_cycles) *mac_cycles = macs;
    readBytes = dram.readBytes(); writeBytes = dram.writeBytes();
    driver=nullptr;
    if (in.op()==Op::KV_WRITE) {
      if(std::memcmp(mem(),before.data(),kMem)) throw std::runtime_error("KV_WRITE modified SRAM");
      dramAfter.assign(dram.data()+dramBase, dram.data()+dramBase+dramImage.size());
    } else if(std::memcmp(dram.data()+dramBase,dramImage.data(),dramImage.size())) throw std::runtime_error("ATTN modified DRAM");
    return cycles;
  }
};

struct Rng {
  std::mt19937_64 g;
  explicit Rng(uint64_t s) : g(s) {}
  uint32_t u32() { return uint32_t(g()); }
  uint32_t range(uint32_t lo, uint32_t hi) { return lo + uint32_t(g() % (uint64_t(hi) - lo + 1)); }  // inclusive
  bool pct(int p) { return int(g() % 100) < p; }
  int8_t i8() {
    if (pct(8)) {
      const int8_t e[] = {127, -128, -127, 0, 1, -1, 64, -64};
      return e[g() % 8];
    }
    return int8_t(g());
  }
  template <class T> T pick(std::initializer_list<T> l) { return *(l.begin() + g() % l.size()); }
};

uint32_t place(Rng& r, uint32_t region, uint32_t size, uint32_t need) {
  if (need > size || uint64_t(region) + size > UINT32_MAX)
    throw std::runtime_error("attention fixture region too small or address overflow");
  uint32_t slots = (size - need) / 16;
  return region + 16 * r.range(0, slots);
}

void fill_random(Rng& r, uint8_t* m, uint32_t addr, uint32_t bytes) {
  for (uint32_t i = 0; i < bytes; ++i) m[addr + i] = uint8_t(r.i8());
}

// Compares memory against expected; reports the first few differences.
bool check(Tb& tb, const std::vector<uint8_t>& expect, const char* name) {
  const uint8_t* m = tb.mem();
  int shown = 0;
  bool ok = true;
  for (uint32_t i = 0; i < kMem; ++i) {
    if (m[i] != expect[i]) {
      ok = false;
      if (shown++ < 6) std::printf("  %s: mismatch at %06x: got %02x expected %02x\n", name, i, m[i], expect[i]);
    }
  }
  return ok;
}

struct CaseResult { long cycles; uint64_t work; bool ok; uint64_t stalls; uint64_t macs; };

enum class Directed { None, EqualScores, HugeGap, SingleKey, Timing, MaxContext, Uniform171, OffsetBase, OffsetPositive, OffsetNegative, ScoreExtreme };

// One ATTN instruction. Random shape unless `dir` pins it.
CaseResult run_attn(Tb& tb, Rng& r, uint8_t deny, Directed dir = Directed::None, uint32_t dirD = 32, bool wide = false, uint32_t context = 0, uint32_t contextRows = 1) {
  uint32_t H = r.pick<uint32_t>({1, 2, 4, 8});
  uint32_t Hkv = H;
  while (Hkv > 1 && r.pct(50)) Hkv /= 2;   // any power-of-two divisor of H
  uint32_t D = r.pick<uint32_t>({16, 32, 64, 128, 256});
  uint32_t pos = r.range(0, 240), M = r.range(1, 16);
  if (dir != Directed::None) { D = dirD; H = Hkv = 1; }
  if (dir == Directed::SingleKey) { pos = 0; M = 1; }
  if (dir == Directed::Timing) { pos = 63; M = 1; }
  if (dir == Directed::MaxContext) { pos = 255; M = 1; H = 6; Hkv = 2; }
  if (dir == Directed::Uniform171) { pos = 170; M = 1; }
  if (dir >= Directed::OffsetBase) { pos = 1; M = 1; }
  if (context) { pos=context-contextRows; M=contextRows; H=Hkv=1; D=dirD; }
  uint32_t G = H / Hkv;
  uint32_t kvs = std::max(256u,context) * D + (context==4096 ? 0 : 16 * r.range(0, 3));
  uint32_t qb = M * H * D;
  uint32_t q = place(r, kRegQ, kRegQSize, qb), out = place(r, kRegO, kRegOSize, qb);
  const uint32_t regionSize = std::max(kRegKSize, ((Hkv * kvs + 255) / 256) * 256);
  const uint32_t vRegion = kRegK + regionSize;
  uint32_t kbase = place(r, kRegK, regionSize, Hkv * kvs), vbase = place(r, vRegion, regionSize, Hkv * kvs);
  tb.dramImage.assign(vRegion + regionSize + 64, 0xA5);
  uint8_t* dm = tb.dramImage.data();
  if (tb.profile) {
    // Naturally aligned D64 rows isolate the three-K plus one-V pass cost.
    kvs = context * D; q = kRegQ; out = kRegO; kbase = kRegK; vbase = vRegion;
  }
  uint8_t* m = tb.mem();
  fill_random(r, m, q, qb);
  fill_random(r, dm, kbase, Hkv * kvs);
  fill_random(r, dm, vbase, Hkv * kvs);
  uint32_t Ms = (1u << 30) + (r.u32() >> 2), Ss = r.range(28, 40);
  uint32_t Mo = (1u << 30) + (r.u32() >> 2), So = r.range(28, 40);
  if (dir == Directed::EqualScores) {
    std::memset(m + q, 0, qb);                        // every score is 0 -> every p is 65534
  } else if (dir == Directed::HugeGap) {
    // |q| >= 64; key t* equals q, every other key is -q: gap = 2*sum(q^2) >= 2^17,
    // and Ss = 28 makes z >= 2^19 for every non-max key -> p = 0 path.
    for (uint32_t i = 0; i < D; ++i) { int v = int(r.range(64, 127)); m[q + i] = uint8_t(int8_t(r.pct(50) ? v : -v)); }
    uint32_t T = pos + M;                            // keys 0..T-1 are visible to the last row
    uint32_t tstar = r.range(0, T - 1);
    for (uint32_t t = 0; t < T; ++t)
      for (uint32_t i = 0; i < D; ++i) dm[kbase + t * D + i] = (t == tstar) ? m[q + i] : uint8_t(-int8_t(m[q + i]));
    Ss = 28;
  }
  if (dir == Directed::Uniform171) {
    std::memset(m + q, 0, qb);
    std::memset(dm + vbase, 127, Hkv * kvs);
    Ms = 1; Ss = 0; Mo = 1u << 30; So = 38;
  }
  if (dir >= Directed::OffsetBase) {
    // Two logits with an identical additive offset must preserve their gap.
    // Scaling before max subtraction and clipping to i16 would flatten both
    // large-positive/negative cases into an incorrect uniform distribution.
    std::memset(m + q, 0, qb);
    std::memset(dm + kbase, 0, Hkv * kvs);
    std::memset(dm + vbase, 0, Hkv * kvs);
    m[q] = 127;
    const int offset = dir == Directed::OffsetPositive ? 126 :
                       dir == Directed::OffsetNegative ? -128 : 0;
    dm[kbase] = uint8_t(int8_t(offset));
    dm[kbase + D] = uint8_t(int8_t(offset + 1));
    dm[vbase] = uint8_t(int8_t(-127)); dm[vbase + D] = 127;
    Ms = 1u << 30; Ss = 24; Mo = 1u << 30; So = 38;
    if (dir == Directed::ScoreExtreme) {
      std::memset(m + q, 127, qb);
      std::memset(dm + kbase, 128, D);
      std::memset(dm + kbase + D, 127, D);
      Ms = UINT32_MAX; Ss = 0;
    }
  }
  std::vector<uint8_t> expect(m, m + kMem);
  Instr in(Op::ATTN, wide ? kFlagAttnWideProb : 0);
  in.setSignal(uint8_t(r.range(0, 31)));
  in[2] = q; in[3] = out; in[4] = kbase; in[5] = vbase; in[6] = M; in[7] = H; in[8] = Hkv; in[9] = D;
  in[10] = kvs; in[11] = Ms; in[12] = Ss; in[13] = Mo; in[14] = So;
  std::vector<int32_t> scores;
  std::vector<uint16_t> probs;
  std::vector<int8_t> qrow(D), orow(D);
  uint64_t keys = 0;
  for (uint32_t mm = 0; mm < M; ++mm) {
    uint32_t T = pos + mm + 1;
    for (uint32_t h = 0; h < H; ++h) {
      uint32_t kvh = h / G;
      for (uint32_t d = 0; d < D; ++d) qrow[d] = int8_t(m[q + (mm * H + h) * D + d]);
      auto keyAt = [&](uint32_t t) { return reinterpret_cast<const int8_t*>(dm + kbase + kvh * kvs + t * D); };
      auto valAt = [&](uint32_t t) { return reinterpret_cast<const int8_t*>(dm + vbase + kvh * kvs + t * D); };
      num::attention_head(qrow, D, T, keyAt, valAt, Ms, Ss, Mo, So, orow, scores, probs, wide);
      for (uint32_t d = 0; d < D; ++d) expect[out + (mm * H + h) * D + d] = uint8_t(orow[d]);
      keys += T;
    }
  }
  uint64_t stalls = 0, macs = 0;
  long cyc = tb.run(in, pos, deny, &stalls, &macs);
  bool ok = cyc >= 0 && check(tb, expect, "ATTN");
  if (dir == Directed::Uniform171 && wide && int8_t(tb.mem()[out]) != 127) ok = false;
  if (dir >= Directed::OffsetBase && int8_t(tb.mem()[out]) != 127) {
    std::printf("  additive-offset independent oracle expected127, got%d\n", int8_t(tb.mem()[out]));
    ok = false;
  }
  const uint64_t expectedMacs = 4 * keys * ((D + 63) / 64);
  if (ok && macs != expectedMacs) { ok = false; std::printf("  ATTN: perf_mac_cycles=%llu, expected %llu (4 passes per 64-lane segment)\n", (unsigned long long)macs, (unsigned long long)expectedMacs); }
  if (!ok) std::printf("  ATTN case: M=%u H=%u Hkv=%u D=%u pos=%u kvs=%u q=%06x out=%06x k=%06x v=%06x Ms=%u Ss=%u Mo=%u So=%u\n",
                       M, H, Hkv, D, pos, kvs, q, out, kbase, vbase, Ms, Ss, Mo, So);
  return {cyc, keys, ok, stalls, macs};
}

CaseResult run_kv(Tb& tb, Rng& r, uint8_t deny, uint32_t directedD = 0, uint32_t context = 256) {
  uint32_t Hkv = r.pick<uint32_t>({1, 2, 4, 8}), D = r.pick<uint32_t>({16, 32, 64, 128, 256});
  uint32_t pos = r.range(0, 240), M = r.range(1, 16);
  if (directedD) { D = directedD; Hkv = 3; M = 16; pos = context - M; }
  uint32_t kvs = context * D + 16 * r.range(0, 3);
  uint32_t sb = M * Hkv * D;
  const uint32_t regionSize = std::max(kRegKSize, ((Hkv * kvs + 255) / 256) * 256);
  uint32_t src = place(r, kRegS, kRegSSize, sb), base = place(r, kRegK, regionSize, Hkv * kvs);
  tb.dramImage.assign(kRegK + regionSize + 64, 0xA5);
  uint8_t* m = tb.mem();
  fill_random(r, m, src, sb);
  std::vector<uint8_t> expect = tb.dramImage;
  for (uint32_t mm = 0; mm < M; ++mm)
    for (uint32_t kvh = 0; kvh < Hkv; ++kvh)
      std::memcpy(expect.data() + base + kvh * kvs + (pos + mm) * D, m + src + (mm * Hkv + kvh) * D, D);
  Instr in(Op::KV_WRITE);
  in.setSignal(uint8_t(r.range(0, 31)));
  in[2] = src; in[3] = base; in[4] = M; in[5] = Hkv; in[6] = D; in[7] = kvs;
  uint64_t stalls = 0, macs = 0;
  long cyc = tb.run(in, pos, deny, &stalls, &macs);
  bool ok = cyc >= 0 && (tb.dramAfter == expect);
  if (ok && macs != 0) { ok = false; std::printf("  KV_WRITE: perf_mac_cycles=%llu, expected 0\n", (unsigned long long)macs); }
  if (!ok) std::printf("  KV_WRITE case: M=%u Hkv=%u D=%u pos=%u kvs=%u src=%06x base=%06x\n", M, Hkv, D, pos, kvs, src, base);
  return {cyc, uint64_t(M) * Hkv, ok, stalls, macs};
}

struct Stats {
  long cases = 0, fails = 0;
  uint64_t cycles = 0, work = 0, stalls = 0;
};

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  uint64_t seed = 1;
  for (int i = 1; i < argc; ++i) if (std::strncmp(argv[i], "+seed=", 6) == 0) seed = std::strtoull(argv[i] + 6, nullptr, 10);
  Tb tb;
  tb.reset();
  {
    Rng r(seed ^ 0xABCDEF);   // random background so untouched bytes are non-trivial
    uint8_t* m = tb.mem();
    for (uint32_t i = 0; i < kMem; ++i) m[i] = uint8_t(r.u32());
  }

  // +profile[=output.json] runs only the bounded bandwidth experiment.
  // The ordinary invocation below retains every existing regression case.
  std::string profilePath;
  bool profile = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "+profile") == 0) profile = true;
    if (std::strncmp(argv[i], "+profile=", 9) == 0) { profile = true; profilePath = argv[i] + 9; }
  }
  if (profile) {
    tb.profile = true;
    std::ofstream file;
    if (!profilePath.empty()) { file.open(profilePath); if (!file) throw std::runtime_error("cannot open profile output"); }
    std::ostream& out = profilePath.empty() ? std::cout : file;
    out << "{\n  \"seed\": " << seed << ", \"dram_latency\": 100, \"max_outstanding\": 32, \"head_dim\": 64, \"query_heads\": 1, \"query_rows\": 1, \"tile_rows\": 256,\n  \"measurements\": [\n";
    bool first = true, passed = true;
    for (uint32_t context : {32u,128u,256u,512u,1024u,2048u,4096u}) {
      Rng rng(seed + context);
      auto cr = run_attn(tb, rng, 0, Directed::None, 64, true, context);
      const uint64_t expected = 4ULL * context * 64;
      const bool ok = cr.ok && tb.readBytes == expected && tb.writeBytes == 0;
      passed &= ok;
      if (!first) out << ",\n";
      first = false;
      out << "    {\"context\": " << context << ", \"cycles\": " << cr.cycles
          << ", \"dram_read_bytes\": " << tb.readBytes << ", \"dram_write_bytes\": " << tb.writeBytes
          << ", \"expected_three_k_plus_v_bytes\": " << expected
          << ", \"single_k_plus_v_logical_bytes\": " << 2ULL*context*64
          << ", \"result\": \"" << (ok ? "MATCH" : "FAIL") << "\"}";
    }
    out << "\n  ],\n  \"status\": \"" << (passed ? "PASS" : "FAIL") << "\"\n}\n";
    tb.top.final();
    return passed ? 0 : 1;
  }

  long total_fail = 0, total_cases = 0;
  for (uint8_t deny : {uint8_t(0), uint8_t(30)}) {
    for (int t = 0; t < 2; ++t) {
      Rng r(seed * 1000003ULL + uint64_t(t) * 7919ULL + deny);
      Stats st;
      for (int c = 0; c < kNCases; ++c) {
        CaseResult cr = (t == 0) ? run_attn(tb, r, deny) : run_kv(tb, r, deny);
        ++st.cases;
        if (!cr.ok) ++st.fails;
        if (cr.cycles >= 0) { st.cycles += uint64_t(cr.cycles); st.work += cr.work; st.stalls += cr.stalls; }
        if (st.fails >= 3) break;
      }
      std::printf("%-9s deny=%2u%%: %ld cases, %ld failures, %.3f cycles/%s (%llu cycles, %llu %s, %llu stall cycles)\n",
                  t == 0 ? "ATTN" : "KV_WRITE", unsigned(deny), st.cases, st.fails,
                  st.work ? double(st.cycles) / double(st.work) : 0.0, t == 0 ? "key" : "row",
                  (unsigned long long)st.cycles, (unsigned long long)st.work, t == 0 ? "keys" : "rows", (unsigned long long)st.stalls);
      total_fail += st.fails;
      total_cases += st.cases;
    }
  }
  // directed cases: all-equal scores, huge gap (p = 0 path), single key; every D, both denial rates
  for (uint8_t deny : {uint8_t(0), uint8_t(30)}) {
    for (uint32_t D : {16u, 32u, 64u}) {
      struct { Directed d; const char* name; } dirs[] = {
          {Directed::EqualScores, "equal-scores"}, {Directed::HugeGap, "huge-gap"}, {Directed::SingleKey, "single-key"}, {Directed::MaxContext, "T256-GQA3"}};
      for (auto& dd : dirs) {
        Rng r(seed * 31ULL + D * 7ULL + deny + uint64_t(dd.d));
        CaseResult cr = run_attn(tb, r, deny, dd.d, D);
        std::printf("ATTN directed %-12s D=%2u deny=%2u%%: %s (%ld cycles)\n", dd.name, D, unsigned(deny), cr.ok ? "ok" : "FAIL", cr.cycles);
        ++total_cases;
        if (!cr.ok) ++total_fail;
      }
    }
  }
  // New Q0.15 mode: independently verify constant-value invariance and
  // randomized shapes/backpressure while retaining the legacy-mode suite.
  for (uint8_t deny : {uint8_t(0), uint8_t(30)}) {
    Rng rngWide(seed + deny + 901);
    for (unsigned i=0; i<100; ++i) {
      auto cr = run_attn(tb, rngWide, deny, Directed::None, 32, true);
      ++total_cases; if (!cr.ok) ++total_fail;
    }
    for (uint32_t dim : {16u,32u,64u}) {
      auto cr = run_attn(tb, rngWide, deny, Directed::Uniform171, dim, true);
      ++total_cases; if (!cr.ok) ++total_fail;
    }
  }
  // Offset invariance, both probability formats, all supported head widths.
  // Independently known winner output127, including maximal unsigned Ms.
  for (uint8_t deny : {uint8_t(0), uint8_t(30)})
    for (uint32_t dim : {16u, 32u, 64u})
      for (bool wide : {false, true})
        for (Directed dir : {Directed::OffsetBase, Directed::OffsetPositive,
                             Directed::OffsetNegative, Directed::ScoreExtreme}) {
          Rng rngOffset(seed + 3101 + uint64_t(dir));
          auto cr = run_attn(tb, rngOffset, deny, dir, dim, wide);
          ++total_cases; if (!cr.ok) ++total_fail;
        }
  for (uint32_t context : {255u,256u,257u,511u,512u,513u,1023u,1024u,4095u,4096u})
    for (uint8_t deny : {uint8_t(0),uint8_t(30)}) {
      Rng r(seed+context+deny);
      auto cr=run_attn(tb,r,deny,Directed::None,16,true,context);
      std::printf("ATTN DRAM tiled T=%u deny=%u: %s (%ld cycles)\n",context,deny,cr.ok?"ok":"FAIL",cr.cycles);
      ++total_cases; if(!cr.ok) ++total_fail;
    }
  for(uint32_t dim : {16u,32u,64u}) for(uint32_t context : {257u,513u,1024u,4096u}) {
    Rng r(seed+dim+context); auto cr=run_attn(tb,r,30,Directed::None,dim,true,context,4);
    std::printf("ATTN DRAM multirow T=%u D=%u: %s\n",context,dim,cr.ok?"ok":"FAIL");
    ++total_cases; if(!cr.ok) ++total_fail;
  }
  // Simultaneously exercise the maximum row count and last legal cache slot.
  // POS=4080 gives causal lengths 4081..4096; the shared oracle also checks
  // all untouched SRAM bytes under randomized SRAM denial and DRAM
  // response/credit backpressure.
  for (uint32_t dim : {16u, 32u, 64u}) {
    Rng r(seed + 1604096 + dim);
    auto cr = run_attn(tb, r, 30, Directed::None, dim, true, 4096, 16);
    std::printf("ATTN DRAM max-row/context T=4096 M=16 D=%u deny=30: %s (%ld cycles)\n",
                dim, cr.ok ? "ok" : "FAIL", cr.cycles);
    ++total_cases;
    if (!cr.ok) ++total_fail;
  }
  // Non-power-of-two grouping at both extended head widths (H/Hkv = 3).
  for (uint32_t dim : {128u, 256u}) {
    Rng r(seed + 1700300 + dim);
    auto cr = run_attn(tb, r, 30, Directed::MaxContext, dim, true);
    std::printf("ATTN wide-head GQA3 T=256 D=%u: %s\n", dim, cr.ok ? "ok" : "FAIL");
    ++total_cases; if (!cr.ok) ++total_fail;
  }
  // wide heads: tile tail, full row count and last context slot.
  for (uint32_t dim : {128u, 256u}) for (uint32_t context : {257u, 4096u}) {
    Rng r(seed + 1704096 + dim + context);
    auto cr = run_attn(tb, r, 30, Directed::None, dim, true, context, 16);
    std::printf("ATTN wide-head T=%u M=16 D=%u: %s\n", context, dim, cr.ok ? "ok" : "FAIL");
    ++total_cases; if (!cr.ok) ++total_fail;
    auto kv = run_kv(tb, r, 30, dim, context);
    std::printf("KV_WRITE wide-head T=%u M=16 Hkv=3 D=%u: %s\n", context, dim, kv.ok ? "ok" : "FAIL");
    ++total_cases; if (!kv.ok) ++total_fail;
  }
  // timing probe: one query, D = 32, T = 64 keys, no denial
  {
    Rng r(seed + 99);
    CaseResult cr = run_attn(tb, r, 0, Directed::Timing, 32);
    std::printf("ATTN directed M=1 H=1 D=32 T=64 deny=0: %ld cycles accept->done, %llu mac cycles, %llu stall cycles (%s)\n",
                cr.cycles, (unsigned long long)cr.macs, (unsigned long long)cr.stalls, cr.ok ? "ok" : "FAIL");
    ++total_cases;
    if (!cr.ok) ++total_fail;
  }
  std::printf("tb_attn: %ld cases, %ld failures -> %s\n", total_cases, total_fail, total_fail ? "FAIL" : "PASS");
  tb.top.final();
  return total_fail ? 1 : 0;
}
