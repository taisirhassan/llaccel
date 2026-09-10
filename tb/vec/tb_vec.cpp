// tb_vec.cpp — randomized, bit-exact tests of vec_engine against numerics.h.
//
// Every VEC op: >= NCASES random cases at grant-denial probability 0 and 30 %.
// Each case fills SRAM with random data, runs one instruction, and compares
// the whole 1 MiB against a reference image (the snapshot with the destination
// region replaced by the numerics.h result), so stray writes are caught too.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "Vtb_vec_top.h"
#include "Vtb_vec_top___024root.h"
#include "llaccel/isa.h"
#include "llaccel/numerics.h"
#include "verilated.h"

using namespace llaccel;

namespace {

constexpr uint32_t kMem = kSramBytes;
constexpr uint32_t kRegA = 0x00000, kRegB = 0x40000, kRegD = 0x80000, kRegT = 0xC0000, kRegSize = 0x40000;
constexpr int kNCases = 100;
constexpr uint64_t kTimeout = 4'000'000;

struct Tb {
  Vtb_vec_top top;
  uint64_t cycle = 0;
  std::vector<uint8_t> snap;

  uint8_t* mem() { return reinterpret_cast<uint8_t*>(&top.rootp->tb_vec_top__DOT__u_sram__DOT__mem[0]); }

  void tick() {
    top.clk = 0; top.eval();
    top.clk = 1; top.eval();
    ++cycle;
  }
  void reset() {
    top.rst_n = 0; top.instr_valid = 0; top.pos = 0; top.deny_pct = 0;
    for (int i = 0; i < 16; ++i) top.instr_flat[i] = 0;
    for (int i = 0; i < 4; ++i) tick();
    top.rst_n = 1;
    tick();
  }
  // Runs one instruction; returns cycles from acceptance to done_pulse (or -1 on timeout / protocol error).
  long run(const Instr& in, uint32_t pos, uint8_t deny, uint64_t* stall_cycles = nullptr, uint64_t* rd_grants = nullptr) {
    for (int i = 0; i < 16; ++i) top.instr_flat[i] = in.w[i];
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
    uint64_t stalls = 0, grants = 0;
    while (!top.done_pulse) {
      if (top.perf_sram_stall) ++stalls;
      grants += top.rd0_active + top.rd1_active;
      tick();
      ++cycles;
      if (cycles > long(kTimeout)) { std::printf("  TIMEOUT\n"); return -1; }
    }
    if (top.done_sig_sem != in.signalSem()) { std::printf("  done_sig_sem mismatch\n"); return -1; }
    if (!top.busy) { std::printf("  busy low in the done_pulse cycle\n"); return -1; }
    tick();
    if (top.done_pulse) { std::printf("  done_pulse longer than one cycle\n"); return -1; }
    if (stall_cycles) *stall_cycles = stalls;
    if (rd_grants) *rd_grants = grants;
    return cycles;
  }
};

struct Rng {
  std::mt19937_64 g;
  explicit Rng(uint64_t s) : g(s) {}
  uint64_t u64() { return g(); }
  uint32_t u32() { return uint32_t(g()); }
  uint32_t range(uint32_t lo, uint32_t hi) { return lo + uint32_t(g() % (uint64_t(hi) - lo + 1)); }  // inclusive
  bool pct(int p) { return int(g() % 100) < p; }
  int16_t i16() {
    if (pct(8)) {
      const int16_t e[] = {32767, -32768, -32767, 0, 1, -1, 16384, -16384, 255, -256};
      return e[g() % 10];
    }
    return int16_t(g());
  }
  template <class T> T pick(std::initializer_list<T> l) { return *(l.begin() + g() % l.size()); }
};

void put16(uint8_t* m, uint32_t addr, int16_t v) { m[addr] = uint8_t(v & 0xFF); m[addr + 1] = uint8_t((uint16_t(v) >> 8) & 0xFF); }
int16_t get16(const uint8_t* m, uint32_t addr) { return int16_t(uint16_t(m[addr]) | (uint16_t(m[addr + 1]) << 8)); }

uint32_t place(Rng& r, uint32_t region, uint32_t need) {
  uint32_t slots = (kRegSize - need) / 16;
  return region + 16 * r.range(0, slots);
}

void fill_random(Rng& r, uint8_t* m, uint32_t addr, uint32_t bytes) {
  for (uint32_t i = 0; i < bytes; i += 2) put16(m, addr + i, r.i16());
}

struct Stats {
  long cases = 0, fails = 0;
  uint64_t cycles = 0, vectors = 0, stalls = 0;
};

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

// ---- individual op generators --------------------------------------------------------------------
struct CaseResult { long cycles; uint64_t vectors; bool ok; uint64_t stalls; };

CaseResult run_flat(Tb& tb, Rng& r, Op op, uint8_t deny) {
  uint32_t count = 16 * r.range(1, 512);
  if (r.pct(10)) count = 16 * r.range(1, 8);
  uint32_t bytes = count * 2;
  bool two = (op == Op::VEC_MUL || op == Op::VEC_ADD);
  uint32_t a = place(r, kRegA, bytes), b = place(r, kRegB, bytes), d = place(r, kRegD, bytes);
  bool inplace = r.pct(15);
  if (inplace) d = a;
  uint8_t* m = tb.mem();
  fill_random(r, m, a, bytes);
  if (two) fill_random(r, m, b, bytes);
  std::vector<uint8_t> expect(m, m + kMem);

  Instr in(op);
  in.setSignal(uint8_t(r.range(0, 31)));
  in[2] = a;
  uint32_t Mi = 0, Si = 0, sho = 0, sh = 0, Mq = 0, Sq = 0;
  if (two) { in[3] = b; in[4] = d; in[5] = count; }
  else     { in[3] = d; in[4] = count; }
  switch (op) {
    case Op::VEC_SILU: Mi = 1u << 30; Si = r.range(26, 34); sho = r.range(12, 20); in[5] = Mi; in[6] = Si; in[7] = sho; break;
    case Op::VEC_MUL:  sh = r.range(8, 24); in[6] = sh; break;
    case Op::VEC_ADD:  sh = r.range(0, 16); in[6] = sh; break;
    case Op::VEC_QUANT: Mq = (1u << 30) + (r.u32() >> 2); Sq = r.range(24, 40); in[5] = Mq; in[6] = Sq; break;
    default: break;
  }
  for (uint32_t i = 0; i < count; ++i) {
    int64_t x = get16(m, a + 2 * i);
    int64_t y = two ? get16(m, b + 2 * i) : 0;
    switch (op) {
      case Op::VEC_SILU: put16(expect.data(), d + 2 * i, int16_t(num::silu16(x, Mi, Si, sho))); break;
      case Op::VEC_MUL:  put16(expect.data(), d + 2 * i, int16_t(num::vmul(x, y, sh))); break;
      case Op::VEC_ADD:  put16(expect.data(), d + 2 * i, int16_t(num::vadd(x, y, sh))); break;
      case Op::VEC_QUANT: expect[d + i] = uint8_t(int8_t(num::vquant(x, Mq, Sq))); break;
      default: break;
    }
  }
  uint64_t stalls = 0;
  long cyc = tb.run(in, 0, deny, &stalls);
  bool ok = cyc >= 0 && check(tb, expect, std::string(opName(op)).c_str());
  if (!ok) std::printf("  %s case: count=%u a=%06x b=%06x d=%06x inplace=%d params=%u %u %u %u %u %u\n", std::string(opName(op)).c_str(),
                       count, a, b, d, inplace, Mi, Si, sho, sh, Mq, Sq);
  return {cyc, count / 16, ok, stalls};
}

CaseResult run_rmsnorm(Tb& tb, Rng& r, uint8_t deny, bool directed_k128 = false) {
  uint32_t M = directed_k128 ? 1 : r.range(1, 16);
  uint32_t K = directed_k128 ? 128 : r.pick<uint32_t>({16, 32, 64, 128, 384, 896});
  uint32_t bytes = M * K * 2;
  uint32_t src = place(r, kRegA, bytes), gamma = place(r, kRegB, K * 2), dst = place(r, kRegD, bytes);
  if (r.pct(15)) dst = src;
  uint8_t* m = tb.mem();
  fill_random(r, m, src, bytes);
  fill_random(r, m, gamma, K * 2);
  if (r.pct(20)) for (uint32_t i = 0; i < K; ++i) put16(m, gamma + 2 * i, int16_t(r.range(0, 4096)));  // "normal" gamma
  std::vector<uint8_t> expect(m, m + kMem);
  uint32_t eps = r.pct(20) ? 0 : r.range(0, 1u << 20);
  uint32_t C = r.pct(20) ? 0xFFFFFFFFu : r.u32();
  if (r.pct(20)) C = r.range(1, 1u << 20);
  uint32_t sh = r.range(16, 40);
  Instr in(Op::VEC_RMSNORM);
  in.setSignal(uint8_t(r.range(0, 31)));
  in[2] = src; in[3] = gamma; in[4] = dst; in[5] = M; in[6] = K; in[7] = eps; in[8] = C; in[9] = sh;
  std::vector<int16_t> x(K), g(K), y(K);
  for (uint32_t k = 0; k < K; ++k) g[k] = get16(m, gamma + 2 * k);
  for (uint32_t mm = 0; mm < M; ++mm) {
    for (uint32_t k = 0; k < K; ++k) x[k] = get16(m, src + mm * K * 2 + 2 * k);
    num::rmsnorm(x, g, y, eps, C, sh);
    for (uint32_t k = 0; k < K; ++k) put16(expect.data(), dst + mm * K * 2 + 2 * k, y[k]);
  }
  uint64_t stalls = 0;
  long cyc = tb.run(in, 0, deny, &stalls);
  bool ok = cyc >= 0 && check(tb, expect, "RMSNORM");
  if (!ok) std::printf("  RMSNORM case: M=%u K=%u src=%06x gamma=%06x dst=%06x eps=%u C=%u sh=%u\n", M, K, src, gamma, dst, eps, C, sh);
  return {cyc, uint64_t(M) * K / 16, ok, stalls};
}

CaseResult run_rope(Tb& tb, Rng& r, uint8_t deny) {
  uint32_t M = r.range(1, 16), H = r.range(1, 8), D = r.pick<uint32_t>({16, 32, 64});
  uint32_t pos = r.range(0, 256 - M);
  uint32_t stride = 2 * D + 16 * r.range(0, 3);
  uint32_t bytes = M * H * D * 2;
  uint32_t src = place(r, kRegA, bytes), dst = place(r, kRegD, bytes), tab = place(r, kRegT, 256 * stride);
  if (r.pct(15)) dst = src;
  uint8_t* m = tb.mem();
  fill_random(r, m, src, bytes);
  bool plausible = r.pct(50);
  for (uint32_t p = 0; p < 256; ++p)
    for (uint32_t i = 0; i < D; ++i)
      put16(m, tab + p * stride + 2 * i, plausible ? int16_t(int32_t(r.range(0, 32768)) - 16384) : r.i16());
  std::vector<uint8_t> expect(m, m + kMem);
  Instr in(Op::VEC_ROPE);
  in.setSignal(uint8_t(r.range(0, 31)));
  in[2] = src; in[3] = dst; in[4] = M; in[5] = H; in[6] = D; in[7] = tab; in[8] = stride;
  std::vector<int16_t> x(H * D), y(H * D), c(D / 2), s(D / 2);
  for (uint32_t mm = 0; mm < M; ++mm) {
    uint32_t p = pos + mm;
    for (uint32_t i = 0; i < H * D; ++i) x[i] = get16(m, src + mm * H * D * 2 + 2 * i);
    for (uint32_t i = 0; i < D / 2; ++i) { c[i] = get16(m, tab + p * stride + 2 * i); s[i] = get16(m, tab + p * stride + D + 2 * i); }
    num::rope(x, y, H, D, c, s);
    for (uint32_t i = 0; i < H * D; ++i) put16(expect.data(), dst + mm * H * D * 2 + 2 * i, y[i]);
  }
  uint64_t stalls = 0;
  long cyc = tb.run(in, pos, deny, &stalls);
  bool ok = cyc >= 0 && check(tb, expect, "ROPE");
  if (!ok) std::printf("  ROPE case: M=%u H=%u D=%u pos=%u src=%06x dst=%06x tab=%06x stride=%u\n", M, H, D, pos, src, dst, tab, stride);
  return {cyc, uint64_t(M) * H * D / 16, ok, stalls};
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  uint64_t seed = 1;
  for (int i = 1; i < argc; ++i) if (std::strncmp(argv[i], "+seed=", 6) == 0) seed = std::strtoull(argv[i] + 6, nullptr, 10);
  Tb tb;
  tb.reset();
  // random background so untouched bytes are non-trivial
  {
    Rng r(seed ^ 0xABCDEF);
    uint8_t* m = tb.mem();
    for (uint32_t i = 0; i < kMem; ++i) m[i] = uint8_t(r.u32());
  }

  struct OpTest { const char* name; int id; };
  const OpTest tests[] = {{"VEC_SILU", 0}, {"VEC_MUL", 1}, {"VEC_ADD", 2}, {"VEC_QUANT", 3}, {"VEC_RMSNORM", 4}, {"VEC_ROPE", 5}};
  long total_fail = 0, total_cases = 0;
  for (uint8_t deny : {uint8_t(0), uint8_t(30)}) {
    for (const OpTest& t : tests) {
      Rng r(seed * 1000003ULL + uint64_t(t.id) * 7919ULL + deny);
      Stats st;
      for (int c = 0; c < kNCases; ++c) {
        CaseResult cr;
        switch (t.id) {
          case 0: cr = run_flat(tb, r, Op::VEC_SILU, deny); break;
          case 1: cr = run_flat(tb, r, Op::VEC_MUL, deny); break;
          case 2: cr = run_flat(tb, r, Op::VEC_ADD, deny); break;
          case 3: cr = run_flat(tb, r, Op::VEC_QUANT, deny); break;
          case 4: cr = run_rmsnorm(tb, r, deny); break;
          default: cr = run_rope(tb, r, deny); break;
        }
        ++st.cases;
        if (!cr.ok) ++st.fails;
        if (cr.cycles >= 0) { st.cycles += uint64_t(cr.cycles); st.vectors += cr.vectors; st.stalls += cr.stalls; }
        if (st.fails >= 3) break;
      }
      std::printf("%-12s deny=%2u%%: %ld cases, %ld failures, %.3f cycles/vector (%llu cycles, %llu vectors, %llu stall cycles)\n",
                  t.name, unsigned(deny), st.cases, st.fails, st.vectors ? double(st.cycles) / double(st.vectors) : 0.0,
                  (unsigned long long)st.cycles, (unsigned long long)st.vectors, (unsigned long long)st.stalls);
      total_fail += st.fails;
      total_cases += st.cases;
    }
  }
  // directed: one K=128 RMSNorm row, no denial
  {
    Rng r(seed + 99);
    CaseResult cr = run_rmsnorm(tb, r, 0, true);
    std::printf("RMSNORM directed M=1 K=128 deny=0: %ld cycles accept->done (%s)\n", cr.cycles, cr.ok ? "ok" : "FAIL");
    ++total_cases;
    if (!cr.ok) ++total_fail;
  }
  std::printf("tb_vec: %ld cases, %ld failures -> %s\n", total_cases, total_fail, total_fail ? "FAIL" : "PASS");
  tb.top.final();
  return total_fail ? 1 : 0;
}
