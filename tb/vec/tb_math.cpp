// tb_math.cpp — unit test of isqrt.sv and udiv.sv against numerics.h::isqrt48 / udiv.
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "Vtb_math_top.h"
#include "llaccel/numerics.h"
#include "verilated.h"

namespace {

struct Case { uint64_t a48; uint32_t a32; uint32_t b24; uint32_t b32; };

int run(Vtb_math_top& top, uint64_t& cycle) {
  top.clk = 0; top.eval();
  top.clk = 1; top.eval();
  ++cycle;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vtb_math_top top;
  uint64_t cycle = 0;

  std::mt19937_64 rng(0x5eed1234ULL);
  std::vector<Case> cases;
  // edge values
  const uint64_t edge48[] = {0, 1, 2, 3, 4, 8, 15, 16, 17, 0xFFFFFFFFULL, (1ULL << 47), (1ULL << 48) - 1,
                             (1ULL << 48) - 2, 0xFFFFFF000000ULL, 0xFFFFFEFFFFFFULL, (uint64_t(1) << 46) + 1};
  const uint32_t edge32[] = {0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 65535, 65536, 1 << 31, 0xFFFFFE};
  for (uint64_t a : edge48) for (uint32_t b : edge32)
    cases.push_back({a, uint32_t(a & 0xFFFFFFFFu) ^ b, b & 0xFFFFFF, b});
  for (uint32_t a : edge32) for (uint32_t b : edge32) cases.push_back({uint64_t(a) << 16, a, b & 0xFFFFFF, b});
  // Exhaustive narrow dividers, including zero-as-one divisors.
  for (uint32_t a = 0; a < 4; ++a) for (uint32_t b = 0; b < 4; ++b)
    cases.push_back({a, a, b, b});
  // exact squares and squares +-1
  for (int i = 0; i < 400; ++i) {
    uint64_t r = rng() % (1ULL << 24);
    uint64_t s = r * r;
    for (int d = -1; d <= 1; ++d) {
      uint64_t v = uint64_t(int64_t(s) + d) & ((1ULL << 48) - 1);
      cases.push_back({v, uint32_t(rng()), uint32_t(rng() % (1u << 24)), uint32_t(rng())});
    }
  }
  // random, with mixed magnitudes
  while (cases.size() < 12000) {
    int sh = int(rng() % 49);
    uint64_t a48 = (rng() & ((1ULL << 48) - 1)) >> sh;
    uint32_t a32 = uint32_t(uint64_t(uint32_t(rng())) >> (rng() % 33));
    uint32_t b24 = uint32_t(rng() % (1u << 24)) >> (rng() % 25);
    uint32_t b32 = uint32_t(uint64_t(uint32_t(rng())) >> (rng() % 33));
    cases.push_back({a48, a32, b24, b32});
  }

  // reset
  top.rst_n = 0; top.start = 0; top.sq_a = 0; top.dv_a = 0; top.dv_b24 = 0; top.dv_b32 = 0;
  for (int i = 0; i < 3; ++i) run(top, cycle);
  top.rst_n = 1;
  run(top, cycle);

  uint64_t fails = 0, n = 0;
  for (unsigned shift = 0; shift < 64; ++shift) {
    for (unsigned trial = 0; trial < 256; ++trial) {
      const int64_t v = trial == 0 ? INT64_MAX : trial == 1 ? INT64_MIN : int64_t(rng());
      const int64_t expected = shift ? int64_t((__int128(v) + (__int128(1) << (shift - 1))) >> shift) : v;
      top.round_v = uint64_t(v); top.round_s = shift; top.eval();
      if (int64_t(top.round_q) != expected || llaccel::num::rshr(v, shift) != expected) ++fails;
    }
  }
  std::printf("rounding: 16384 full-width cases checked against 128-bit oracle\n");
  for (const Case& c : cases) {
    top.sq_a = c.a48; top.dv_a = c.a32; top.dv_b24 = c.b24; top.dv_b32 = c.b32;
    top.start = 1;
    run(top, cycle);
    top.start = 0;
    // done is registered at the last iteration edge: visible 24 edges (isqrt) / 32 edges (udiv)
    // after the edge that sampled start (see the timing notes in isqrt.sv / udiv.sv)
    int sq_lat = -1, d24_lat = -1, d32_lat = -1, d1_lat = -1, d2_lat = -1;
    for (int k = 1; k <= 40; ++k) {
      run(top, cycle);
      if (top.d1_done && d1_lat < 0) d1_lat = k;
      if (top.d2_done && d2_lat < 0) d2_lat = k;
      if (top.sq_done && sq_lat < 0) sq_lat = k;
      if (top.d24_done && d24_lat < 0) d24_lat = k;
      if (top.d32_done && d32_lat < 0) d32_lat = k;
    }
    uint32_t exp_sq = llaccel::num::isqrt48(c.a48);
    uint32_t exp_d24 = llaccel::num::udiv(c.a32, c.b24);
    uint32_t exp_d32 = llaccel::num::udiv(c.a32, c.b32);
    bool ok = true;
    if (top.d1_q != llaccel::num::udiv(c.a32 & 1, c.b32 & 1) || d1_lat != 1 ||
        top.d2_q != llaccel::num::udiv(c.a32 & 3, c.b32 & 3) || d2_lat != 2) {
      ok = false;
      std::printf("narrow udiv FAIL a=%u b=%u q1=%u lat1=%d q2=%u lat2=%d\n",
                  c.a32 & 3, c.b32 & 3, unsigned(top.d1_q), d1_lat, unsigned(top.d2_q), d2_lat);
    }
    // Also check the mathematical definition independently of the C++
    // reference, which intentionally uses the same bit-serial algorithm.
    const uint64_t root = top.sq_q;
    if (top.sq_q != exp_sq || sq_lat != 24 || root * root > c.a48 ||
        (root + 1) * (root + 1) <= c.a48) {
      ok = false;
      std::printf("isqrt FAIL a=%llx got=%u exp=%u lat=%d\n", (unsigned long long)c.a48, (unsigned)top.sq_q, exp_sq, sq_lat);
    }
    if (top.d24_q != exp_d24 || d24_lat != 32) {
      ok = false;
      std::printf("udiv32/24 FAIL a=%u b=%u got=%u exp=%u lat=%d\n", c.a32, c.b24, (unsigned)top.d24_q, exp_d24, d24_lat);
    }
    if (top.d32_q != exp_d32 || d32_lat != 32) {
      ok = false;
      std::printf("udiv32/32 FAIL a=%u b=%u got=%u exp=%u lat=%d\n", c.a32, c.b32, (unsigned)top.d32_q, exp_d32, d32_lat);
    }
    if (!ok) ++fails;
    ++n;
    if (fails > 20) break;
  }
  std::printf("tb_math: %llu operand sets (isqrt48, udiv 1/1, 2/2, 32/24, 32/32 each), %llu failures\n",
              (unsigned long long)n, (unsigned long long)fails);
  top.final();
  return fails ? 1 : 0;
}
