// Executable form of docs/NUMERICS.md. Header-only, no floating point.
// Used by the functional ISA simulator, the Verilator unit testbenches (as the
// reference) and the compiler (for self-checks). The numpy golden model
// (python/llaccel/golden.py) implements exactly the same formulas.
#pragma once
#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "llaccel/luts.h"

namespace llaccel::num {

inline constexpr int64_t rshr(int64_t v, uint32_t s) {
  if (s == 0) return v;
  return (v + (int64_t(1) << (s - 1))) >> s;
}
inline constexpr int64_t sat8(int64_t v) { return std::clamp<int64_t>(v, -128, 127); }
inline constexpr int64_t satu8(int64_t v) { return std::clamp<int64_t>(v, 0, 255); }
inline constexpr int64_t sat16(int64_t v) { return std::clamp<int64_t>(v, -32768, 32767); }
inline constexpr int64_t satu16(int64_t v) { return std::clamp<int64_t>(v, 0, 65535); }
inline constexpr int64_t mulshift(int64_t v, uint32_t M, uint32_t S) { return rshr(v * int64_t(M), S); }

// floor(sqrt(t)) for t < 2^48, bit-serial exactly as the RTL does it.
inline constexpr uint32_t isqrt48(uint64_t t) {
  uint64_t rem = 0, root = 0;
  for (int i = 23; i >= 0; --i) {
    rem = (rem << 2) | ((t >> (2 * i)) & 3);
    root <<= 1;
    uint64_t trial = (root << 1) | 1;
    if (trial <= rem) { rem -= trial; root |= 1; }
  }
  return uint32_t(root);
}
inline constexpr uint32_t udiv(uint64_t a, uint64_t b) { return uint32_t(a / (b ? b : 1)); }

// ---- GEMM epilogue -------------------------------------------------------------
struct RqEntry { int32_t M; int32_t S; };

inline int64_t requant16(int32_t acc, RqEntry rq) { return sat16(mulshift(acc, uint32_t(rq.M), uint32_t(rq.S))); }
inline int64_t requant8(int32_t acc, RqEntry rq) { return sat8(mulshift(acc, uint32_t(rq.M), uint32_t(rq.S))); }

// ---- SiLU -----------------------------------------------------------------------
inline int64_t silu16(int64_t x, uint32_t Mi, uint32_t Si, uint32_t shOut) {
  int64_t u = sat16(mulshift(x, Mi, Si));
  int64_t idx = (u >> 8) + 128;  // arithmetic shift: u in [-32768, 32767] -> idx in [0, 255]
  int64_t f = u & 255;
  int64_t sg = int64_t(kSigmoidLut[idx]) + (((int64_t(kSigmoidLut[idx + 1]) - int64_t(kSigmoidLut[idx])) * f) >> 8);
  return sat16(rshr(x * sg, shOut));
}

// ---- elementwise ----------------------------------------------------------------------
inline int64_t vmul(int64_t a, int64_t b, uint32_t sh) { return sat16(rshr(a * b, sh)); }
inline int64_t vadd(int64_t a, int64_t b, uint32_t shB) { return sat16(a + rshr(b, shB)); }
inline int64_t vquant(int64_t x, uint32_t M, uint32_t S) { return sat8(mulshift(x, M, S)); }

// ---- RMSNorm ----------------------------------------------------------------------------
// x, g: K elements; writes K outputs.
inline void rmsnorm(std::span<const int16_t> x, std::span<const int16_t> g, std::span<int16_t> y,
                    uint32_t epsT, uint32_t C, uint32_t shPost) {
  uint64_t ss = 0;
  for (auto v : x) ss += uint64_t(int64_t(v) * int64_t(v));
  uint64_t tt = ss + epsT;
  uint32_t r = isqrt48(tt);
  uint64_t inv = std::min<uint64_t>(65535, udiv(C, std::max<uint32_t>(r, 1)));
  for (size_t k = 0; k < x.size(); ++k) {
    int64_t xg = int64_t(x[k]) * int64_t(g[k]);
    y[k] = int16_t(sat16(rshr(xg * int64_t(inv), shPost)));
  }
}

// ---- RoPE (rotate_half) -----------------------------------------------------------------------
// x: H*D elements of one row; cosv/sinv: D/2 each (Q1.14).
inline void rope(std::span<const int16_t> x, std::span<int16_t> y, uint32_t H, uint32_t D,
                 std::span<const int16_t> cosv, std::span<const int16_t> sinv) {
  for (uint32_t h = 0; h < H; ++h)
    for (uint32_t i = 0; i < D / 2; ++i) {
      int64_t x1 = x[h * D + i], x2 = x[h * D + i + D / 2];
      int64_t c = cosv[i], s = sinv[i];
      y[h * D + i] = int16_t(sat16(rshr(x1 * c - x2 * s, 14)));
      y[h * D + i + D / 2] = int16_t(sat16(rshr(x2 * c + x1 * s, 14)));
    }
}

// ---- Attention (one query row, one head) ---------------------------------------------------
// q: D int8; keyAt(t)/valAt(t) return pointers to D int8 each; T keys.
template <class KeyFn, class ValFn>
inline void attention_head(std::span<const int8_t> q, uint32_t D, uint32_t T, KeyFn keyAt, ValFn valAt,
                           uint32_t Ms, uint32_t Ss, uint32_t Mo, uint32_t So, std::span<int8_t> out,
                           std::vector<int32_t>& scores, std::vector<uint16_t>& probs) {
  scores.resize(T);
  probs.resize(T);
  int32_t mx = INT32_MIN;
  for (uint32_t t = 0; t < T; ++t) {
    const int8_t* k = keyAt(t);
    int32_t s = 0;
    for (uint32_t d = 0; d < D; ++d) s += int32_t(q[d]) * int32_t(k[d]);
    scores[t] = s;
    mx = std::max(mx, s);
  }
  uint32_t sum = 0;
  for (uint32_t t = 0; t < T; ++t) {
    uint64_t z = uint64_t(mulshift(int64_t(mx) - int64_t(scores[t]), Ms, Ss));
    uint16_t p = 0;
    if (z < 4096) p = uint16_t((uint32_t(kExpIntLut[z >> 8]) * uint32_t(kExpFracLut[z & 255]) + (1u << 15)) >> 16);
    probs[t] = p;
    sum += p;
  }
  uint32_t inv = udiv(uint64_t(1) << 31, sum);
  std::vector<int32_t> o(D, 0);
  for (uint32_t t = 0; t < T; ++t) {
    uint32_t pn = uint32_t(satu8((int64_t(probs[t]) * int64_t(inv) + (int64_t(1) << 22)) >> 23));
    const int8_t* v = valAt(t);
    for (uint32_t d = 0; d < D; ++d) o[d] += int32_t(pn) * int32_t(v[d]);
  }
  for (uint32_t d = 0; d < D; ++d) out[d] = int8_t(sat8(mulshift(o[d], Mo, So)));
}

}  // namespace llaccel::num
