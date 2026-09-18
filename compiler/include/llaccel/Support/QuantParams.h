// Compiler-side parameter formulas from docs/NUMERICS.md ("Compiler:" lines).
// Header-only and MLIR-free so the numerics self-check test can include it.
// All arithmetic is exact double followed by llround; every function documents
// the NUMERICS.md line it implements. Errors are reported through
// std::expected (the compiler is built with -fno-exceptions, like MLIR).
//
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <vector>

namespace llaccel::qp {

template <typename T> using Result = std::expected<T, std::string>;

struct MulShift {
  int64_t M;  // in [2^30, 2^31)
  int64_t S;  // in [0, 63]
};

/// Normalise a positive real ratio r to M * 2^-S ~= r with M in [2^30, 2^31).
/// Fails if r <= 0 or S would be negative (r >= 2^31).
inline Result<MulShift> normalizeMulShift(double r) {
  if (!(r > 0.0) || !std::isfinite(r))
    return std::unexpected("normalizeMulShift: ratio must be positive and finite");
  int e = 0;
  double f = std::frexp(r, &e);  // r = f * 2^e, f in [0.5, 1)
  int64_t M = std::llround(std::ldexp(f, 31));
  int64_t S = 31 - e;
  if (M >= (int64_t(1) << 31)) {  // f rounded up to 1.0
    M = int64_t(1) << 30;
    S -= 1;
  }
  if (S < 0)
    return std::unexpected("normalizeMulShift: ratio too large (S < 0)");
  if (S > 63) {  // tiny ratio: keep S = 63 and accept a smaller M (may be 0)
    M = std::llround(std::ldexp(r, 63));
    S = 63;
  }
  return MulShift{M, S};
}

/// i16 exponent for a tensor with the given abs-max: e = max(ceil(log2(absmax/32767)), -15).
inline int64_t i16Exponent(double absmax) {
  if (!(absmax > 0.0))
    return -15;
  double e = std::ceil(std::log2(absmax) - std::log2(32767.0));
  return std::max<int64_t>(int64_t(e), -15);
}

/// i8 per-tensor scale: absmax / 127 (absmax 0 -> scale 1, tensor is all zero).
inline double i8Scale(double absmax) { return absmax > 0.0 ? absmax / 127.0 : 1.0; }

/// Quantize one value to int8 with the given scale (round-half-away, saturate).
inline int8_t quantI8(double v, double scale) {
  long long q = std::llround(std::clamp(v / scale, -128.0, 127.0));
  if (q > 127) q = 127;
  if (q < -128) q = -128;
  return int8_t(q);
}
/// Quantize one value to int16 at exponent e.
inline int16_t quantI16(double v, int64_t e) {
  long long q = std::llround(std::clamp(std::ldexp(v, int(-e)), -32768.0, 32767.0));
  if (q > 32767) q = 32767;
  if (q < -32768) q = -32768;
  return int16_t(q);
}

/// Per-output-channel weight scale: max|W[n,:]| / 127, zero rows -> 1.
inline double weightRowScale(const float *row, int64_t K) {
  double mx = 0.0;
  for (int64_t k = 0; k < K; ++k) mx = std::max(mx, double(std::fabs(row[k])));
  return mx > 0.0 ? mx / 127.0 : 1.0;
}

/// GEMM requant entry (NUMERICS.md GEMM "Compiler:" line).
///   i16 out: M*2^-S ~= s_a * s_w / 2^e_out ;  i8 out: s_a * s_w / s_out
inline Result<MulShift> gemmRq(double s_a, double s_w, bool outI8, int64_t e_out, double s_out) {
  double r = outI8 ? (s_a * s_w / s_out) : (s_a * s_w / std::ldexp(1.0, int(e_out)));
  return normalizeMulShift(r);
}

/// Bias in accumulator units: llround(b / (s_a * s_w[n])).
inline int32_t gemmBias(double b, double s_a, double s_w) {
  long long v = std::llround(std::clamp(b / (s_a * s_w), double(INT32_MIN), double(INT32_MAX)));
  if (v > INT32_MAX) v = INT32_MAX;
  if (v < INT32_MIN) v = INT32_MIN;
  return int32_t(v);
}

/// QUANT (i16 -> i8): M*2^-S ~= 2^e_x / s_y.
inline Result<MulShift> quantParams(int64_t e_x, double s_y) {
  return normalizeMulShift(std::ldexp(1.0, int(e_x)) / s_y);
}

struct RmsNormParams {
  int64_t eps_t;
  int64_t C;
  int64_t R;
  int64_t sh_post;
};

/// RMSNorm parameters.
///   eps_t   = llround(eps * K * 2^(-2 e_x))
///   R       = largest value <= 31 with C = llround(2^R sqrt(K)) < 2^32 and the
///             calibrated inv = C / isqrt(K * rms_q^2 + eps_t) < 65535/4.
///             rms_q uses the measured minimum row RMS when supplied, otherwise
///             the legacy absmax_x/4 estimate. The additional factor4 margin
///             protects rows quieter than those observed during calibration.
///   sh_post = R - e_g + e_y   (must be in [0, 63])
inline Result<RmsNormParams> rmsnormParams(int64_t K, double eps, int64_t e_x, int64_t e_g,
                                           int64_t e_y, double absmax_x,
                                           const std::string &name,
                                           std::optional<double> rms_min = std::nullopt) {
  if (rms_min && (!std::isfinite(*rms_min) || *rms_min < 0 || *rms_min > absmax_x))
    return std::unexpected("rmsnorm " + name + ": rms_min must be finite and in [0, absmax]");
  RmsNormParams p{};
  const double epsT = eps * double(K) * std::ldexp(1.0, int(-2 * e_x));
  if (!std::isfinite(epsT) || epsT < 0 || epsT >= double(UINT32_MAX) + 0.5)
    return std::unexpected("rmsnorm " + name + ": eps_t does not fit the ISA u32 operand");
  p.eps_t = std::llround(epsT);
  double rmsQ = rms_min.value_or(absmax_x / 4.0) / std::ldexp(1.0, int(e_x));
  double rTyp = std::floor(std::sqrt(double(K) * rmsQ * rmsQ + double(p.eps_t)));
  if (rTyp < 1.0) rTyp = 1.0;
  int64_t R = -1;
  for (int64_t cand = 31; cand >= 0; --cand) {
    double C = double(std::llround(std::ldexp(std::sqrt(double(K)), int(cand))));
    if (C >= 4294967296.0) continue;
    double invTyp = std::floor(C / rTyp);
    if (invTyp < 65535.0 / 4.0) {
      R = cand;
      p.C = int64_t(C);
      break;
    }
  }
  if (R < 0)
    return std::unexpected("rmsnorm " + name + ": no R in [0,31] keeps inv below 65535/4 "
                           "(input abs-max " + std::to_string(absmax_x) + " too small for e_x)");
  p.R = R;
  p.sh_post = R - e_g + e_y;
  if (p.sh_post < 0 || p.sh_post > 63)
    return std::unexpected("rmsnorm " + name + ": sh_post = " + std::to_string(p.sh_post) +
                           " outside [0, 63]");
  return p;
}

struct SiluParams {
  int64_t Mi;
  int64_t Si;
  int64_t sh_out;
};

/// SiLU: Mi = 2^30, Si = 30 - (e_x + 12) (so Mi*2^-Si = 2^(e_x+12)); sh_out = 16 + e_y - e_x.
inline Result<SiluParams> siluParams(int64_t e_x, int64_t e_y, const std::string &name) {
  SiluParams p{int64_t(1) << 30, 30 - (e_x + 12), 16 + e_y - e_x};
  if (p.Si < 0 || p.Si > 63)
    return std::unexpected("silu " + name + ": Si = " + std::to_string(p.Si) + " outside [0, 63]");
  if (p.sh_out < 0 || p.sh_out > 63)
    return std::unexpected("silu " + name + ": sh_out = " + std::to_string(p.sh_out) +
                           " outside [0, 63]");
  return p;
}

/// MUL: y = sat16(rshr(a*b, sh)) with a*b at exponent e_a+e_b  =>  sh = e_y - e_a - e_b in [0, 63].
inline Result<int64_t> mulShift(int64_t e_a, int64_t e_b, int64_t e_y, const std::string &name) {
  int64_t sh = e_y - e_a - e_b;
  if (sh < 0 || sh > 63)
    return std::unexpected("mul " + name + ": sh = " + std::to_string(sh) + " outside [0, 63]");
  return sh;
}

/// ADD: y = sat16(a + rshr(b, sh_b)), e_y = e_a  =>  sh_b = e_a - e_b >= 0.
inline Result<int64_t> addShift(int64_t e_a, int64_t e_b, const std::string &name) {
  int64_t sh = e_a - e_b;
  if (sh < 0 || sh > 63)
    return std::unexpected("add " + name + ": sh_b = " + std::to_string(sh) +
                           " outside [0, 63] (b must not have a larger exponent than a)");
  return sh;
}

struct AttnParams {
  MulShift s;  // Ms, Ss
  MulShift o;  // Mo, So
};

/// Attention: Ms*2^-Ss ~= 256 s_q s_k / sqrt(D) ; Mo*2^-So ~= s_v / (256 s_out).
inline Result<AttnParams> attnParams(double s_q, double s_k, double s_v, double s_out, int64_t D) {
  auto s = normalizeMulShift(256.0 * s_q * s_k / std::sqrt(double(D)));
  if (!s) return std::unexpected(s.error());
  auto o = normalizeMulShift(s_v / (256.0 * s_out));
  if (!o) return std::unexpected(o.error());
  return AttnParams{*s, *o};
}

/// RoPE tables: cos/sin[p][i] = llround(2^14 cos/sin(p * theta_i)), theta_i = base^(-2i/D).
inline void ropeTables(int64_t maxSeq, int64_t D, double base, std::vector<int16_t> &cosT,
                       std::vector<int16_t> &sinT) {
  int64_t half = D / 2;
  cosT.assign(size_t(maxSeq * half), 0);
  sinT.assign(size_t(maxSeq * half), 0);
  for (int64_t i = 0; i < half; ++i) {
    double theta = std::pow(base, -2.0 * double(i) / double(D));
    for (int64_t p = 0; p < maxSeq; ++p) {
      double ang = double(p) * theta;
      long long c = std::llround(16384.0 * std::cos(ang));
      long long s = std::llround(16384.0 * std::sin(ang));
      if (c > 32767) c = 32767;
      if (s > 32767) s = 32767;
      cosT[size_t(p * half + i)] = int16_t(c);
      sinT[size_t(p * half + i)] = int16_t(s);
    }
  }
}

}  // namespace llaccel::qp
