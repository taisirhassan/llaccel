// Sanity checks of include/llaccel/numerics.h against floating-point references.
// These are tolerance checks (the bit-exact cross-checks are numpy vs C++ vs RTL
// elsewhere); they catch gross formula errors in the shared header.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "llaccel/numerics.h"

using namespace llaccel::num;

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++fails; } } while (0)

int main() {
  std::mt19937_64 rng(1);
  // rshr: round half up
  CHECK(rshr(5, 1) == 3, "rshr(5,1)");
  CHECK(rshr(-5, 1) == -2, "rshr(-5,1)=%lld", (long long)rshr(-5, 1));
  CHECK(rshr(7, 0) == 7, "rshr s=0");
  CHECK(sat16(40000) == 32767 && sat16(-40000) == -32768 && sat8(200) == 127, "saturation");
  // isqrt
  for (int i = 0; i < 100000; ++i) {
    uint64_t t = rng() & ((uint64_t(1) << 48) - 1);
    uint32_t r = isqrt48(t);
    CHECK(uint64_t(r) * r <= t && uint64_t(r + 1) * (r + 1) > t, "isqrt %llu -> %u", (unsigned long long)t, r);
  }
  CHECK(isqrt48(0) == 0 && isqrt48(1) == 1 && isqrt48((uint64_t(1) << 48) - 1) == (1u << 24) - 1, "isqrt edges");
  // rmsnorm vs float: K=128, x ~ N(0,1) at e=-8, gamma ~ 1 at e=-12
  {
    const uint32_t K = 128;
    const int ex = -8, eg = -12, ey = -8;
    std::normal_distribution<double> nd(0, 1);
    double maxErr = 0;
    for (int trial = 0; trial < 200; ++trial) {
      std::vector<int16_t> x(K), g(K), y(K);
      std::vector<double> xf(K), gf(K);
      for (uint32_t k = 0; k < K; ++k) {
        xf[k] = nd(rng); gf[k] = 1.0 + 0.3 * nd(rng);
        x[k] = int16_t(sat16(std::llround(xf[k] / std::ldexp(1.0, ex))));
        g[k] = int16_t(sat16(std::llround(gf[k] / std::ldexp(1.0, eg))));
      }
      double eps = 1e-5;
      uint32_t epsT = uint32_t(std::llround(eps * K * std::ldexp(1.0, -2 * ex)));
      int R = 20;
      uint32_t C = uint32_t(std::llround(std::ldexp(1.0, R) * std::sqrt(double(K))));
      uint32_t shPost = uint32_t(R - eg + ey);
      rmsnorm(x, g, y, epsT, C, shPost);
      double ss = 0;
      for (uint32_t k = 0; k < K; ++k) ss += xf[k] * xf[k];
      double rms = std::sqrt(ss / K + eps);
      for (uint32_t k = 0; k < K; ++k) {
        double ref = xf[k] * gf[k] / rms;
        double got = y[k] * std::ldexp(1.0, ey);
        maxErr = std::max(maxErr, std::fabs(ref - got));
      }
    }
    CHECK(maxErr < 0.05, "rmsnorm max abs error %.4f", maxErr);
    std::printf("rmsnorm max abs err vs float: %.4f\n", maxErr);
  }
  // silu vs float at e=-8
  {
    const int ex = -8, ey = -8;
    uint32_t Mi = 1u << 30, Si = uint32_t(30 - (ex + 12)), sh = uint32_t(16 + ey - ex);
    double maxErr = 0;
    for (int v = -32768; v < 32768; v += 7) {
      double xf = v * std::ldexp(1.0, ex);
      double ref = xf / (1.0 + std::exp(-xf));
      double got = silu16(v, Mi, Si, sh) * std::ldexp(1.0, ey);
      // The sigmoid LUT ends at sigma(8) = 0.99966, so the tail error is ~3.4e-4 relative.
      double err = std::fabs(ref - got) / (1.0 + std::fabs(xf));
      maxErr = std::max(maxErr, err);
    }
    CHECK(maxErr < 0.01, "silu max relative error %.4f", maxErr);
    std::printf("silu max (abs err)/(1+|x|) vs float: %.5f\n", maxErr);
  }
  // attention softmax vs float
  {
    const uint32_t D = 32, T = 40;
    double sq = 0.02, sk = 0.02, sv = 0.02, so = 0.02;  // output is a convex combination of V, so s_out = s_v
    std::vector<int8_t> q(D), out(D);
    std::vector<std::vector<int8_t>> K(T, std::vector<int8_t>(D)), V(T, std::vector<int8_t>(D));
    std::uniform_int_distribution<int> u8(-127, 127);
    for (auto& e : q) e = int8_t(u8(rng));
    for (auto& r : K) for (auto& e : r) e = int8_t(u8(rng));
    for (auto& r : V) for (auto& e : r) e = int8_t(u8(rng));
    double msf = 256.0 * sq * sk / std::sqrt(double(D));
    int Ss = 0; while (msf * std::ldexp(1.0, Ss) < std::ldexp(1.0, 30)) ++Ss;
    uint32_t Ms = uint32_t(std::llround(msf * std::ldexp(1.0, Ss)));
    double mof = sv / (256.0 * so);
    int So = 0; while (mof * std::ldexp(1.0, So) < std::ldexp(1.0, 30)) ++So;
    uint32_t Mo = uint32_t(std::llround(mof * std::ldexp(1.0, So)));
    std::vector<int32_t> sc; std::vector<uint16_t> pr;
    attention_head(q, D, T, [&](uint32_t t) { return K[t].data(); }, [&](uint32_t t) { return V[t].data(); }, Ms, Ss, Mo, So, out, sc, pr);
    // float reference
    std::vector<double> s(T);
    double mx = -1e30;
    for (uint32_t t = 0; t < T; ++t) { s[t] = 0; for (uint32_t d = 0; d < D; ++d) s[t] += q[d] * sq * K[t][d] * sk; s[t] /= std::sqrt(double(D)); mx = std::max(mx, s[t]); }
    double sum = 0; for (auto& e : s) { e = std::exp(e - mx); sum += e; }
    double maxErr = 0;
    for (uint32_t d = 0; d < D; ++d) {
      double ref = 0; for (uint32_t t = 0; t < T; ++t) ref += s[t] / sum * V[t][d] * sv;
      maxErr = std::max(maxErr, std::fabs(ref - out[d] * so));
    }
    CHECK(maxErr < 0.03, "attention max abs err %.4f", maxErr);
    std::printf("attention max abs err vs float: %.4f\n", maxErr);
  }
  std::printf(fails ? "numerics_selftest: %d FAILURES\n" : "numerics_selftest: all checks passed\n", fails);
  return fails ? 1 : 0;
}
