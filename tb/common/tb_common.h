// tb_common.h — shared helpers for the llaccel Verilator testbenches.
//
//  * DramDriver<Top>: drives llaccel::DramModel on a top's DRAM ports, one call
//    per clock cycle (sample the request before the rising edge, tick the model
//    after it, present the response and req_ready for the next cycle).
//  * SramImage: byte-level read/write of the 16 behavioral sram_bank memories
//    through their Verilator public signals (bank = addr[7:4], word = addr[23:8]).
//  * Reference models for the instructions the testbenches exercise, built
//    only from include/llaccel/numerics.h primitives.
//  * Instruction builders (llaccel::Instr) and the ISA.md weight tiling helper.
#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "llaccel/dram_model.h"
#include "llaccel/isa.h"
#include "llaccel/numerics.h"

namespace tb {

// ---------------------------------------------------------------------------------
// Random numbers
// ---------------------------------------------------------------------------------
struct Rng {
  std::mt19937_64 g;
  explicit Rng(uint64_t seed) : g(seed) {}
  uint64_t u64() { return g(); }
  // uniform in [lo, hi]
  int64_t range(int64_t lo, int64_t hi) { return lo + int64_t(g() % uint64_t(hi - lo + 1)); }
  bool coin(uint32_t pct = 50) { return (g() % 100) < pct; }
  void fill(uint8_t* p, size_t n) { for (size_t i = 0; i < n; ++i) p[i] = uint8_t(g()); }
  template <class T> T pick(std::span<const T> v) { return v[g() % v.size()]; }
  template <class T, size_t N> T pick(const T (&v)[N]) { return v[g() % N]; }
};

inline uint64_t seedFromArgs(int argc, char** argv, uint64_t dflt = 1) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("+seed=", 0) == 0) return std::strtoull(a.c_str() + 6, nullptr, 0);
  }
  return dflt;
}
inline uint64_t argValue(int argc, char** argv, const char* key, uint64_t dflt) {
  std::string k = std::string("+") + key + "=";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind(k, 0) == 0) return std::strtoull(a.c_str() + k.size(), nullptr, 0);
  }
  return dflt;
}

// ---------------------------------------------------------------------------------
// SRAM image access through the 16 public bank memories.
// Each bank is a VlUnpacked<VlWide<4>, DEPTH>: DEPTH contiguous 16-byte words,
// byte i of a word at offset i (Verilator stores VlWide little-endian per 32-bit
// word on this host, which matches "byte i at rdata[8*i +: 8]").
// ---------------------------------------------------------------------------------
class SramImage {
 public:
  static constexpr uint32_t kBytes = llaccel::kSramBytes;
  std::array<uint8_t*, llaccel::kNumBanks> bank{};

  uint8_t* ptr(uint32_t addr) const {
    uint32_t b = (addr >> 4) & 15, w = addr >> 8, o = addr & 15;
    return bank[b] + size_t(w) * 16 + o;
  }
  uint8_t rd8(uint32_t a) const { return *ptr(a); }
  void wr8(uint32_t a, uint8_t v) { *ptr(a) = v; }
  void write(uint32_t addr, const void* src, size_t n) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    for (size_t i = 0; i < n; ++i) wr8(addr + uint32_t(i), s[i]);
  }
  void read(uint32_t addr, void* dst, size_t n) const {
    uint8_t* d = static_cast<uint8_t*>(dst);
    for (size_t i = 0; i < n; ++i) d[i] = rd8(addr + uint32_t(i));
  }
  void fill(const std::vector<uint8_t>& img) { write(0, img.data(), img.size()); }
  std::vector<uint8_t> snapshot() const { std::vector<uint8_t> v(kBytes); read(0, v.data(), kBytes); return v; }
};

// Collect the 16 bank memory base pointers of a top whose banks are
// `g_bank[b].u_bank.mem` directly under the top module. PREFIX is the Verilator
// name of the top module (e.g. tb_gemm_top). Usage: TB_COLLECT_BANKS(top, tb_gemm_top, img).
#define TB_BANK_MEM(top, PREFIX, b) ((top).rootp->PREFIX##__DOT__g_bank__BRA__##b##__KET____DOT__u_bank__DOT__mem)
#define TB_COLLECT_BANKS(top, PREFIX, img)                                                                   \
  do {                                                                                                       \
    (img).bank[0] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 0)[0]);                             \
    (img).bank[1] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 1)[0]);                             \
    (img).bank[2] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 2)[0]);                             \
    (img).bank[3] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 3)[0]);                             \
    (img).bank[4] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 4)[0]);                             \
    (img).bank[5] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 5)[0]);                             \
    (img).bank[6] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 6)[0]);                             \
    (img).bank[7] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 7)[0]);                             \
    (img).bank[8] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 8)[0]);                             \
    (img).bank[9] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 9)[0]);                             \
    (img).bank[10] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 10)[0]);                           \
    (img).bank[11] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 11)[0]);                           \
    (img).bank[12] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 12)[0]);                           \
    (img).bank[13] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 13)[0]);                           \
    (img).bank[14] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 14)[0]);                           \
    (img).bank[15] = reinterpret_cast<uint8_t*>(&TB_BANK_MEM(top, PREFIX, 15)[0]);                           \
  } while (0)

// ---------------------------------------------------------------------------------
// DRAM driver. Requires the top to have the docs/ARCH.md DRAM ports:
//   dram_req_valid, dram_req_ready, dram_req_we, dram_req_addr, dram_req_wdata (512),
//   dram_req_wstrb (64), dram_rsp_valid, dram_rsp_rdata (512).
// Call order per cycle (clk low, inputs stable):
//   preEdge(top)  -> samples req_valid && req_ready and hands the request to the model
//   top.clk = 1; top.eval();
//   postEdge(top) -> ticks the model, drives rsp_valid / rsp_rdata / req_ready
//   top.clk = 0; top.eval();
// ---------------------------------------------------------------------------------
template <class Top>
class DramDriver {
 public:
  llaccel::DramModel& dram;
  uint64_t reqReads = 0, reqWrites = 0;         // accepted requests
  uint64_t fetchLo = 0, fetchHi = 0;            // [lo, hi): address window counted as instruction fetches
  uint64_t fetchBeats = 0;

  explicit DramDriver(llaccel::DramModel& d) : dram(d) {}

  void init(Top& t) {
    t.dram_req_ready = dram.reqReady();
    t.dram_rsp_valid = 0;
  }
  void preEdge(Top& t) {
    bool valid = t.dram_req_valid && t.dram_req_ready;
    if (valid) {
      uint32_t addr = t.dram_req_addr;
      bool we = t.dram_req_we;
      uint64_t strb = t.dram_req_wstrb;
      bool ok = dram.request(true, we, addr, reinterpret_cast<const uint8_t*>(t.dram_req_wdata.data()),
                             reinterpret_cast<const uint8_t*>(&strb));
      if (!ok) { std::fprintf(stderr, "DRAM model refused a request the DUT saw as accepted\n"); std::abort(); }
      if (we) ++reqWrites; else ++reqReads;
      if (!we && addr >= fetchLo && addr < fetchHi) ++fetchBeats;
    }
  }
  void postEdge(Top& t) {
    uint8_t rdata[64];
    bool rsp = dram.tick(rdata);
    t.dram_rsp_valid = rsp;
    if (rsp) std::memcpy(t.dram_rsp_rdata.data(), rdata, 64);
    t.dram_req_ready = dram.reqReady();
  }
};

// ---------------------------------------------------------------------------------
// ISA.md weight tiling: W is [N][K] int8 row-major; tile (nt, kt) at
// (nt * K/16 + kt) * 256, byte n*16 + k of the tile = W[nt*16 + n][kt*16 + k].
// ---------------------------------------------------------------------------------
inline std::vector<uint8_t> tileWeights(const int8_t* W, uint32_t N, uint32_t K) {
  uint32_t NT = N / 16, KT = K / 16;
  std::vector<uint8_t> out(size_t(NT) * KT * 256);
  for (uint32_t nt = 0; nt < NT; ++nt)
    for (uint32_t kt = 0; kt < KT; ++kt)
      for (uint32_t n = 0; n < 16; ++n)
        for (uint32_t k = 0; k < 16; ++k)
          out[(size_t(nt) * KT + kt) * 256 + n * 16 + k] = uint8_t(W[size_t(nt * 16 + n) * K + kt * 16 + k]);
  return out;
}

// ---------------------------------------------------------------------------------
// GEMM reference (docs/NUMERICS.md "GEMM"), from numerics.h primitives only.
// ---------------------------------------------------------------------------------
struct GemmParams {
  uint32_t M = 1, N = 16, K = 16;
  bool hasBias = false;
  bool outI8 = false;
  llaccel::Epilogue mode = llaccel::Epilogue::NONE;
  uint8_t auxShift = 0;
  uint32_t siluMi = 0, siluSi = 0, siluSh = 0;
};

// A: [M][K] i8, W: [N][K] i8, rq: N entries, bias: N i32 (nullptr if !hasBias),
// aux: [M][N] i16 (nullptr unless RESADD/MUL). Returns the output bytes
// ([M][N] i16 little-endian, or [M][N] i8).
inline std::vector<uint8_t> refGemm(const GemmParams& p, const int8_t* A, const int8_t* W,
                                    const llaccel::num::RqEntry* rq, const int32_t* bias, const int16_t* aux) {
  using namespace llaccel;
  using namespace llaccel::num;
  std::vector<uint8_t> out(size_t(p.M) * p.N * (p.outI8 ? 1 : 2));
  for (uint32_t m = 0; m < p.M; ++m)
    for (uint32_t n = 0; n < p.N; ++n) {
      int64_t acc = 0;
      for (uint32_t k = 0; k < p.K; ++k) acc += int64_t(A[size_t(m) * p.K + k]) * int64_t(W[size_t(n) * p.K + k]);
      if (p.hasBias) acc += bias[n];
      if (p.outI8) {
        out[size_t(m) * p.N + n] = uint8_t(int8_t(sat8(mulshift(acc, uint32_t(rq[n].M), uint32_t(rq[n].S)))));
        continue;
      }
      int64_t t = sat16(mulshift(acc, uint32_t(rq[n].M), uint32_t(rq[n].S)));
      int64_t y = t;
      switch (p.mode) {
        case Epilogue::NONE: y = t; break;
        case Epilogue::RESADD: y = sat16(t + aux[size_t(m) * p.N + n]); break;
        case Epilogue::SILU: y = silu16(t, p.siluMi, p.siluSi, p.siluSh); break;
        case Epilogue::MUL: y = sat16(rshr(t * int64_t(aux[size_t(m) * p.N + n]), p.auxShift)); break;
      }
      int16_t v = int16_t(y);
      std::memcpy(&out[(size_t(m) * p.N + n) * 2], &v, 2);
    }
  return out;
}

// ---------------------------------------------------------------------------------
// Instruction builders (word positions per docs/ISA.md).
// ---------------------------------------------------------------------------------
inline llaccel::Instr mkDmaLoad(uint32_t sramDst, uint32_t dramSrc, uint32_t rows, uint32_t rowBytes,
                                uint32_t srcStride, uint32_t dstStride) {
  llaccel::Instr i(llaccel::Op::DMA_LOAD);
  i.w[2] = sramDst; i.w[3] = dramSrc; i.w[4] = rows; i.w[5] = rowBytes; i.w[6] = srcStride; i.w[7] = dstStride;
  return i;
}
inline llaccel::Instr mkDmaStore(uint32_t sramSrc, uint32_t dramDst, uint32_t rows, uint32_t rowBytes,
                                 uint32_t srcStride, uint32_t dstStride) {
  llaccel::Instr i(llaccel::Op::DMA_STORE);
  i.w[2] = sramSrc; i.w[3] = dramDst; i.w[4] = rows; i.w[5] = rowBytes; i.w[6] = srcStride; i.w[7] = dstStride;
  return i;
}
inline llaccel::Instr mkGemm(const GemmParams& p, uint32_t a, uint32_t w, uint32_t out, uint32_t rq, uint32_t bias,
                             uint32_t aux) {
  llaccel::Instr i(llaccel::Op::GEMM, p.hasBias ? uint8_t(llaccel::kFlagHasBias) : 0);
  i.w[2] = a; i.w[3] = w; i.w[4] = out; i.w[5] = rq; i.w[6] = bias; i.w[7] = aux;
  i.w[8] = p.M; i.w[9] = p.N; i.w[10] = p.K;
  i.w[11] = llaccel::packEp(p.mode, p.outI8, p.auxShift);
  i.w[12] = p.siluMi;
  i.w[13] = (p.siluSi & 0xFF) | ((p.siluSh & 0xFF) << 8);
  return i;
}
inline llaccel::Instr mkVecRmsnorm(uint32_t src, uint32_t gamma, uint32_t dst, uint32_t M, uint32_t K, uint32_t epsT,
                                   uint32_t C, uint32_t shPost) {
  llaccel::Instr i(llaccel::Op::VEC_RMSNORM);
  i.w[2] = src; i.w[3] = gamma; i.w[4] = dst; i.w[5] = M; i.w[6] = K; i.w[7] = epsT; i.w[8] = C; i.w[9] = shPost;
  return i;
}
inline llaccel::Instr mkVecQuant(uint32_t src, uint32_t dst, uint32_t count, uint32_t M, uint32_t S) {
  llaccel::Instr i(llaccel::Op::VEC_QUANT);
  i.w[2] = src; i.w[3] = dst; i.w[4] = count; i.w[5] = M; i.w[6] = S;
  return i;
}
inline llaccel::Instr mkHalt() { return llaccel::Instr(llaccel::Op::HALT); }

// Number of 64-B DRAM beats a DMA row [addr, addr + bytes) touches.
inline uint64_t rowBeats(uint64_t addr, uint64_t bytes) {
  if (bytes == 0) return 0;
  return ((addr + bytes + 63) / 64) - (addr / 64);
}

// ---------------------------------------------------------------------------------
// Result reporting
// ---------------------------------------------------------------------------------
struct Report {
  int passed = 0, failed = 0;
  void tally(bool ok, const char* what) {
    if (ok) ++passed; else { ++failed; std::printf("  FAIL: %s\n", what); }
  }
  int finish(const char* name) const {
    std::printf("%s: %d passed, %d failed -> %s\n", name, passed, failed, failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
  }
};

// First differing byte between two buffers, or -1.
inline long firstDiff(const uint8_t* a, const uint8_t* b, size_t n) {
  for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) return long(i);
  return -1;
}

}  // namespace tb
