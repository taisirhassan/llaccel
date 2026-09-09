// Functional (ISA-level) simulator: executes llaccel programs with the exact
// integer semantics of docs/NUMERICS.md (via numerics.h) and the memory-access
// pattern of docs/ARCH.md so that its byte counters can be compared against the
// RTL's. No cycle timing. Two policies: sequential (program order) and random
// legal interleaving of the engine queues honouring only the semaphores — the
// latter exposes missing dependencies in compiler-generated programs.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <random>
#include <stdexcept>
#include <string>

#include "llaccel/device.h"
#include "llaccel/numerics.h"

namespace llaccel {
namespace {

using namespace num;

class FuncSim final : public Device {
 public:
  explicit FuncSim(const DeviceOptions& opt) : opt_(opt), sram_(kSramBytes, 0) {
    dram_.resize(std::max<uint64_t>(opt.dramBytes, 1u << 20), 0);
  }
  std::string name() const override { return opt_.interleaveSeed ? "func-sim (interleaved)" : "func-sim"; }
  void dramWrite(uint64_t addr, const void* src, uint64_t n) override {
    ensureDram(addr + n);
    std::memcpy(&dram_[addr], src, n);
  }
  void dramRead(uint64_t addr, void* dst, uint64_t n) const override {
    if (addr + n > dram_.size()) throw std::runtime_error("dramRead out of range");
    std::memcpy(dst, &dram_[addr], n);
  }
  uint64_t dramSize() const override { return dram_.size(); }
  std::vector<uint8_t> sramSnapshot() const override { return sram_; }

  PerfCounters run(uint32_t pc, uint32_t pos) override {
    perf_.fill(0);
    sem_.fill(0);
    pos_ = pos;
    std::array<std::deque<Instr>, 5> queues;  // indexed by Engine
    std::mt19937 rng(opt_.interleaveSeed);
    bool halted = false;
    uint32_t fetchPc = pc;
    uint64_t issued = 0;
    const uint64_t maxInstr = 50'000'000;
    while (true) {
      // Candidate actions.
      std::vector<int> cands;  // 0 = issue, 1..4 = execute head of engine queue
      for (int e = 1; e <= 4; ++e)
        if (!queues[e].empty()) cands.push_back(e);
      bool canIssue = false;
      Instr next;
      if (!halted) {
        if (fetchPc + kInstrBytes > dram_.size()) throw std::runtime_error("instruction fetch out of DRAM range");
        next = Instr::fromBytes(&dram_[fetchPc]);
        perf_[PERF_DRAM_RD_BYTES] += 0;  // instruction fetch bytes are accounted at issue below
        bool waitOk = next.waitSem() == kNoSem || sem_[next.waitSem()] >= next.waitVal();
        int eng = int(engineOf(next.op()));
        bool qOk = eng == 0 || queues[eng].size() < kQueueDepth;
        canIssue = waitOk && qOk;
        if (canIssue) cands.push_back(0);
      }
      if (cands.empty()) {
        if (halted) break;  // all queues drained after HALT
        throw std::runtime_error("func-sim deadlock at pc=0x" + hex(fetchPc) + " (" + std::string(opName(next.op())) +
                                 " waits sem " + std::to_string(next.waitSem()) + " >= " + std::to_string(next.waitVal()) +
                                 ", current " + std::to_string(next.waitSem() == kNoSem ? 0 : sem_[next.waitSem()]) + ")");
      }
      int choice;
      if (opt_.interleaveSeed == 0) {
        // Sequential policy: drain engines before issuing (deterministic program order).
        choice = cands[0] != 0 ? cands[0] : 0;
      } else {
        choice = cands[std::uniform_int_distribution<size_t>(0, cands.size() - 1)(rng)];
      }
      if (choice == 0) {
        fetchPc += kInstrBytes;
        perf_[PERF_INSTR_ISSUED]++;
        perf_[PERF_DRAM_RD_BYTES] += kInstrBytes;
        if (++issued > maxInstr) throw std::runtime_error("func-sim: instruction limit exceeded (runaway program?)");
        Op op = next.op();
        if (op == Op::HALT) {
          halted = true;
        } else if (op == Op::NOP) {
          signal(next);
        } else {
          queues[int(engineOf(op))].push_back(next);
        }
      } else {
        Instr in = queues[choice].front();
        queues[choice].pop_front();
        execute(in);
        signal(in);
      }
    }
    return perf_;
  }

 private:
  static std::string hex(uint64_t v) { char b[32]; std::snprintf(b, sizeof b, "%llx", (unsigned long long)v); return b; }
  void ensureDram(uint64_t n) { if (n > dram_.size()) dram_.resize(n, 0); }
  void signal(const Instr& in) { if (in.signalSem() != kNoSem) sem_[in.signalSem()]++; }

  // ---- SRAM accessors --------------------------------------------------------------
  void chk(uint64_t addr, uint64_t n, const char* what) const {
    if (addr + n > sram_.size()) throw std::runtime_error(std::string("SRAM access out of range in ") + what + " @0x" + hex(addr));
  }
  int8_t rd8(uint32_t a) const { return int8_t(sram_[a]); }
  int16_t rd16(uint32_t a) const { int16_t v; std::memcpy(&v, &sram_[a], 2); return v; }
  int32_t rd32(uint32_t a) const { int32_t v; std::memcpy(&v, &sram_[a], 4); return v; }
  void wr8(uint32_t a, int8_t v) { sram_[a] = uint8_t(v); }
  void wr16(uint32_t a, int16_t v) { std::memcpy(&sram_[a], &v, 2); }

  void execute(const Instr& in) {
    if (opt_.trace) trace(in);
    switch (in.op()) {
      case Op::DMA_LOAD: return dma(in, true);
      case Op::DMA_STORE: return dma(in, false);
      case Op::GEMM: return gemm(in);
      case Op::VEC_RMSNORM: return rmsnormOp(in);
      case Op::VEC_ROPE: return ropeOp(in);
      case Op::VEC_SILU: return elementwise(in, 0);
      case Op::VEC_MUL: return elementwise(in, 1);
      case Op::VEC_ADD: return elementwise(in, 2);
      case Op::VEC_QUANT: return elementwise(in, 3);
      case Op::ATTN: return attn(in);
      case Op::KV_WRITE: return kvWrite(in);
      default: throw std::runtime_error("func-sim: unexpected opcode " + std::to_string(int(in.op())));
    }
  }

  void trace(const Instr& in) {
    std::printf("  %-12s", std::string(opName(in.op())).c_str());
    for (int i = 2; i < 16; ++i) std::printf(" %08x", in[i]);
    if (in.waitSem() != kNoSem) std::printf("  wait s%u>=%u", in.waitSem(), in.waitVal());
    if (in.signalSem() != kNoSem) std::printf("  sig s%u", in.signalSem());
    std::printf("\n");
  }

  void dma(const Instr& in, bool load) {
    uint32_t sramA = in[2], dramA = in[3], rows = in[4], rowBytes = in[5], srcStride = in[6], dstStride = in[7];
    if (rowBytes % 16 || sramA % 16) throw std::runtime_error("DMA row/addr not 16-B aligned");
    for (uint32_t r = 0; r < rows; ++r) {
      if (load) {
        uint64_t s = uint64_t(dramA) + uint64_t(r) * srcStride, d = uint64_t(sramA) + uint64_t(r) * dstStride;
        if (s + rowBytes > dram_.size()) throw std::runtime_error("DMA_LOAD source out of DRAM range");
        chk(d, rowBytes, "DMA_LOAD");
        std::memcpy(&sram_[d], &dram_[s], rowBytes);
      } else {
        uint64_t s = uint64_t(sramA) + uint64_t(r) * srcStride, d = uint64_t(dramA) + uint64_t(r) * dstStride;
        chk(s, rowBytes, "DMA_STORE");
        ensureDram(d + rowBytes);
        std::memcpy(&dram_[d], &sram_[s], rowBytes);
      }
    }
    uint64_t bytes = uint64_t(rows) * rowBytes;
    perf_[load ? PERF_SRAM_WR_BYTES : PERF_SRAM_RD_BYTES] += bytes;
    perf_[load ? PERF_DRAM_RD_BYTES : PERF_DRAM_WR_BYTES] += bytes;
    perf_[PERF_DMA_BUSY] += (bytes + 63) / 64;  // ideal beats, informational
  }

  void gemm(const Instr& in) {
    uint32_t a = in[2], w = in[3], out = in[4], rq = in[5], bias = in[6], aux = in[7];
    uint32_t M = in[8], N = in[9], K = in[10], ep = in[11], siluMi = in[12], siluSi = in[13] & 0xFF, siluSh = (in[13] >> 8) & 0xFF;
    bool hasBias = in.flags() & kFlagHasBias;
    Epilogue mode = epMode(ep);
    bool outI8 = epOutI8(ep);
    uint8_t auxShift = epAuxShift(ep);
    if (M == 0 || M > kGemmTM || N % 16 || K % 16 || N == 0 || K == 0) throw std::runtime_error("GEMM bad shape");
    if (w % 256) throw std::runtime_error("GEMM weight address not 256-B aligned");
    if (outI8 && mode != Epilogue::NONE) throw std::runtime_error("GEMM i8 output with epilogue");
    if (mode != Epilogue::NONE && !opt_.epilogueFusion) throw std::runtime_error("GEMM epilogue used on a target without EPILOGUE_FUSION");
    uint32_t NT = N / 16, KT = K / 16;
    uint32_t outRow = outI8 ? N : 2 * N;
    chk(a, uint64_t(M) * K, "GEMM A"); chk(w, uint64_t(NT) * KT * 256, "GEMM W"); chk(rq, uint64_t(N) * 8, "GEMM rq");
    chk(out, uint64_t(M) * outRow, "GEMM out");
    if (hasBias) chk(bias, uint64_t(N) * 4, "GEMM bias");
    if (mode == Epilogue::RESADD || mode == Epilogue::MUL) chk(aux, uint64_t(M) * 2 * N, "GEMM aux");
    std::vector<int32_t> acc(M * 16);
    for (uint32_t nt = 0; nt < NT; ++nt) {
      std::fill(acc.begin(), acc.end(), 0);
      for (uint32_t kt = 0; kt < KT; ++kt) {
        const uint8_t* tile = &sram_[w + (nt * KT + kt) * 256];
        for (uint32_t m = 0; m < M; ++m) {
          const uint8_t* arow = &sram_[a + m * K + kt * 16];
          for (uint32_t n = 0; n < 16; ++n) {
            int32_t s = 0;
            for (uint32_t k = 0; k < 16; ++k) s += int32_t(int8_t(arow[k])) * int32_t(int8_t(tile[n * 16 + k]));
            acc[m * 16 + n] += s;
          }
        }
        perf_[PERF_SRAM_RD_BYTES] += 256 + uint64_t(M) * 16;
        perf_[PERF_GEMM_MAC_CYCLES] += M;
      }
      perf_[PERF_SRAM_RD_BYTES] += 128 + (hasBias ? 64 : 0);
      for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < 16; ++n) {
          uint32_t ch = nt * 16 + n;
          int64_t v = acc[m * 16 + n];
          if (hasBias) v += rd32(bias + ch * 4);
          RqEntry e{rd32(rq + ch * 8), rd32(rq + ch * 8 + 4)};
          if (outI8) {
            wr8(out + m * outRow + ch, int8_t(sat8(mulshift(v, uint32_t(e.M), uint32_t(e.S)))));
            continue;
          }
          int64_t t = sat16(mulshift(v, uint32_t(e.M), uint32_t(e.S)));
          int64_t y;
          switch (mode) {
            case Epilogue::NONE: y = t; break;
            case Epilogue::RESADD: y = sat16(t + rd16(aux + m * 2 * N + ch * 2)); break;
            case Epilogue::SILU: y = silu16(t, siluMi, siluSi, siluSh); break;
            case Epilogue::MUL: y = sat16(rshr(t * int64_t(rd16(aux + m * 2 * N + ch * 2)), auxShift)); break;
            default: throw std::runtime_error("GEMM bad epilogue mode");
          }
          wr16(out + m * outRow + ch * 2, int16_t(y));
        }
        if (mode == Epilogue::RESADD || mode == Epilogue::MUL) perf_[PERF_SRAM_RD_BYTES] += 32;
        perf_[PERF_SRAM_WR_BYTES] += outI8 ? 16 : 32;
        perf_[PERF_GEMM_EPILOGUE_CYCLES]++;
      }
    }
    perf_[PERF_GEMM_BUSY] += uint64_t(NT) * KT * (1 + M) + uint64_t(NT) * M;  // ideal, no stalls
  }

  void rmsnormOp(const Instr& in) {
    uint32_t src = in[2], gamma = in[3], dst = in[4], M = in[5], K = in[6], epsT = in[7], C = in[8], shPost = in[9];
    if (K % 16 || K == 0) throw std::runtime_error("RMSNORM K not a multiple of 16");
    chk(src, uint64_t(M) * K * 2, "RMSNORM src"); chk(gamma, uint64_t(K) * 2, "RMSNORM gamma"); chk(dst, uint64_t(M) * K * 2, "RMSNORM dst");
    std::vector<int16_t> x(K), g(K), y(K);
    for (uint32_t k = 0; k < K; ++k) g[k] = rd16(gamma + k * 2);
    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t k = 0; k < K; ++k) x[k] = rd16(src + (m * K + k) * 2);
      rmsnorm(x, g, y, epsT, C, shPost);
      for (uint32_t k = 0; k < K; ++k) wr16(dst + (m * K + k) * 2, y[k]);
    }
    perf_[PERF_SRAM_RD_BYTES] += uint64_t(M) * (2 * K * 2 + K * 2);  // two passes over x + gamma per row
    perf_[PERF_SRAM_WR_BYTES] += uint64_t(M) * K * 2;
    perf_[PERF_VEC_BUSY] += uint64_t(M) * (2 * K / 16 + 24 + 32);
  }

  void ropeOp(const Instr& in) {
    uint32_t src = in[2], dst = in[3], M = in[4], H = in[5], D = in[6], table = in[7], stride = in[8];
    if (D != 16 && D != 32 && D != 64) throw std::runtime_error("ROPE bad D");
    uint32_t row = H * D;
    chk(src, uint64_t(M) * row * 2, "ROPE src"); chk(dst, uint64_t(M) * row * 2, "ROPE dst");
    std::vector<int16_t> x(row), y(row), c(D / 2), s(D / 2);
    for (uint32_t m = 0; m < M; ++m) {
      uint32_t p = pos_ + m;
      chk(uint64_t(table) + uint64_t(p) * stride, 2 * D, "ROPE table");
      for (uint32_t i = 0; i < D / 2; ++i) {
        c[i] = rd16(table + p * stride + i * 2);
        s[i] = rd16(table + p * stride + (D / 2 + i) * 2);
      }
      for (uint32_t i = 0; i < row; ++i) x[i] = rd16(src + (m * row + i) * 2);
      rope(x, y, H, D, c, s);
      for (uint32_t i = 0; i < row; ++i) wr16(dst + (m * row + i) * 2, y[i]);
    }
    perf_[PERF_SRAM_RD_BYTES] += uint64_t(M) * (row * 2 + H * 2 * D);
    perf_[PERF_SRAM_WR_BYTES] += uint64_t(M) * row * 2;
    perf_[PERF_VEC_BUSY] += uint64_t(M) * H * (D / 2 / 16 + 1);
  }

  // kind: 0 SILU(src,dst,count,Mi,Si,sh_out) 1 MUL(a,b,dst,count,sh) 2 ADD(a,b,dst,count,sh_b) 3 QUANT(src,dst,count,M,S)
  void elementwise(const Instr& in, int kind) {
    if (kind == 0 || kind == 3) {
      uint32_t src = in[2], dst = in[3], count = in[4], p0 = in[5], p1 = in[6], p2 = in[7];
      if (count % 16) throw std::runtime_error("VEC count not a multiple of 16");
      chk(src, uint64_t(count) * 2, "VEC src"); chk(dst, uint64_t(count) * (kind == 3 ? 1 : 2), "VEC dst");
      for (uint32_t i = 0; i < count; ++i) {
        int64_t x = rd16(src + i * 2);
        if (kind == 0) wr16(dst + i * 2, int16_t(silu16(x, p0, p1, p2)));
        else wr8(dst + i, int8_t(vquant(x, p0, p1)));
      }
      perf_[PERF_SRAM_RD_BYTES] += uint64_t(count) * 2;
      perf_[PERF_SRAM_WR_BYTES] += uint64_t(count) * (kind == 3 ? 1 : 2);
      perf_[PERF_VEC_BUSY] += count / 16;
    } else {
      uint32_t a = in[2], b = in[3], dst = in[4], count = in[5], sh = in[6];
      if (count % 16) throw std::runtime_error("VEC count not a multiple of 16");
      chk(a, uint64_t(count) * 2, "VEC a"); chk(b, uint64_t(count) * 2, "VEC b"); chk(dst, uint64_t(count) * 2, "VEC dst");
      for (uint32_t i = 0; i < count; ++i) {
        int64_t x = rd16(a + i * 2), y = rd16(b + i * 2);
        wr16(dst + i * 2, int16_t(kind == 1 ? vmul(x, y, sh) : vadd(x, y, sh)));
      }
      perf_[PERF_SRAM_RD_BYTES] += uint64_t(count) * 4;
      perf_[PERF_SRAM_WR_BYTES] += uint64_t(count) * 2;
      perf_[PERF_VEC_BUSY] += count / 16;
    }
  }

  void attn(const Instr& in) {
    uint32_t q = in[2], out = in[3], kbase = in[4], vbase = in[5], M = in[6], H = in[7], Hkv = in[8], D = in[9],
             kvStride = in[10], Ms = in[11], Ss = in[12], Mo = in[13], So = in[14];
    if (D != 16 && D != 32 && D != 64) throw std::runtime_error("ATTN bad D");
    if (Hkv == 0 || H % Hkv) throw std::runtime_error("ATTN H not a multiple of Hkv");
    uint32_t row = H * D, grp = H / Hkv;
    chk(q, uint64_t(M) * row, "ATTN q"); chk(out, uint64_t(M) * row, "ATTN out");
    std::vector<int8_t> qv(D), o(D);
    std::vector<int32_t> scores;
    std::vector<uint16_t> probs;
    for (uint32_t m = 0; m < M; ++m) {
      uint32_t T = pos_ + m + 1;
      if (T > kAttnTMax) throw std::runtime_error("ATTN: T exceeds ATTN_TMAX");
      for (uint32_t h = 0; h < H; ++h) {
        uint32_t kvh = h / grp;
        uint64_t kb = uint64_t(kbase) + uint64_t(kvh) * kvStride, vb = uint64_t(vbase) + uint64_t(kvh) * kvStride;
        chk(kb, uint64_t(T) * D, "ATTN K"); chk(vb, uint64_t(T) * D, "ATTN V");
        for (uint32_t d = 0; d < D; ++d) qv[d] = rd8(q + m * row + h * D + d);
        attention_head(qv, D, T, [&](uint32_t t) { return reinterpret_cast<const int8_t*>(&sram_[kb + t * D]); },
                       [&](uint32_t t) { return reinterpret_cast<const int8_t*>(&sram_[vb + t * D]); }, Ms, Ss, Mo, So, o,
                       scores, probs);
        for (uint32_t d = 0; d < D; ++d) wr8(out + m * row + h * D + d, o[d]);
        perf_[PERF_SRAM_RD_BYTES] += D + 2ull * T * D;
        perf_[PERF_SRAM_WR_BYTES] += D;
        perf_[PERF_ATTN_MAC_CYCLES] += 2ull * T;
        perf_[PERF_ATTN_BUSY] += 3ull * T + 32;
      }
    }
  }

  void kvWrite(const Instr& in) {
    uint32_t src = in[2], base = in[3], M = in[4], Hkv = in[5], D = in[6], kvStride = in[7];
    uint32_t row = Hkv * D;
    chk(src, uint64_t(M) * row, "KV_WRITE src");
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t kvh = 0; kvh < Hkv; ++kvh) {
        uint64_t d = uint64_t(base) + uint64_t(kvh) * kvStride + uint64_t(pos_ + m) * D;
        chk(d, D, "KV_WRITE dst");
        std::memcpy(&sram_[d], &sram_[src + m * row + kvh * D], D);
      }
    perf_[PERF_SRAM_RD_BYTES] += uint64_t(M) * row;
    perf_[PERF_SRAM_WR_BYTES] += uint64_t(M) * row;
    perf_[PERF_ATTN_BUSY] += uint64_t(M) * Hkv;
  }

  DeviceOptions opt_;
  std::vector<uint8_t> dram_, sram_;
  std::array<uint32_t, kNumSem> sem_{};
  PerfCounters perf_{};
  uint32_t pos_ = 0;
};

}  // namespace

std::unique_ptr<Device> makeFuncSim(const DeviceOptions& opt) { return std::make_unique<FuncSim>(opt); }

}  // namespace llaccel
