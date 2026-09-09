// llaccel ISA definitions — single source of truth shared by the compiler,
// the runtime / functional simulator and the Verilator testbenches.
// Mirrors rtl/llaccel_pkg.sv and docs/ISA.md.
#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace llaccel {

// ---- device parameters ----------------------------------------------------
inline constexpr uint32_t kSramBytes = 1u << 20;
inline constexpr uint32_t kNumBanks = 16;
inline constexpr uint32_t kBankBytes = 16;
inline constexpr uint32_t kLineBytes = kNumBanks * kBankBytes;  // 256
inline constexpr uint32_t kGemmTN = 16;
inline constexpr uint32_t kGemmTK = 16;
inline constexpr uint32_t kGemmTM = 16;
inline constexpr uint32_t kVecLanes = 16;
inline constexpr uint32_t kAttnLanes = 64;
inline constexpr uint32_t kAttnTMax = 256;
inline constexpr uint32_t kNumSem = 32;
inline constexpr uint32_t kQueueDepth = 8;
inline constexpr uint32_t kDramBeat = 64;
inline constexpr uint32_t kInstrBytes = 64;
inline constexpr uint32_t kNumPerf = 24;
inline constexpr uint8_t kNoSem = 0xFF;

// ---- opcodes ----------------------------------------------------------------
enum class Op : uint8_t {
  NOP = 0x00,
  HALT = 0x01,
  DMA_LOAD = 0x10,
  DMA_STORE = 0x11,
  GEMM = 0x20,
  VEC_RMSNORM = 0x30,
  VEC_ROPE = 0x31,
  VEC_SILU = 0x32,
  VEC_MUL = 0x33,
  VEC_ADD = 0x34,
  VEC_QUANT = 0x35,
  ATTN = 0x40,
  KV_WRITE = 0x41,
};

enum class Engine : uint8_t { CP = 0, DMA = 1, GEMM = 2, VEC = 3, ATTN = 4 };

inline constexpr Engine engineOf(Op op) {
  auto v = static_cast<uint8_t>(op);
  if (v < 0x10) return Engine::CP;
  if (v < 0x20) return Engine::DMA;
  if (v < 0x30) return Engine::GEMM;
  if (v < 0x40) return Engine::VEC;
  return Engine::ATTN;
}

inline constexpr std::string_view opName(Op op) {
  switch (op) {
    case Op::NOP: return "NOP";
    case Op::HALT: return "HALT";
    case Op::DMA_LOAD: return "DMA_LOAD";
    case Op::DMA_STORE: return "DMA_STORE";
    case Op::GEMM: return "GEMM";
    case Op::VEC_RMSNORM: return "VEC_RMSNORM";
    case Op::VEC_ROPE: return "VEC_ROPE";
    case Op::VEC_SILU: return "VEC_SILU";
    case Op::VEC_MUL: return "VEC_MUL";
    case Op::VEC_ADD: return "VEC_ADD";
    case Op::VEC_QUANT: return "VEC_QUANT";
    case Op::ATTN: return "ATTN";
    case Op::KV_WRITE: return "KV_WRITE";
  }
  return "?";
}

// GEMM epilogue modes (ep word: mode[3:0] | out_i8[4] | aux_shift[15:8])
enum class Epilogue : uint8_t { NONE = 0, RESADD = 1, SILU = 2, MUL = 3 };
inline constexpr uint32_t kFlagHasBias = 1u << 0;

// ---- instruction word ---------------------------------------------------------
// word0 = opcode | flags<<8 | wait_sem<<16 | signal_sem<<24 ; word1 = wait_val ;
// words 2..15 = operands in docs/ISA.md order.
struct Instr {
  std::array<uint32_t, 16> w{};

  Instr() { w[0] = static_cast<uint32_t>(Op::NOP) | (uint32_t(kNoSem) << 16) | (uint32_t(kNoSem) << 24); }
  explicit Instr(Op op, uint8_t flags = 0) : Instr() {
    w[0] = uint32_t(op) | (uint32_t(flags) << 8) | (uint32_t(kNoSem) << 16) | (uint32_t(kNoSem) << 24);
  }

  Op op() const { return static_cast<Op>(w[0] & 0xFF); }
  uint8_t flags() const { return (w[0] >> 8) & 0xFF; }
  uint8_t waitSem() const { return (w[0] >> 16) & 0xFF; }
  uint8_t signalSem() const { return (w[0] >> 24) & 0xFF; }
  uint32_t waitVal() const { return w[1]; }
  uint32_t operator[](size_t i) const { return w[i]; }
  uint32_t& operator[](size_t i) { return w[i]; }

  void setWait(uint8_t sem, uint32_t val) { w[0] = (w[0] & 0xFF00FFFFu) | (uint32_t(sem) << 16); w[1] = val; }
  void setSignal(uint8_t sem) { w[0] = (w[0] & 0x00FFFFFFu) | (uint32_t(sem) << 24); }
  void setFlags(uint8_t f) { w[0] = (w[0] & 0xFFFF00FFu) | (uint32_t(f) << 8); }

  // Operand accessors (word indices per ISA.md). Names match the docs.
  // DMA_LOAD / DMA_STORE
  uint32_t& dmaSram() { return w[2]; }  uint32_t& dmaDram() { return w[3]; }
  uint32_t& dmaRows() { return w[4]; }  uint32_t& dmaRowBytes() { return w[5]; }
  uint32_t& dmaSrcStride() { return w[6]; }  uint32_t& dmaDstStride() { return w[7]; }
  // GEMM
  uint32_t& gA() { return w[2]; }  uint32_t& gW() { return w[3]; }  uint32_t& gOut() { return w[4]; }
  uint32_t& gRq() { return w[5]; } uint32_t& gBias() { return w[6]; } uint32_t& gAux() { return w[7]; }
  uint32_t& gM() { return w[8]; }  uint32_t& gN() { return w[9]; }   uint32_t& gK() { return w[10]; }
  uint32_t& gEp() { return w[11]; } uint32_t& gSiluMi() { return w[12]; } uint32_t& gSiluSiSh() { return w[13]; }
  // VEC
  uint32_t& vSrc() { return w[2]; } uint32_t& vSrc1() { return w[3]; } uint32_t& vDst() { return w[4]; }
  // ATTN
  uint32_t& aQ() { return w[2]; } uint32_t& aOut() { return w[3]; } uint32_t& aK() { return w[4]; } uint32_t& aV() { return w[5]; }

  void toBytes(uint8_t* dst) const { std::memcpy(dst, w.data(), kInstrBytes); }
  static Instr fromBytes(const uint8_t* src) { Instr i; std::memcpy(i.w.data(), src, kInstrBytes); return i; }
};
static_assert(sizeof(Instr) == kInstrBytes);

inline constexpr uint32_t packEp(Epilogue mode, bool outI8, uint8_t auxShift) {
  return uint32_t(mode) | (uint32_t(outI8) << 4) | (uint32_t(auxShift) << 8);
}
inline constexpr Epilogue epMode(uint32_t ep) { return static_cast<Epilogue>(ep & 0xF); }
inline constexpr bool epOutI8(uint32_t ep) { return (ep >> 4) & 1; }
inline constexpr uint8_t epAuxShift(uint32_t ep) { return (ep >> 8) & 0xFF; }

// ---- perf counter indices ------------------------------------------------------
enum Perf : uint32_t {
  PERF_CYCLES = 0, PERF_INSTR_ISSUED, PERF_CP_STALL_WAIT, PERF_CP_STALL_QFULL, PERF_CP_STALL_FETCH,
  PERF_GEMM_BUSY, PERF_GEMM_MAC_CYCLES, PERF_GEMM_SRAM_STALL, PERF_GEMM_EPILOGUE_CYCLES,
  PERF_VEC_BUSY, PERF_VEC_SRAM_STALL, PERF_ATTN_BUSY, PERF_ATTN_SRAM_STALL, PERF_ATTN_MAC_CYCLES,
  PERF_DMA_BUSY, PERF_DMA_SRAM_STALL, PERF_DMA_DRAM_WAIT,
  PERF_SRAM_RD_BYTES, PERF_SRAM_WR_BYTES, PERF_DRAM_RD_BYTES, PERF_DRAM_WR_BYTES,
  PERF_GEMM_IDLE_QEMPTY, PERF_VEC_IDLE_QEMPTY, PERF_ATTN_IDLE_QEMPTY,
};
inline constexpr std::array<std::string_view, kNumPerf> kPerfNames = {
    "cycles", "instr_issued", "cp_stall_wait", "cp_stall_qfull", "cp_stall_fetch",
    "gemm_busy", "gemm_mac_cycles", "gemm_sram_stall", "gemm_epilogue_cycles",
    "vec_busy", "vec_sram_stall", "attn_busy", "attn_sram_stall", "attn_mac_cycles",
    "dma_busy", "dma_sram_stall", "dma_dram_wait",
    "sram_rd_bytes", "sram_wr_bytes", "dram_rd_bytes", "dram_wr_bytes",
    "gemm_idle_qempty", "vec_idle_qempty", "attn_idle_qempty"};

// ---- .llbin container --------------------------------------------------------------
inline constexpr uint32_t kLlbinMagic = 0x4E424C4Cu;  // "LLBN"
inline constexpr uint32_t kLlbinVersion = 1;
enum class Section : uint32_t { DRAM_IMAGE = 1, PROGRAM = 2, META_JSON = 3 };
struct SectionHeader {
  uint32_t kind;
  uint32_t flags;  // PROGRAM: M rows the program was compiled for
  uint64_t offset;
  uint64_t size;
};
static_assert(sizeof(SectionHeader) == 24);

}  // namespace llaccel
