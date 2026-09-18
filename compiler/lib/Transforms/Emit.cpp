//===- Emit.cpp - .llbin container (docs/ISA.md, docs/DIALECT.md §3) ------===//
//
// DRAM image layout (all sections 64-B aligned, weights 256-B aligned):
//   [embedding table][input buffer (16 rows)][logits buffer (16 rows)]
//   [dram scratch (if any)][packed resident constants][tiled int8 weights ...]
//   [program M=prefill][program M=1]
// Row strides of the host-visible buffers are rounded up to 64 bytes so every
// DMA row start is a DRAM beat boundary. Weights use the ISA.md tiled layout:
// tile (nt, kt) at (nt*K/16 + kt)*256, byte n*16+k = W[nt*16+n][kt*16+k].
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"

#include "llvm/Support/raw_ostream.h"

#include "llaccel/isa.h"

namespace mlir::llaccel {
#define GEN_PASS_DEF_LLACCELEMIT
#include "llaccel/Transforms/Passes.h.inc"
} // namespace mlir::llaccel

using namespace mlir;
using namespace mlir::llaccel;
namespace isa = ::llaccel;

namespace {

struct DramLayout {
  llvm::StringMap<int64_t> addr;
  std::vector<uint8_t> image;
  int64_t place(StringRef name, int64_t bytes, int64_t align) {
    int64_t a = roundUp(int64_t(image.size()), align);
    image.resize(size_t(a + bytes), 0);
    addr[name] = a;
    return a;
  }
};

/// Tiled int8 weight (ISA.md) from the row-major [N][K] data.
std::vector<uint8_t> tileWeight(ArrayRef<int8_t> w, int64_t N, int64_t K) {
  std::vector<uint8_t> out(size_t(N * K));
  int64_t KT = K / 16;
  for (int64_t nt = 0; nt < N / 16; ++nt)
    for (int64_t kt = 0; kt < KT; ++kt) {
      uint8_t *tile = out.data() + (nt * KT + kt) * 256;
      for (int64_t n = 0; n < 16; ++n)
        for (int64_t k = 0; k < 16; ++k)
          tile[n * 16 + k] = uint8_t(w[size_t((nt * 16 + n) * K + kt * 16 + k)]);
    }
  return out;
}

struct Encoder {
  const DramLayout &dram;
  ProgramStats stats;
  std::string error;

  uint32_t sramAddr(Value buf, int64_t off) {
    auto alloc = buf.getDefiningOp<AllocOp>();
    return uint32_t(alloc.getAddr().value_or(0) + off);
  }
  uint32_t dramAddr(FlatSymbolRefAttr sym, int64_t off, Operation *op) {
    auto it = dram.addr.find(sym.getValue());
    if (it == dram.addr.end()) {
      error = ("DRAM symbol `" + sym.getValue() + "` has no address").str();
      return 0;
    }
    if (off < 0 || uint64_t(it->second) + uint64_t(off) > UINT32_MAX) {
      error = "DRAM symbol address exceeds 32-bit ISA operand";
      return 0;
    }
    return uint32_t(it->second + off);
  }

  void common(IsaOp op, isa::Instr &in) {
    if (auto w = op.getWaitSem())
      in.setWait(uint8_t(*w), uint32_t(op.getWaitValue()));
    if (auto s = op.getSignalSem())
      in.setSignal(uint8_t(*s));
    if (op.getWaitSem())
      stats.nWaits++;
  }

  std::optional<isa::Instr> encode(Operation *op) {
    isa::Instr in;
    llvm::TypeSwitch<Operation *>(op)
        .Case<DmaLoadOp>([&](DmaLoadOp o) {
          in = isa::Instr(isa::Op::DMA_LOAD);
          in.dmaSram() = sramAddr(o.getDst(), o.getDstOff());
          in.dmaDram() = dramAddr(o.getSrcAttr(), o.getSrcOff(), o);
          in.dmaRows() = uint32_t(o.getRows());
          in.dmaRowBytes() = uint32_t(o.getRowBytes());
          in.dmaSrcStride() = uint32_t(o.getSrcStride());
          in.dmaDstStride() = uint32_t(o.getDstStride());
          stats.dmaLoadBytes += o.getRows() * o.getRowBytes();
          stats.nDma++;
        })
        .Case<DmaStoreOp>([&](DmaStoreOp o) {
          in = isa::Instr(isa::Op::DMA_STORE);
          in.dmaSram() = sramAddr(o.getSrc(), o.getSrcOff());
          in.dmaDram() = dramAddr(o.getDstAttr(), o.getDstOff(), o);
          in.dmaRows() = uint32_t(o.getRows());
          in.dmaRowBytes() = uint32_t(o.getRowBytes());
          in.dmaSrcStride() = uint32_t(o.getSrcStride());
          in.dmaDstStride() = uint32_t(o.getDstStride());
          stats.dmaStoreBytes += o.getRows() * o.getRowBytes();
          stats.nDma++;
        })
        .Case<GemmOp>([&](GemmOp o) {
          in = isa::Instr(isa::Op::GEMM, o.getBias() ? isa::kFlagHasBias : 0);
          in.gA() = sramAddr(o.getA(), o.getAOff());
          in.gW() = sramAddr(o.getW(), o.getWOff());
          in.gOut() = sramAddr(o.getOut(), o.getOutOff());
          in.gRq() = sramAddr(o.getRq(), o.getRqOff());
          in.gBias() = o.getBias() ? sramAddr(o.getBias(), o.getBiasOff().value_or(0)) : 0;
          in.gAux() = o.getAux() ? sramAddr(o.getAux(), o.getAuxOff().value_or(0)) : 0;
          in.gM() = uint32_t(o.getM());
          in.gN() = uint32_t(o.getN());
          in.gK() = uint32_t(o.getK());
          in.gEp() = isa::packEp(isa::Epilogue(uint8_t(o.getEpilogue())), o.getOutI8(),
                                 uint8_t(o.getAuxShift()));
          in.gSiluMi() = uint32_t(o.getSiluMi());
          in.gSiluSiSh() = uint32_t(o.getSiluSi() & 0xFF) | (uint32_t(o.getSiluShOut() & 0xFF) << 8);
          stats.gemmMacs += o.getM() * o.getN() * o.getK();
          stats.nGemm++;
        })
        .Case<IsaRmsNormOp>([&](IsaRmsNormOp o) {
          in = isa::Instr(isa::Op::VEC_RMSNORM);
          in[2] = sramAddr(o.getSrc(), o.getSrcOff());
          in[3] = sramAddr(o.getGamma(), o.getGammaOff());
          in[4] = sramAddr(o.getDst(), o.getDstOff());
          in[5] = uint32_t(o.getM());
          in[6] = uint32_t(o.getK());
          in[7] = uint32_t(o.getEpsT());
          in[8] = uint32_t(o.getC());
          in[9] = uint32_t(o.getShPost());
          stats.nVec++;
        })
        .Case<IsaRopeOp>([&](IsaRopeOp o) {
          in = isa::Instr(isa::Op::VEC_ROPE);
          in[2] = sramAddr(o.getSrc(), o.getSrcOff());
          in[3] = sramAddr(o.getDst(), o.getDstOff());
          in[4] = uint32_t(o.getM());
          in[5] = uint32_t(o.getH());
          in[6] = uint32_t(o.getD());
          in[7] = sramAddr(o.getTable(), o.getTableOff());
          in[8] = uint32_t(o.getTableStride());
          stats.nVec++;
        })
        .Case<IsaSiluOp>([&](IsaSiluOp o) {
          in = isa::Instr(isa::Op::VEC_SILU);
          in[2] = sramAddr(o.getSrc(), o.getSrcOff());
          in[3] = sramAddr(o.getDst(), o.getDstOff());
          in[4] = uint32_t(o.getCount());
          in[5] = uint32_t(o.getMi());
          in[6] = uint32_t(o.getSi());
          in[7] = uint32_t(o.getShOut());
          stats.nVec++;
        })
        .Case<IsaMulOp>([&](IsaMulOp o) {
          in = isa::Instr(isa::Op::VEC_MUL);
          in[2] = sramAddr(o.getA(), o.getAOff());
          in[3] = sramAddr(o.getB(), o.getBOff());
          in[4] = sramAddr(o.getDst(), o.getDstOff());
          in[5] = uint32_t(o.getCount());
          in[6] = uint32_t(o.getSh());
          stats.nVec++;
        })
        .Case<IsaAddOp>([&](IsaAddOp o) {
          in = isa::Instr(isa::Op::VEC_ADD);
          in[2] = sramAddr(o.getA(), o.getAOff());
          in[3] = sramAddr(o.getB(), o.getBOff());
          in[4] = sramAddr(o.getDst(), o.getDstOff());
          in[5] = uint32_t(o.getCount());
          in[6] = uint32_t(o.getShB());
          stats.nVec++;
        })
        .Case<IsaQuantOp>([&](IsaQuantOp o) {
          in = isa::Instr(isa::Op::VEC_QUANT);
          in[2] = sramAddr(o.getSrc(), o.getSrcOff());
          in[3] = sramAddr(o.getDst(), o.getDstOff());
          in[4] = uint32_t(o.getCount());
          in[5] = uint32_t(o.getM());
          in[6] = uint32_t(o.getS());
          stats.nVec++;
        })
        .Case<IsaAttnOp>([&](IsaAttnOp o) {
          in = isa::Instr(isa::Op::ATTN, isa::kFlagAttnWideProb);
          in[2] = sramAddr(o.getQ(), o.getQOff());
          in[3] = sramAddr(o.getOut(), o.getOutOff());
          in[4] = dramAddr(o.getKbaseAttr(), o.getKbaseOff(), o);
          in[5] = dramAddr(o.getVbaseAttr(), o.getVbaseOff(), o);
          in[6] = uint32_t(o.getM());
          in[7] = uint32_t(o.getH());
          in[8] = uint32_t(o.getHkv());
          in[9] = uint32_t(o.getD());
          in[10] = uint32_t(o.getKvStride());
          in[11] = uint32_t(o.getMs());
          in[12] = uint32_t(o.getSs());
          in[13] = uint32_t(o.getMo());
          in[14] = uint32_t(o.getSo());
          stats.nAttn++;
        })
        .Case<IsaKvWriteOp>([&](IsaKvWriteOp o) {
          in = isa::Instr(isa::Op::KV_WRITE);
          in[2] = sramAddr(o.getSrc(), o.getSrcOff());
          in[3] = dramAddr(o.getBaseAttr(), o.getBaseOff(), o);
          in[4] = uint32_t(o.getM());
          in[5] = uint32_t(o.getHkv());
          in[6] = uint32_t(o.getD());
          in[7] = uint32_t(o.getKvStride());
          stats.nAttn++;
        })
        .Case<IsaNopOp>([&](IsaNopOp) {
          in = isa::Instr(isa::Op::NOP);
          stats.nNop++;
        })
        .Case<IsaHaltOp>([&](IsaHaltOp) { in = isa::Instr(isa::Op::HALT); })
        .Default([&](Operation *o) { error = "cannot encode op " + o->getName().getStringRef().str(); });
    if (!error.empty())
      return std::nullopt;
    common(cast<IsaOp>(op), in);
    stats.instructions++;
    return in;
  }
};

struct ProgramImage {
  int64_t M;
  std::vector<uint8_t> bytes;
  int64_t pc = 0;
  ProgramStats stats;
  int64_t sramPeak = 0, sramLivePeak = 0;
};

struct EmitPass : public mlir::llaccel::impl::LLAccelEmitBase<EmitPass> {
  using LLAccelEmitBase::LLAccelEmitBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Location loc = module.getLoc();
    auto mi = ModelInfo::read(module);
    if (failed(mi))
      return signalPassFailure();
    if (output.empty()) {
      module.emitError("llaccel-emit needs output=");
      return signalPassFailure();
    }
    std::string sched = schedule;
    if (auto s = module->getAttrOfType<StringAttr>(kScheduleAttr))
      sched = s.getValue().str();

    // ---- DRAM image: data first, programs last ----
    DramLayout dram;
    WeightOp emb = lookupWeight(module, sym::EMBED);
    if (!emb || kindOf(emb) != kind::EMBED) {
      module.emitError("no quantized `@embed` table");
      return signalPassFailure();
    }
    int64_t embRow = mi->dim * 2;
    {
      ArrayRef<char> raw = emb.getRawData();
      int64_t a = dram.place(sym::EMBED, int64_t(raw.size()), 64);
      std::memcpy(dram.image.data() + a, raw.data(), raw.size());
    }
    int64_t inStride = roundUp(mi->dim * 2, 64), lgStride = roundUp(mi->vocabPadded * 2, 64);
    int64_t kvBytes = 0, constBytes = 0;
    for (auto w : module.getOps<WeightOp>()) {
      StringRef k = kindOf(w);
      if (k == kind::DRAM_INPUT || k == kind::DRAM_LOGITS || k == kind::DRAM_SCRATCH || k == kind::DRAM_KV) {
        dram.place(w.getSymName(), w.getByteSize(), 64);
        if (k == kind::DRAM_KV) kvBytes += w.getByteSize();
      } else if (k == kind::CONSTS || k == kind::GAMMA || k == kind::RQ || k == kind::BIAS) {
        ArrayRef<char> raw = w.getRawData();
        int64_t a = dram.place(w.getSymName(), int64_t(raw.size()), 64);
        std::memcpy(dram.image.data() + a, raw.data(), raw.size());
        if (k == kind::CONSTS)
          constBytes = int64_t(raw.size());
      }
    }
    for (auto w : module.getOps<WeightOp>()) {
      if (kindOf(w) != kind::W)
        continue;
      auto t = w.getTensorType();
      int64_t N = t.getDimSize(0), K = t.getDimSize(1);
      if (N % 16 || K % 16) {
        w.emitError("weight is not padded to 16");
        return signalPassFailure();
      }
      std::vector<uint8_t> tiled = tileWeight(weightDataAs<int8_t>(w), N, K);
      int64_t a = dram.place(w.getSymName(), int64_t(tiled.size()), 256);
      std::memcpy(dram.image.data() + a, tiled.data(), tiled.size());
    }
    if (!dram.addr.contains(sym::DRAM_INPUT) || !dram.addr.contains(sym::DRAM_LOGITS)) {
      module.emitError("module has no dram_input/dram_logits regions (run llaccel-lower-to-isa)");
      return signalPassFailure();
    }

    if (dram.image.size() > uint64_t(UINT32_MAX) + 1) {
      module.emitError("DRAM image exceeds 32-bit device address space");
      return signalPassFailure();
    }

    // ---- programs ----
    std::vector<ProgramImage> programs;
    for (auto fn : module.getOps<func::FuncOp>()) {
      auto mAttr = fn->getAttrOfType<IntegerAttr>(kProgramMAttr);
      if (!mAttr)
        continue;
      ProgramImage p;
      p.M = mAttr.getInt();
      Encoder enc{dram};
      for (Operation &op : fn.getBody().front()) {
        if (!llvm::isa<IsaOp>(op))
          continue;
        auto in = enc.encode(&op);
        if (!in) {
          op.emitError(enc.error);
          return signalPassFailure();
        }
        p.bytes.resize(p.bytes.size() + isa::kInstrBytes);
        in->toBytes(p.bytes.data() + p.bytes.size() - isa::kInstrBytes);
      }
      p.stats = enc.stats;
      if (auto a = fn->getAttrOfType<IntegerAttr>(kSramPeakAttr))
        p.sramPeak = a.getInt();
      if (auto a = fn->getAttrOfType<IntegerAttr>(kSramLivePeakAttr))
        p.sramLivePeak = a.getInt();
      programs.push_back(std::move(p));
    }
    if (programs.empty()) {
      module.emitError("no programs to emit (run llaccel-lower-to-isa)");
      return signalPassFailure();
    }
    llvm::sort(programs, [](const ProgramImage &a, const ProgramImage &b) { return a.M > b.M; });
    for (ProgramImage &p : programs) {
      p.pc = dram.place(("program_m" + Twine(p.M)).str(), int64_t(p.bytes.size()), 64);
      std::memcpy(dram.image.data() + p.pc, p.bytes.data(), p.bytes.size());
    }
    dram.image.resize(size_t(roundUp(int64_t(dram.image.size()), 64)), 0);

    // ---- META_JSON ----
    llvm::json::Object meta;
    meta["model"] = mi->toJson();
    meta["required_capabilities"] = llvm::json::Object{{"attention_head_dim", mi->headDim}};
    llvm::json::Object d;
    d["image_bytes"] = int64_t(dram.image.size());
    d["embedding"] = llvm::json::Object{{"addr", dram.addr[sym::EMBED]}, {"row_bytes", embRow},
                                        {"rows", mi->vocabPadded}};
    d["input"] = llvm::json::Object{{"addr", dram.addr[sym::DRAM_INPUT]}, {"row_bytes", inStride},
                                    {"rows", int64_t(16)}};
    d["logits"] = llvm::json::Object{{"addr", dram.addr[sym::DRAM_LOGITS]}, {"row_bytes", lgStride},
                                     {"rows", int64_t(16)}};
    if (dram.addr.contains(sym::DRAM_SCRATCH))
      d["scratch"] = llvm::json::Object{{"addr", dram.addr[sym::DRAM_SCRATCH]}};
    llvm::json::Object weights;
    for (auto w : module.getOps<WeightOp>())
      if (kindOf(w) == kind::W)
        weights[w.getSymName()] = dram.addr[w.getSymName()];
    d["weights"] = std::move(weights);
    llvm::json::Array kvRegions;
    for (auto w : module.getOps<WeightOp>())
      if (kindOf(w) == kind::DRAM_KV)
        kvRegions.push_back(llvm::json::Object{{"name", w.getSymName()},
                            {"addr", dram.addr[w.getSymName()]}, {"bytes", w.getByteSize()}});
    d["kv_cache"] = std::move(kvRegions);
    d["kv_cache_bytes"] = kvBytes;
    d["consts"] = dram.addr.lookup(sym::CONSTS);
    meta["dram"] = std::move(d);
    llvm::json::Array progs;
    int64_t peak = 0, livePeak = 0;
    llvm::json::Object statsJ;
    for (const ProgramImage &p : programs) {
      progs.push_back(llvm::json::Object{
          {"M", p.M}, {"pc", p.pc}, {"n_instr", int64_t(p.bytes.size() / isa::kInstrBytes)}});
      peak = std::max(peak, p.sramPeak);
      livePeak = std::max(livePeak, p.sramLivePeak);
      statsJ[("M" + Twine(p.M)).str()] = llvm::json::Object{
          {"instructions", p.stats.instructions},
          {"dma_load_bytes", p.stats.dmaLoadBytes},
          {"dma_store_bytes", p.stats.dmaStoreBytes},
          {"gemm_macs", p.stats.gemmMacs},
          {"dma", p.stats.nDma},
          {"gemm", p.stats.nGemm},
          {"vec", p.stats.nVec},
          {"attn", p.stats.nAttn},
          {"nop", p.stats.nNop},
          {"waits", p.stats.nWaits},
          {"sram_peak_used", p.sramPeak},
          {"sram_live_peak", p.sramLivePeak}};
    }
    meta["programs"] = std::move(progs);
    meta["sram"] = llvm::json::Object{{"bytes", int64_t(isa::kSramBytes)},
                                      {"peak_used", peak},
                                      {"live_peak", livePeak},
                                      {"kv_cache_bytes", int64_t(0)},
                                      {"resident_const_bytes", constBytes}};
    meta["stats"] = std::move(statsJ);
    meta["target"] = target;
    meta["fusion"] = bool(fusion);
    meta["schedule"] = sched;
    std::string metaStr = jsonToString(llvm::json::Value(std::move(meta)));

    // ---- container ----
    std::vector<char> out;
    auto put32 = [&](uint32_t v) { out.insert(out.end(), reinterpret_cast<char *>(&v), reinterpret_cast<char *>(&v) + 4); };
    auto put64 = [&](uint64_t v) { out.insert(out.end(), reinterpret_cast<char *>(&v), reinterpret_cast<char *>(&v) + 8); };
    uint32_t nSec = uint32_t(2 + programs.size());
    put32(isa::kLlbinMagic);
    put32(isa::kLlbinVersion);
    put32(nSec);
    int64_t headerBytes = 12 + int64_t(nSec) * int64_t(sizeof(isa::SectionHeader));
    int64_t cursor = roundUp(headerBytes, 64);
    struct Sec { uint32_t kind, flags; uint64_t off, size; const char *data; };
    std::vector<Sec> secs;
    auto addSec = [&](uint32_t kind, uint32_t flags, const char *data, size_t size) {
      secs.push_back({kind, flags, uint64_t(cursor), size, data});
      cursor = roundUp(cursor + int64_t(size), 64);
    };
    addSec(uint32_t(isa::Section::DRAM_IMAGE), 0, reinterpret_cast<const char *>(dram.image.data()),
           dram.image.size());
    for (const ProgramImage &p : programs)
      addSec(uint32_t(isa::Section::PROGRAM), uint32_t(p.M),
             reinterpret_cast<const char *>(p.bytes.data()), p.bytes.size());
    addSec(uint32_t(isa::Section::META_JSON), 0, metaStr.data(), metaStr.size());
    for (const Sec &s : secs) {
      put32(s.kind);
      put32(s.flags);
      put64(s.off);
      put64(s.size);
    }
    out.resize(size_t(cursor), 0);
    for (const Sec &s : secs)
      std::memcpy(out.data() + s.off, s.data, s.size);
    if (failed(writeFile(output, out, loc)))
      return signalPassFailure();

    if (printStats) {
      llvm::outs() << "llaccel-compile: wrote " << output << " (" << out.size() << " bytes; DRAM image "
                   << dram.image.size() << " bytes, target " << target << ", fusion "
                   << (fusion ? "on" : "off") << ", schedule " << sched << ")\n";
      llvm::outs() << "  SRAM: peak used " << peak << " / " << isa::kSramBytes << " bytes ("
                   << llvm::format("%.1f", 100.0 * double(peak) / double(isa::kSramBytes))
                   << "%), live peak " << livePeak << ", DRAM KV cache " << kvBytes
                   << ", resident constants " << constBytes << "\n";
      for (const ProgramImage &p : programs) {
        const ProgramStats &s = p.stats;
        llvm::outs() << "  program M=" << p.M << ": " << s.instructions << " instructions (dma "
                     << s.nDma << ", gemm " << s.nGemm << ", vec " << s.nVec << ", attn " << s.nAttn
                     << ", nop " << s.nNop << ", waits " << s.nWaits << "); DMA load "
                     << s.dmaLoadBytes << " B, store " << s.dmaStoreBytes << " B; GEMM MACs "
                     << s.gemmMacs << "; pc 0x" << llvm::format_hex_no_prefix(p.pc, 1) << "\n";
      }
    }
  }
};

} // namespace
