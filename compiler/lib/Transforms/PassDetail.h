//===- PassDetail.h - shared helpers for the llaccel passes ------*- C++ -*-===//
#pragma once

#include "llaccel/Dialect/LLAccelOps.h"
#include "llaccel/Support/ConstantData.h"
#include "llaccel/Support/ModelInfo.h"
#include "llaccel/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::llaccel {

/// Constant kinds (WeightOp `kind`).
namespace kind {
inline constexpr llvm::StringLiteral W = "w", RQ = "rq", BIAS = "bias", GAMMA = "gamma",
                                     EMBED = "embed", ROPE_COS = "rope_cos", ROPE_SIN = "rope_sin",
                                     ROPE_TABLE = "rope_table", CONSTS = "consts",
                                     DRAM_INPUT = "dram_input", DRAM_LOGITS = "dram_logits",
                                     DRAM_SCRATCH = "dram_scratch", DRAM_KV = "dram_kv";
}

/// Symbol names fixed by docs/DIALECT.md and by this compiler.
namespace sym {
inline constexpr llvm::StringLiteral EMBED = "embed", ROPE_COS = "rope_cos", ROPE_SIN = "rope_sin",
                                     ROPE_TABLE = "rope_table", CONSTS = "consts",
                                     DRAM_INPUT = "dram_input", DRAM_LOGITS = "dram_logits",
                                     DRAM_SCRATCH = "dram_scratch";
}

/// Alloc regions (AllocOp `region`).
namespace region {
inline constexpr llvm::StringLiteral ACT = "act", WEIGHT = "weight", CONST = "const", KV = "kv",
                                     TMP = "tmp";
}

/// Function attribute carrying the program's M (rows).
inline constexpr llvm::StringLiteral kProgramMAttr = "llaccel.program_m";
/// Function attribute set by llaccel-alloc-sram: high-water mark in bytes.
inline constexpr llvm::StringLiteral kSramPeakAttr = "llaccel.sram_peak";
/// Function attribute set by llaccel-alloc-sram: peak of simultaneously live bytes.
inline constexpr llvm::StringLiteral kSramLivePeakAttr = "llaccel.sram_live_peak";
/// Module attribute carrying the schedule mode used.
inline constexpr llvm::StringLiteral kScheduleAttr = "llaccel.schedule";

inline int64_t roundUp(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

inline RankedTensorType actType(MLIRContext *ctx, int64_t width, unsigned bits) {
  return RankedTensorType::get({ShapedType::kDynamic, width}, IntegerType::get(ctx, bits));
}
inline bool isI8Tensor(Value v) {
  return cast<RankedTensorType>(v.getType()).getElementTypeBitWidth() == 8;
}
inline bool isI16Tensor(Value v) {
  return cast<RankedTensorType>(v.getType()).getElementTypeBitWidth() == 16;
}
inline llvm::StringRef kindOf(WeightOp w) { return w.getKindAttr() ? w.getKind().value() : ""; }

/// The single high-level function of a pre-lowering module (`@forward`).
inline func::FuncOp getForwardFunc(ModuleOp module) {
  for (auto fn : module.getOps<func::FuncOp>())
    return fn;
  return nullptr;
}

/// Per-program stats also used by the emitter.
struct ProgramStats {
  int64_t instructions = 0, dmaLoadBytes = 0, dmaStoreBytes = 0, gemmMacs = 0;
  int64_t nDma = 0, nGemm = 0, nVec = 0, nAttn = 0, nNop = 0, nWaits = 0;
};

} // namespace mlir::llaccel
