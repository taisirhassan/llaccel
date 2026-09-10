//===- ModelInfo.h - the `llaccel.model` module attribute -------*- C++ -*-===//
#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/JSON.h"

#include <cstdint>

namespace mlir::llaccel {

/// Model hyper-parameters (docs/DIALECT.md) plus the values quantization adds.
struct ModelInfo {
  int64_t dim = 0, nLayers = 0, nHeads = 0, nKvHeads = 0, headDim = 0, ffn = 0;
  int64_t vocab = 0, vocabPadded = 0, maxSeq = 0;
  double ropeBase = 10000.0, rmsEps = 1e-5;
  bool quantized = false;  // E_RES / E_LOGIT / vocabPadded valid
  int64_t eRes = 0, eLogit = 0;

  static FailureOr<ModelInfo> read(ModuleOp module);
  void write(ModuleOp module) const;
  llvm::json::Object toJson() const;
};

} // namespace mlir::llaccel
