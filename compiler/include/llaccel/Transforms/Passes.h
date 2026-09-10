//===- Passes.h - llaccel compiler passes ------------------------*- C++ -*-===//
#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::llaccel {

#define GEN_PASS_DECL
#include "llaccel/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "llaccel/Transforms/Passes.h.inc"

/// Write qgraph.json + qweights.bin (docs/DIALECT.md section 2) for the
/// quantized (+fused) high-level module into `dir`.
LogicalResult dumpQGraph(ModuleOp module, llvm::StringRef dir);

} // namespace mlir::llaccel
