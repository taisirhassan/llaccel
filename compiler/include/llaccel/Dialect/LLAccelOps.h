//===- LLAccelOps.h - llaccel operations -------------------------*- C++ -*-===//
#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llaccel/Dialect/LLAccelDialect.h"
#include "llaccel/isa.h"

#include <optional>

namespace mlir::llaccel {

/// Engine queue an ISA instruction is issued to (mirrors ::llaccel::Engine).
enum class IsaEngine : uint8_t { CP = 0, DMA = 1, GEMM = 2, VEC = 3, ATTN = 4 };
inline llvm::StringRef isaEngineName(IsaEngine e) {
  switch (e) {
  case IsaEngine::CP: return "cp";
  case IsaEngine::DMA: return "dma";
  case IsaEngine::GEMM: return "gemm";
  case IsaEngine::VEC: return "vec";
  case IsaEngine::ATTN: return "attn";
  }
  return "?";
}

/// One byte range touched by an instruction. Either an SRAM range (buf set,
/// offset relative to the buffer) or a DRAM range (dram symbol set).
struct BufAccess {
  Value buf;                 // SRAM buffer (null for DRAM)
  FlatSymbolRefAttr dram;    // DRAM symbol (null for SRAM)
  int64_t offset = 0;        // bytes from buffer / symbol start
  int64_t size = 0;          // bytes
  bool write = false;
};

} // namespace mlir::llaccel

#include "llaccel/Dialect/LLAccelInterfaces.h.inc"

#define GET_OP_CLASSES
#include "llaccel/Dialect/LLAccelOps.h.inc"

namespace mlir::llaccel {

/// ---- value annotations (discardable attrs on defining ops / block args) ----
std::optional<llvm::StringRef> getValueName(Value v);
std::optional<int64_t> getValueExp(Value v);
std::optional<double> getValueScale(Value v);
void setValueName(Value v, llvm::StringRef name);
void setValueExp(Value v, int64_t exp);
void setValueScale(Value v, double scale);

/// Width (last dim) and element byte size of a ranked tensor value.
int64_t tensorWidth(Value v);
int64_t tensorElemBytes(Value v);

/// Look up a weight symbol in the module enclosing `from`.
WeightOp lookupWeight(Operation *from, llvm::StringRef name);
WeightOp lookupWeight(Operation *from, FlatSymbolRefAttr ref);

} // namespace mlir::llaccel
