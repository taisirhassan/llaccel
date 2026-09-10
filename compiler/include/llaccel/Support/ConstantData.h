//===- ConstantData.h - constant tensors attached to llaccel.weight -*- C++ -*-===//
//
// Quantized weights and compiler-generated tables are stored on the
// `llaccel.weight` op as `dense_resource` blobs (DenseResourceElementsAttr),
// so the module stays self-describing and re-parsable by llaccel-opt.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "llaccel/Dialect/LLAccelOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::llaccel {

/// Create (or replace) `llaccel.weight @name` at the end of the module's
/// weight list with the given integer element type and little-endian data.
/// `kind` classifies the constant (see WeightOp description).
WeightOp createWeight(ModuleOp module, llvm::StringRef name, llvm::ArrayRef<int64_t> shape,
                      unsigned elemBits, llvm::ArrayRef<char> data, llvm::StringRef kind,
                      Operation *insertBefore = nullptr);

/// Replace the data (and type) of an existing weight.
void setWeightData(WeightOp w, llvm::ArrayRef<int64_t> shape, unsigned elemBits,
                   llvm::ArrayRef<char> data);

/// Typed views of the raw data.
template <typename T> llvm::ArrayRef<T> weightDataAs(WeightOp w) {
  auto raw = w.getRawData();
  return llvm::ArrayRef<T>(reinterpret_cast<const T *>(raw.data()), raw.size() / sizeof(T));
}

/// ---- exporter files ----
struct F32Weights {
  std::vector<float> data;
  struct Entry { std::string name; std::vector<int64_t> shape; int64_t offset; };
  std::vector<Entry> entries;
  const Entry *find(llvm::StringRef name) const;
  /// Pointer to the first element of `name` (nullptr if missing / out of range).
  const float *get(llvm::StringRef name, int64_t expectedElems) const;
};

/// Load weights.bin + weights.json; emits diagnostics through `loc`.
FailureOr<F32Weights> loadF32Weights(llvm::StringRef binPath, llvm::StringRef jsonPath,
                                     Location loc);
/// Load calib.json into a name -> absmax map.
FailureOr<llvm::StringMap<double>> loadCalib(llvm::StringRef path, Location loc);

/// Read a whole file; nullopt (with diagnostic) on failure.
std::optional<std::string> readFile(llvm::StringRef path, Location loc);
/// Write a whole file; failure with diagnostic.
LogicalResult writeFile(llvm::StringRef path, llvm::ArrayRef<char> data, Location loc);

/// Pretty JSON serialisation.
std::string jsonToString(const llvm::json::Value &v);

} // namespace mlir::llaccel
