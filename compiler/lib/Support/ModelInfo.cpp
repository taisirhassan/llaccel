//===- ModelInfo.cpp ------------------------------------------------------===//
#include "llaccel/Support/ModelInfo.h"

#include "llaccel/Dialect/LLAccelDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::llaccel;

FailureOr<ModelInfo> ModelInfo::read(ModuleOp module) {
  auto dict = module->getAttrOfType<DictionaryAttr>(LLAccelDialect::kModelAttr);
  if (!dict)
    return module.emitError("module lacks the `llaccel.model` dictionary attribute");
  ModelInfo m;
  auto geti = [&](StringRef key, int64_t &out, bool required = true) -> LogicalResult {
    auto a = dict.getAs<IntegerAttr>(key);
    if (!a) {
      if (required)
        return module.emitError("llaccel.model: missing integer field `") << key << "`";
      return success();
    }
    out = a.getInt();
    return success();
  };
  auto getf = [&](StringRef key, double &out) -> LogicalResult {
    auto a = dict.getAs<FloatAttr>(key);
    if (!a)
      return module.emitError("llaccel.model: missing float field `") << key << "`";
    out = a.getValueAsDouble();
    return success();
  };
  if (failed(geti("dim", m.dim)) || failed(geti("n_layers", m.nLayers)) ||
      failed(geti("n_heads", m.nHeads)) || failed(geti("n_kv_heads", m.nKvHeads)) ||
      failed(geti("head_dim", m.headDim)) || failed(geti("ffn", m.ffn)) ||
      failed(geti("vocab", m.vocab)) || failed(geti("max_seq", m.maxSeq)) ||
      failed(getf("rope_base", m.ropeBase)) || failed(getf("rms_eps", m.rmsEps)))
    return failure();
  if (dict.contains("E_RES")) {
    m.quantized = true;
    if (failed(geti("E_RES", m.eRes)) || failed(geti("E_LOGIT", m.eLogit)) ||
        failed(geti("vocab_padded", m.vocabPadded)))
      return failure();
  } else {
    m.vocabPadded = (m.vocab + 15) / 16 * 16;
  }
  return m;
}

void ModelInfo::write(ModuleOp module) const {
  Builder b(module.getContext());
  SmallVector<NamedAttribute> attrs{
      b.getNamedAttr("dim", b.getI64IntegerAttr(dim)),
      b.getNamedAttr("n_layers", b.getI64IntegerAttr(nLayers)),
      b.getNamedAttr("n_heads", b.getI64IntegerAttr(nHeads)),
      b.getNamedAttr("n_kv_heads", b.getI64IntegerAttr(nKvHeads)),
      b.getNamedAttr("head_dim", b.getI64IntegerAttr(headDim)),
      b.getNamedAttr("ffn", b.getI64IntegerAttr(ffn)),
      b.getNamedAttr("vocab", b.getI64IntegerAttr(vocab)),
      b.getNamedAttr("max_seq", b.getI64IntegerAttr(maxSeq)),
      b.getNamedAttr("rope_base", b.getF64FloatAttr(ropeBase)),
      b.getNamedAttr("rms_eps", b.getF64FloatAttr(rmsEps)),
  };
  if (quantized) {
    attrs.push_back(b.getNamedAttr("vocab_padded", b.getI64IntegerAttr(vocabPadded)));
    attrs.push_back(b.getNamedAttr("E_RES", b.getI64IntegerAttr(eRes)));
    attrs.push_back(b.getNamedAttr("E_LOGIT", b.getI64IntegerAttr(eLogit)));
  }
  module->setAttr(LLAccelDialect::kModelAttr, b.getDictionaryAttr(attrs));
}

llvm::json::Object ModelInfo::toJson() const {
  llvm::json::Object o;
  o["dim"] = dim;
  o["n_layers"] = nLayers;
  o["n_heads"] = nHeads;
  o["n_kv_heads"] = nKvHeads;
  o["head_dim"] = headDim;
  o["ffn"] = ffn;
  o["vocab"] = vocab;
  o["vocab_padded"] = vocabPadded;
  o["max_seq"] = maxSeq;
  o["rope_base"] = ropeBase;
  o["rms_eps"] = rmsEps;
  if (quantized) {
    o["E_RES"] = eRes;
    o["E_LOGIT"] = eLogit;
  }
  return o;
}
