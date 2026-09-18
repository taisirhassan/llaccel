//===- QGraphDump.cpp - qgraph.json + qweights.bin (docs/DIALECT.md §2) ---===//
//
// Serialises the quantized (+fused) high-level module in the exact form the
// numpy golden model executes. Tensor blobs are padded to 64 bytes like the
// Python reference quantizer so the two dumps are directly comparable.
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace mlir;
using namespace mlir::llaccel;

namespace {

struct Blob {
  std::vector<char> bytes;
  llvm::json::Object tensors;

  void add(StringRef name, WeightOp w, StringRef dtype, ArrayRef<int64_t> shape,
           std::optional<int64_t> exp) {
    llvm::json::Object spec;
    spec["dtype"] = dtype;
    llvm::json::Array sh;
    for (auto d : shape) sh.push_back(d);
    spec["shape"] = std::move(sh);
    spec["offset"] = int64_t(bytes.size());
    if (exp)
      spec["exp"] = *exp;
    ArrayRef<char> raw = w.getRawData();
    bytes.insert(bytes.end(), raw.begin(), raw.end());
    size_t pad = (64 - bytes.size() % 64) % 64;
    bytes.insert(bytes.end(), pad, 0);
    tensors[name] = std::move(spec);
  }
};

std::string nm(Value v) { return getValueName(v).value_or("?").str(); }

llvm::json::Value symOrNull(std::optional<StringRef> s) {
  if (s)
    return llvm::json::Value(s->str());
  return llvm::json::Value(nullptr);
}

} // namespace

LogicalResult mlir::llaccel::dumpQGraph(ModuleOp module, StringRef dir, int64_t prefillM) {
  Location loc = module.getLoc();
  auto mi = ModelInfo::read(module);
  if (failed(mi))
    return failure();
  if (!mi->quantized)
    return emitError(loc) << "--dump-qgraph: module is not quantized";
  func::FuncOp fn = getForwardFunc(module);
  if (!fn)
    return emitError(loc) << "--dump-qgraph: no high-level function";

  // ---- tensors ----
  Blob blob;
  for (auto w : module.getOps<WeightOp>()) {
    StringRef k = kindOf(w);
    auto t = w.getTensorType();
    if (k == kind::EMBED || k == kind::GAMMA)
      blob.add(w.getSymName(), w, "i16", t.getShape(), w.getExp());
    else if (k == kind::ROPE_COS || k == kind::ROPE_SIN)
      blob.add(w.getSymName(), w, "i16", t.getShape(), std::nullopt);
    else if (k == kind::W)
      blob.add(w.getSymName(), w, "i8", t.getShape(), std::nullopt);
    else if (k == kind::RQ)
      blob.add(w.getSymName(), w, "rq", t.getShape().take_front(1), std::nullopt);
    else if (k == kind::BIAS)
      blob.add(w.getSymName(), w, "i32", t.getShape(), std::nullopt);
  }

  // ---- ops ----
  llvm::json::Array ops;
  llvm::json::Object exps, scales;
  bool fusion = false;
  auto annotate = [&](Value v) {
    auto n = getValueName(v);
    if (!n)
      return;
    if (auto e = getValueExp(v))
      exps[*n] = *e;
    if (auto s = getValueScale(v))
      scales[*n] = *s;
  };
  annotate(fn.getArgument(0));
  for (Operation &op : fn.getBody().front()) {
    for (Value r : op.getResults()) annotate(r);
    llvm::TypeSwitch<Operation *>(&op)
        .Case<RmsNormOp>([&](RmsNormOp o) {
          ops.push_back(llvm::json::Object{{"op", "rmsnorm"},
                                           {"in", nm(o.getInput())},
                                           {"gamma", o.getGamma().str()},
                                           {"out", nm(o.getOutput())},
                                           {"K", tensorWidth(o.getInput()) / o.getHeads()},
                                           {"heads", o.getHeads()},
                                           {"eps_t", o.getEpsT().value_or(0)},
                                           {"C", o.getC().value_or(0)},
                                           {"sh_post", o.getShPost().value_or(0)}});
        })
        .Case<QuantOp>([&](QuantOp o) {
          ops.push_back(llvm::json::Object{{"op", "quant"},
                                           {"in", nm(o.getInput())},
                                           {"out", nm(o.getOutput())},
                                           {"M", int64_t(o.getM())},
                                           {"S", int64_t(o.getS())}});
        })
        .Case<LinearOp>([&](LinearOp o) {
          if (o.getEpilogue() != EpilogueMode::none)
            fusion = true;
          llvm::json::Value silu(nullptr);
          if (o.getEpilogue() == EpilogueMode::silu)
            silu = llvm::json::Object{{"Mi", o.getSiluMi().value_or(0)},
                                      {"Si", o.getSiluSi().value_or(0)},
                                      {"sh_out", o.getSiluShOut().value_or(0)}};
          ops.push_back(llvm::json::Object{
              {"op", "linear"},
              {"in", nm(o.getInput())},
              {"w", o.getWeight().str()},
              {"rq", symOrNull(o.getRq())},
              {"bias", symOrNull(o.getBias())},
              {"out", nm(o.getOutput())},
              {"N", o.getN()},
              {"K", o.getK()},
              {"out_dtype", isI8Tensor(o.getOutput()) ? "i8" : "i16"},
              {"epilogue", stringifyEpilogueMode(o.getEpilogue()).str()},
              {"aux", o.getAux() ? llvm::json::Value(nm(o.getAux())) : llvm::json::Value(nullptr)},
              {"aux_shift", o.getAuxShift().value_or(0)},
              {"silu", std::move(silu)}});
        })
        .Case<RopeOp>([&](RopeOp o) {
          int64_t H = o.getHeads();
          ops.push_back(llvm::json::Object{{"op", "rope"},
                                           {"in", nm(o.getInput())},
                                           {"out", nm(o.getOutput())},
                                           {"H", H},
                                           {"D", tensorWidth(o.getInput()) / H}});
        })
        .Case<AttentionOp>([&](AttentionOp o) {
          ops.push_back(llvm::json::Object{{"op", "kv_write"},
                                           {"layer", o.getLayer()},
                                           {"k", nm(o.getK())},
                                           {"v", nm(o.getV())},
                                           {"Hkv", o.getKvHeads()},
                                           {"D", o.getHeadDim()}});
          ops.push_back(llvm::json::Object{{"op", "attention"}, {"prob_bits", int64_t(15)},
                                           {"q", nm(o.getQ())},
                                           {"layer", o.getLayer()},
                                           {"out", nm(o.getOutput())},
                                           {"H", o.getHeads()},
                                           {"Hkv", o.getKvHeads()},
                                           {"D", o.getHeadDim()},
                                           {"Ms", o.getMs().value_or(0)},
                                           {"Ss", o.getSs().value_or(0)},
                                           {"Mo", o.getMo().value_or(0)},
                                           {"So", o.getSo().value_or(0)}});
        })
        .Case<AddOp>([&](AddOp o) {
          ops.push_back(llvm::json::Object{{"op", "add"},
                                           {"a", nm(o.getA())},
                                           {"b", nm(o.getB())},
                                           {"out", nm(o.getOutput())},
                                           {"sh_b", o.getShB().value_or(0)}});
        })
        .Case<MulOp>([&](MulOp o) {
          ops.push_back(llvm::json::Object{{"op", "mul"},
                                           {"a", nm(o.getA())},
                                           {"b", nm(o.getB())},
                                           {"out", nm(o.getOutput())},
                                           {"sh", o.getSh().value_or(0)}});
        })
        .Case<SiluOp>([&](SiluOp o) {
          ops.push_back(llvm::json::Object{{"op", "silu"},
                                           {"in", nm(o.getInput())},
                                           {"out", nm(o.getOutput())},
                                           {"Mi", o.getMi().value_or(0)},
                                           {"Si", o.getSi().value_or(0)},
                                           {"sh_out", o.getShOut().value_or(0)}});
        });
  }
  auto ret = cast<func::ReturnOp>(fn.getBody().front().getTerminator());

  llvm::json::Object root;
  auto model = mi->toJson();
  model["prefill_m"] = prefillM;
  root["model"] = std::move(model);
  root["weights_file"] = "qweights.bin";
  root["tensors"] = std::move(blob.tensors);
  root["ops"] = std::move(ops);
  root["input"] = nm(fn.getArgument(0));
  root["output"] = nm(ret.getOperand(0));
  root["exps"] = std::move(exps);
  root["scales"] = std::move(scales);
  root["fusion"] = fusion;

  if (auto ec = llvm::sys::fs::create_directories(dir))
    return emitError(loc) << "cannot create `" << dir << "`: " << ec.message();
  SmallString<256> jsonPath(dir), binPath(dir);
  llvm::sys::path::append(jsonPath, "qgraph.json");
  llvm::sys::path::append(binPath, "qweights.bin");
  std::string js = jsonToString(llvm::json::Value(std::move(root)));
  js += "\n";
  if (failed(writeFile(jsonPath, ArrayRef<char>(js.data(), js.size()), loc)) ||
      failed(writeFile(binPath, blob.bytes, loc)))
    return failure();
  return success();
}
