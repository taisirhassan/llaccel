//===- Quantize.cpp - f32 graph -> integer graph (docs/NUMERICS.md) -------===//
//
// Every "Compiler:" decision of NUMERICS.md is made here, once:
//   * weights: int8 per-output-channel (s[n] = max|W[n,:]|/127, zero rows -> 1)
//   * i8 activations: s = absmax/127; i16 activations: e = max(ceil(log2(absmax/32767)), -15)
//   * E_RES names only the embedding/input exponent. Residual branches reserve
//     headroom for their immediate ADD; ADD aligns the finer operand to the
//     coarser exponent using sh_b. E_LOGIT comes from calibrated logits.
//   * GEMM outputs: q, k, g, u -> i16 at their own exponent; v -> i8 (it feeds
//     the KV cache directly); o, d -> per-residual i16; lm_head -> i16 at E_LOGIT
//     with N padded to vocab_padded (multiple of 16).
//   * `llaccel.quant` (i16 -> i8, scale absmax/127 of the i16 tensor) is inserted
//     in front of every consumer that needs i8: linears (h, h2, f, hn) and the
//     attention q/k operands (qr, kr).
//   * requant tables {M in [2^30,2^31), S}, biases, RMSNorm eps_t/C/sh_post,
//     SiLU Mi/Si/sh_out, MUL/ADD shifts, attention Ms/Ss/Mo/So (QuantParams.h).
//   * constants: gammas (i16, own exponent), embedding (i16 at E_RES, padded to
//     vocab_padded rows), RoPE cos/sin (Q1.14, [max_seq][D/2]) plus the device
//     table layout rope_table ([max_seq][cos D/2 | sin D/2]).
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"
#include "llaccel/Support/QuantParams.h"

#include "llvm/ADT/StringSet.h"

namespace mlir::llaccel {
#define GEN_PASS_DEF_LLACCELQUANTIZE
#include "llaccel/Transforms/Passes.h.inc"
} // namespace mlir::llaccel

using namespace mlir;
using namespace mlir::llaccel;
namespace qp = ::llaccel::qp;

namespace {

template <typename T> ArrayRef<char> bytesOf(const std::vector<T> &v) {
  return ArrayRef<char>(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(T));
}

class Quantizer {
public:
  Quantizer(ModuleOp module, const F32Weights &f32, const llvm::StringMap<double> &calib,
            ModelInfo mi)
      : module(module), ctx(module.getContext()), f32(f32), calib(calib), mi(mi) {}

  LogicalResult run();

private:
  // ---- lookups ----
  FailureOr<double> absmax(Operation *at, StringRef name) {
    auto it = calib.find(name);
    if (it == calib.end())
      return at->emitError("calib.json has no entry for tensor `") << name << "`";
    return it->second;
  }
  // Selected clipping bounds are separate from true maxima: RMS reciprocal
  // range planning still reads the latter. Missing bounds retain max calibration.
  FailureOr<double> quantAbsmax(Operation *at, StringRef name, unsigned bits) {
    auto raw = absmax(at, name);
    if (failed(raw)) return failure();
    auto it = calib.find((name + (bits == 8 ? ".i8_absmax" : ".i16_absmax")).str());
    if (it == calib.end()) return *raw;
    double bound = it->second;
    if (!std::isfinite(bound) || bound < 0 || bound > *raw || (*raw > 0 && bound == 0))
      return at->emitError("invalid selected quantization bound for `") << name << "`";
    return bound;
  }
  FailureOr<StringRef> nameOf(Value v, Operation *at) {
    if (auto n = getValueName(v))
      return *n;
    return at->emitError("value has no `llaccel.name` annotation");
  }
  FailureOr<int64_t> expOf(Value v, Operation *at) {
    if (!isI16Tensor(v))
      return at->emitError("expected an i16 operand");
    if (auto e = getValueExp(v))
      return *e;
    return at->emitError("i16 value has no `llaccel.exp` annotation");
  }
  FailureOr<double> scaleOf(Value v, Operation *at) {
    if (!isI8Tensor(v))
      return at->emitError("expected an i8 operand");
    if (auto s = getValueScale(v))
      return *s;
    return at->emitError("i8 value has no `llaccel.scale` annotation");
  }
  template <typename T> FailureOr<T> check(qp::Result<T> r, Operation *at) {
    if (!r)
      return at->emitError(r.error());
    return *r;
  }
  const float *f32Of(WeightOp w, int64_t elems, Operation *at) {
    const float *p = f32.get(w.getSymName(), elems);
    if (!p)
      at->emitError("weights.bin/json has no f32 tensor `") << w.getSymName() << "` with "
                                                              << elems << " elements";
    return p;
  }

  // ---- constants ----
  LogicalResult quantizeEmbedding();
  FailureOr<int64_t> quantizeGamma(WeightOp g, Operation *at);
  LogicalResult ensureRopeTables(int64_t D, Operation *at);
  LogicalResult quantizeLinearWeights(LinearOp op, double sA, bool outI8, int64_t eOut,
                                      double sOut, int64_t nPad);

  // ---- activations ----
  FailureOr<Value> ensureI8(Value v, Operation *user);
  LogicalResult visit(RmsNormOp op);
  LogicalResult visit(LinearOp op);
  LogicalResult visit(RopeOp op);
  LogicalResult visit(AttentionOp op);
  LogicalResult visit(SiluOp op);
  LogicalResult visit(MulOp op);
  LogicalResult visit(AddOp op);

  ModuleOp module;
  MLIRContext *ctx;
  const F32Weights &f32;
  const llvm::StringMap<double> &calib;
  ModelInfo mi;
  func::FuncOp fn;
  Value returned;
  DenseMap<Value, Value> quantCache;
  bool ropeDone = false;
};

LogicalResult Quantizer::run() {
  fn = getForwardFunc(module);
  if (!fn)
    return module.emitError("module has no func.func to quantize");
  if (fn.getNumArguments() != 1 || fn.getNumResults() != 1)
    return fn.emitError("expected exactly one argument and one result");
  if (fn.isExternal() || !llvm::hasSingleElement(fn.getBody()))
    return fn.emitError("expected a defined, single-block forward function");
  Block &body = fn.getBody().front();
  auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!ret)
    return fn.emitError("expected a func.return terminator");
  returned = ret.getOperand(0);

  if (mi.dim % 16 || mi.ffn % 16 || (mi.nKvHeads * mi.headDim) % 16 ||
      (mi.nHeads * mi.headDim) % 16)
    return module.emitError("dim, ffn, n_heads*head_dim and n_kv_heads*head_dim must be "
                            "multiples of 16 (GEMM K / VEC count granularity)");
  if (mi.headDim != 16 && mi.headDim != 32 && mi.headDim != 64 && mi.headDim != 128 && mi.headDim != 256)
    return module.emitError("head_dim must be 16, 32, 64, 128 or 256");
  mi.vocabPadded = roundUp(mi.vocab, 16);

  // ---- E_RES / E_LOGIT ----
  BlockArgument arg = body.getArgument(0);
  auto inName = nameOf(arg, fn);
  if (failed(inName))
    return failure();
  auto inputMaximum = quantAbsmax(fn, *inName, 16);
  if (failed(inputMaximum))
    return failure();
  int64_t eRes = qp::i16Exponent(*inputMaximum);
  auto outName = nameOf(returned, ret);
  if (failed(outName))
    return failure();
  auto lgAbs = quantAbsmax(ret, *outName, 16);
  if (failed(lgAbs))
    return failure();
  mi.eRes = eRes;
  mi.eLogit = qp::i16Exponent(*lgAbs);
  mi.quantized = true;

  // ---- input ----
  arg.setType(actType(ctx, mi.dim, 16));
  setValueExp(arg, eRes);

  if (failed(quantizeEmbedding()))
    return failure();

  // ---- ops, in program order ----
  for (Operation &op : llvm::make_early_inc_range(body)) {
    LogicalResult r =
        llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            .Case<RmsNormOp, LinearOp, RopeOp, AttentionOp, SiluOp, MulOp, AddOp>(
                [&](auto o) { return visit(o); })
            .Case<QuantOp>([&](QuantOp) { return success(); })
            .Case<func::ReturnOp>([&](func::ReturnOp) { return success(); })
            .Default([&](Operation *o) {
              return o->emitError("unsupported op in the high-level graph");
            });
    if (failed(r))
      return failure();
  }
  if (!isI16Tensor(returned) || tensorWidth(returned) != mi.vocabPadded)
    return ret.emitError("the returned value must be the lm_head linear (i16 at E_LOGIT)");
  fn.setType(FunctionType::get(ctx, {arg.getType()}, {returned.getType()}));
  mi.write(module);
  return success();
}

// ---- constants ---------------------------------------------------------------------------------

LogicalResult Quantizer::quantizeEmbedding() {
  WeightOp emb = lookupWeight(module, sym::EMBED);
  if (!emb)
    return module.emitError("module has no `@embed` weight");
  auto t = emb.getTensorType();
  if (t.getRank() != 2 || t.getDimSize(0) != mi.vocab || t.getDimSize(1) != mi.dim)
    return emb.emitError("@embed must be tensor<vocab x dim x f32>");
  const float *src = f32Of(emb, mi.vocab * mi.dim, emb);
  if (!src)
    return failure();
  std::vector<int16_t> q(size_t(mi.vocabPadded * mi.dim), 0);
  for (int64_t i = 0; i < mi.vocab * mi.dim; ++i)
    q[size_t(i)] = qp::quantI16(src[i], mi.eRes);
  setWeightData(emb, {mi.vocabPadded, mi.dim}, 16, bytesOf(q));
  emb.setKindAttr(StringAttr::get(ctx, kind::EMBED));
  emb.setExpAttr(IntegerAttr::get(IntegerType::get(ctx, 64), mi.eRes));
  return success();
}

FailureOr<int64_t> Quantizer::quantizeGamma(WeightOp g, Operation *at) {
  if (kindOf(g) == kind::GAMMA)
    return g.getExp().value();
  auto t = g.getTensorType();
  if (t.getRank() != 1 || !t.getElementType().isF32())
    return at->emitError("gamma `") << g.getSymName() << "` must be a rank-1 f32 tensor";
  int64_t K = t.getDimSize(0);
  const float *src = f32Of(g, K, at);
  if (!src)
    return failure();
  double mx = 0.0;
  for (int64_t k = 0; k < K; ++k) mx = std::max(mx, double(std::fabs(src[k])));
  int64_t eG = qp::i16Exponent(mx);
  std::vector<int16_t> q(static_cast<size_t>(K));
  for (int64_t k = 0; k < K; ++k) q[size_t(k)] = qp::quantI16(src[k], eG);
  setWeightData(g, {K}, 16, bytesOf(q));
  g.setKindAttr(StringAttr::get(ctx, kind::GAMMA));
  g.setExpAttr(IntegerAttr::get(IntegerType::get(ctx, 64), eG));
  return eG;
}

LogicalResult Quantizer::ensureRopeTables(int64_t D, Operation *at) {
  if (ropeDone)
    return success();
  if (D != mi.headDim)
    return at->emitError("rope head width does not match the model's head_dim");
  std::vector<int16_t> cosT, sinT;
  auto cosInput = lookupWeight(module, "rope_cos_input");
  auto sinInput = lookupWeight(module, "rope_sin_input");
  if (bool(cosInput) != bool(sinInput))
    return at->emitError("explicit RoPE requires both rope_cos_input and rope_sin_input");
  if (cosInput) {
    auto convert = [&](WeightOp input, std::vector<int16_t> &output) -> LogicalResult {
      auto t = input.getTensorType();
      if (t.getRank() != 2 || t.getDimSize(0) != mi.maxSeq || t.getDimSize(1) != D / 2)
        return at->emitError("explicit RoPE table must have shape [max_seq, head_dim/2]");
      auto *values = f32Of(input, mi.maxSeq * D / 2, at);
      if (!values) return failure();
      output.resize(size_t(mi.maxSeq * D / 2));
      for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(values[i]) || values[i] < -2.0f || values[i] > 32767.0f / 16384.0f)
          return at->emitError("explicit RoPE entries must be finite and representable in signed Q1.14");
        output[i] = qp::quantI16(values[i], -14);
      }
      return success();
    };
    if (failed(convert(cosInput, cosT)) || failed(convert(sinInput, sinT))) return failure();
    cosInput.erase();
    sinInput.erase();
  } else {
    qp::ropeTables(mi.maxSeq, D, mi.ropeBase, cosT, sinT);
  }
  int64_t half = D / 2;
  std::vector<int16_t> table(size_t(mi.maxSeq * D));
  for (int64_t p = 0; p < mi.maxSeq; ++p)
    for (int64_t i = 0; i < half; ++i) {
      table[size_t(p * D + i)] = cosT[size_t(p * half + i)];
      table[size_t(p * D + half + i)] = sinT[size_t(p * half + i)];
    }
  createWeight(module, sym::ROPE_COS, {mi.maxSeq, half}, 16, bytesOf(cosT), kind::ROPE_COS);
  createWeight(module, sym::ROPE_SIN, {mi.maxSeq, half}, 16, bytesOf(sinT), kind::ROPE_SIN);
  createWeight(module, sym::ROPE_TABLE, {mi.maxSeq, D}, 16, bytesOf(table), kind::ROPE_TABLE);
  ropeDone = true;
  return success();
}

LogicalResult Quantizer::quantizeLinearWeights(LinearOp op, double sA, bool outI8, int64_t eOut,
                                               double sOut, int64_t nPad) {
  WeightOp w = lookupWeight(op, op.getWeightAttr());
  if (!w)
    return op.emitError("unknown weight symbol `") << op.getWeight() << "`";
  auto t = w.getTensorType();
  if (t.getRank() != 2 || !t.getElementType().isF32())
    return op.emitError("weight `") << w.getSymName()
                                    << "` is not an f32 matrix (a weight may only feed one linear)";
  int64_t N = t.getDimSize(0), K = t.getDimSize(1);
  const float *src = f32Of(w, N * K, op);
  if (!src)
    return failure();
  std::vector<int8_t> q(size_t(nPad * K), 0);
  std::vector<double> sW(size_t(nPad), 1.0);
  for (int64_t n = 0; n < N; ++n) {
    sW[size_t(n)] = qp::weightRowScale(src + n * K, K);
    for (int64_t k = 0; k < K; ++k)
      q[size_t(n * K + k)] = qp::quantI8(src[n * K + k], sW[size_t(n)]);
  }
  setWeightData(w, {nPad, K}, 8, bytesOf(q));
  w.setKindAttr(StringAttr::get(ctx, kind::W));
  if (nPad != N)
    w.setOrigNAttr(IntegerAttr::get(IntegerType::get(ctx, 64), N));

  std::vector<int32_t> rq(size_t(nPad * 2));
  for (int64_t n = 0; n < nPad; ++n) {
    auto ms = check(qp::gemmRq(sA, sW[size_t(n)], outI8, eOut, sOut), op);
    if (failed(ms))
      return failure();
    rq[size_t(2 * n)] = int32_t(ms->M);
    rq[size_t(2 * n + 1)] = int32_t(ms->S);
  }
  std::string rqName = (w.getSymName() + ".rq").str();
  createWeight(module, rqName, {nPad, 2}, 32, bytesOf(rq), kind::RQ, w->getNextNode());
  op.setRqAttr(FlatSymbolRefAttr::get(ctx, rqName));

  if (op.getBiasAttr()) {
    WeightOp b = lookupWeight(op, op.getBiasAttr());
    if (!b)
      return op.emitError("unknown bias symbol `") << *op.getBias() << "`";
    auto bt = b.getTensorType();
    if (bt.getRank() != 1 || bt.getDimSize(0) != N || !bt.getElementType().isF32())
      return op.emitError("bias `") << b.getSymName() << "` must be tensor<" << N << "xf32>";
    const float *bs = f32Of(b, N, op);
    if (!bs)
      return failure();
    std::vector<int32_t> bq(size_t(nPad), 0);
    for (int64_t n = 0; n < N; ++n) bq[size_t(n)] = qp::gemmBias(bs[n], sA, sW[size_t(n)]);
    setWeightData(b, {nPad}, 32, bytesOf(bq));
    b.setKindAttr(StringAttr::get(ctx, kind::BIAS));
  }
  return success();
}

// ---- activations -------------------------------------------------------------------------------

FailureOr<Value> Quantizer::ensureI8(Value v, Operation *user) {
  if (isI8Tensor(v))
    return v;
  if (auto it = quantCache.find(v); it != quantCache.end())
    return it->second;
  auto name = nameOf(v, user);
  if (failed(name))
    return failure();
  auto eX = expOf(v, user);
  if (failed(eX))
    return failure();
  auto a = quantAbsmax(user, *name, 8);
  if (failed(a))
    return failure();
  double sY = qp::i8Scale(*a);
  auto ms = check(qp::quantParams(*eX, sY), user);
  if (failed(ms))
    return failure();
  OpBuilder b(ctx);
  if (Operation *def = v.getDefiningOp())
    b.setInsertionPointAfter(def);
  else
    b.setInsertionPointToStart(&fn.getBody().front());
  auto q = b.create<QuantOp>(user->getLoc(), actType(ctx, tensorWidth(v), 8), v, ms->M, ms->S);
  setValueName(q.getOutput(), (*name + ".q").str());
  setValueScale(q.getOutput(), sY);
  quantCache[v] = q.getOutput();
  return q.getOutput();
}

LogicalResult Quantizer::visit(RmsNormOp op) {
  auto eX = expOf(op.getInput(), op);
  auto inName = nameOf(op.getInput(), op);
  auto name = nameOf(op.getOutput(), op);
  if (failed(eX) || failed(inName) || failed(name))
    return failure();
  WeightOp g = lookupWeight(op, op.getGammaAttr());
  if (!g)
    return op.emitError("unknown gamma symbol");
  auto eG = quantizeGamma(g, op);
  if (failed(eG))
    return failure();
  int64_t width = tensorWidth(op.getInput()), heads = op.getHeads();
  if (heads <= 0 || heads > 255 || width % heads || (width / heads) % 16)
    return op.emitError("RMSNorm heads must divide input width into groups divisible by 16");
  int64_t K = width / heads;
  if (g.getTensorType().getDimSize(0) != K)
    return op.emitError("gamma length does not match the input width");
  auto aY = quantAbsmax(op, *name, 16);
  auto aX = absmax(op, *inName);
  if (failed(aY) || failed(aX))
    return failure();
  int64_t eY = qp::i16Exponent(*aY);
  std::optional<double> rmsMin;
  if (auto it = calib.find((*inName + ".rms_min").str()); it != calib.end())
    rmsMin = it->second;
  auto p = check(qp::rmsnormParams(K, op.getEps().convertToDouble(), *eX, *eG, eY, *aX,
                                 name->str(), rmsMin), op);
  if (failed(p))
    return failure();
  auto i64 = [&](int64_t v) { return IntegerAttr::get(IntegerType::get(ctx, 64), v); };
  op.setEpsTAttr(i64(p->eps_t));
  op.setCAttr(i64(p->C));
  op.setShPostAttr(i64(p->sh_post));
  op.getOutput().setType(actType(ctx, width, 16));
  setValueExp(op.getOutput(), eY);
  return success();
}

LogicalResult Quantizer::visit(LinearOp op) {
  if (op.getEpilogue() != EpilogueMode::none || op.getAux() || op.getDest())
    return op.emitError("quantize expects unfused, untiled linears");
  auto in = ensureI8(op.getInput(), op);
  if (failed(in))
    return failure();
  op.getInputMutable().assign(*in);
  auto sA = scaleOf(*in, op);
  auto name = nameOf(op.getOutput(), op);
  if (failed(sA) || failed(name))
    return failure();
  Value res = op.getOutput();
  int64_t N = op.getN(), K = op.getK();
  bool isLast = res == returned;
  bool outI8 = !res.use_empty() && llvm::all_of(res.getUsers(), [&](Operation *u) {
    auto a = dyn_cast<AttentionOp>(u);
    return a && a.getV() == res;
  });
  auto a = quantAbsmax(op, *name, outI8 ? 8 : 16);
  if (failed(a))
    return failure();
  int64_t nPad = isLast ? mi.vocabPadded : N;
  if (!isLast && N % 16)
    return op.emitError("output width ") << N << " is not a multiple of 16";
  if (K % 16)
    return op.emitError("input width ") << K << " is not a multiple of 16";
  double sOut = 0.0;
  int64_t eOut = 0;
  if (outI8)
    sOut = qp::i8Scale(*a);
  else if (isLast)
    eOut = mi.eLogit;
  else {
    // RoPE preserves the tensor exponent, so reserve headroom for both
    // the projection and its rotated result before requantizing weights.
    double maximum = *a;
    for (Operation *user : res.getUsers()) {
      if (auto rope = dyn_cast<RopeOp>(user)) {
        auto rotatedName = nameOf(rope.getOutput(), rope);
        if (failed(rotatedName)) return failure();
        auto rotatedMaximum = quantAbsmax(rope, *rotatedName, 16);
        if (failed(rotatedMaximum)) return failure();
        maximum = std::max(maximum, *rotatedMaximum);
      }
    }
    eOut = qp::i16Exponent(maximum);
    // The ISA ADD right-shifts only its second operand. Reserve the immediate
    // sum's calibrated headroom in this branch's GEMM output, then ADD can put
    // the coarser operand first without degrading earlier residuals/embeddings.
    for (Operation *user : res.getUsers()) {
      if (auto add = dyn_cast<AddOp>(user)) {
        Value other = add.getA() == res ? add.getB() : add.getA();
        for (Value value : {other, Value(add.getOutput())}) {
          auto valueName = nameOf(value, add);
          if (failed(valueName)) return failure();
          auto valueMaximum = quantAbsmax(add, *valueName, 16);
          if (failed(valueMaximum)) return failure();
          eOut = std::max(eOut, qp::i16Exponent(*valueMaximum));
        }
        if (auto otherExp = getValueExp(other))
          eOut = std::max(eOut, *otherExp);
      }
    }
  }
  if (failed(quantizeLinearWeights(op, *sA, outI8, eOut, sOut, nPad)))
    return failure();
  res.setType(actType(ctx, nPad, outI8 ? 8 : 16));
  if (outI8)
    setValueScale(res, sOut);
  else
    setValueExp(res, eOut);
  return success();
}

LogicalResult Quantizer::visit(RopeOp op) {
  auto eX = expOf(op.getInput(), op);
  if (failed(eX))
    return failure();
  int64_t width = tensorWidth(op.getInput());
  if (op.getHeads() <= 0 || width % op.getHeads())
    return op.emitError("width is not a multiple of heads");
  if (failed(ensureRopeTables(width / op.getHeads(), op)))
    return failure();
  op.getOutput().setType(op.getInput().getType());
  setValueExp(op.getOutput(), *eX);  // RoPE preserves the exponent
  return success();
}

LogicalResult Quantizer::visit(AttentionOp op) {
  auto q = ensureI8(op.getQ(), op);
  auto k = ensureI8(op.getK(), op);
  auto v = ensureI8(op.getV(), op);
  if (failed(q) || failed(k) || failed(v))
    return failure();
  op.getQMutable().assign(*q);
  op.getKMutable().assign(*k);
  op.getVMutable().assign(*v);
  auto sQ = scaleOf(*q, op), sK = scaleOf(*k, op), sV = scaleOf(*v, op);
  auto name = nameOf(op.getOutput(), op);
  if (failed(sQ) || failed(sK) || failed(sV) || failed(name))
    return failure();
  int64_t H = op.getHeads(), Hkv = op.getKvHeads(), D = op.getHeadDim();
  if (H != mi.nHeads || Hkv != mi.nKvHeads || D != mi.headDim || Hkv <= 0 || H % Hkv)
    return op.emitError("attention heads/kv_heads/head_dim do not match the model");
  if (tensorWidth(*q) != H * D || tensorWidth(*k) != Hkv * D || tensorWidth(*v) != Hkv * D)
    return op.emitError("attention operand widths do not match heads*head_dim");
  auto a = quantAbsmax(op, *name, 8);
  if (failed(a))
    return failure();
  double sOut = qp::i8Scale(*a);
  auto p = check(qp::attnParams(*sQ, *sK, *sV, sOut, D), op);
  if (failed(p))
    return failure();
  auto i64 = [&](int64_t v) { return IntegerAttr::get(IntegerType::get(ctx, 64), v); };
  op.setMsAttr(i64(p->s.M));
  op.setSsAttr(i64(p->s.S));
  op.setMoAttr(i64(p->o.M));
  op.setSoAttr(i64(p->o.S));
  op.getOutput().setType(actType(ctx, H * D, 8));
  setValueScale(op.getOutput(), sOut);
  return success();
}

LogicalResult Quantizer::visit(SiluOp op) {
  auto eX = expOf(op.getInput(), op);
  auto name = nameOf(op.getOutput(), op);
  if (failed(eX) || failed(name))
    return failure();
  auto a = quantAbsmax(op, *name, 16);
  if (failed(a))
    return failure();
  int64_t eY = qp::i16Exponent(*a);
  auto p = check(qp::siluParams(*eX, eY, name->str()), op);
  if (failed(p))
    return failure();
  auto i64 = [&](int64_t v) { return IntegerAttr::get(IntegerType::get(ctx, 64), v); };
  op.setMiAttr(i64(p->Mi));
  op.setSiAttr(i64(p->Si));
  op.setShOutAttr(i64(p->sh_out));
  op.getOutput().setType(op.getInput().getType());
  setValueExp(op.getOutput(), eY);
  return success();
}

LogicalResult Quantizer::visit(MulOp op) {
  auto eA = expOf(op.getA(), op), eB = expOf(op.getB(), op);
  auto name = nameOf(op.getOutput(), op);
  if (failed(eA) || failed(eB) || failed(name))
    return failure();
  auto a = quantAbsmax(op, *name, 16);
  if (failed(a))
    return failure();
  int64_t eY = qp::i16Exponent(*a);
  auto sh = check(qp::mulShift(*eA, *eB, eY, name->str()), op);
  if (failed(sh))
    return failure();
  op.setShAttr(IntegerAttr::get(IntegerType::get(ctx, 64), *sh));
  op.getOutput().setType(op.getA().getType());
  setValueExp(op.getOutput(), eY);
  return success();
}

LogicalResult Quantizer::visit(AddOp op) {
  auto eA = expOf(op.getA(), op), eB = expOf(op.getB(), op);
  auto name = nameOf(op.getOutput(), op);
  if (failed(eA) || failed(eB) || failed(name))
    return failure();
  if (*eA < *eB) {
    Value a = op.getA(), b = op.getB();
    op.getAMutable().assign(b);
    op.getBMutable().assign(a);
    std::swap(*eA, *eB);
  }
  auto maximum = quantAbsmax(op, *name, 16);
  if (failed(maximum)) return failure();
  if (qp::i16Exponent(*maximum) > *eA)
    return op.emitError("ADD needs a producer with enough calibrated sum headroom");
  auto sh = check(qp::addShift(*eA, *eB, name->str()), op);
  if (failed(sh))
    return failure();
  op.setShBAttr(IntegerAttr::get(IntegerType::get(ctx, 64), *sh));
  op.getOutput().setType(op.getA().getType());
  setValueExp(op.getOutput(), *eA);
  return success();
}

struct QuantizePass : public mlir::llaccel::impl::LLAccelQuantizeBase<QuantizePass> {
  using LLAccelQuantizeBase::LLAccelQuantizeBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (calib.empty() || weights.empty() || weightsJson.empty()) {
      module.emitError("llaccel-quantize needs calib=, weights= and weights-json=");
      return signalPassFailure();
    }
    auto mi = ModelInfo::read(module);
    if (failed(mi))
      return signalPassFailure();
    if (mi->quantized) {
      module.emitError("module is already quantized");
      return signalPassFailure();
    }
    auto f32 = loadF32Weights(weights, weightsJson, module.getLoc());
    if (failed(f32))
      return signalPassFailure();
    auto cal = loadCalib(calib, module.getLoc());
    if (failed(cal))
      return signalPassFailure();
    Quantizer q(module, *f32, *cal, *mi);
    if (failed(q.run()))
      return signalPassFailure();
  }
};

} // namespace
