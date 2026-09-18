//===- LowerToISA.cpp - high-level ops -> llaccel.isa.* programs ----------===//
//
// Produces two functions from `@forward`: `@prefill_m<M>` (M = prefill-m rows)
// and `@decode_m1`. Each is a straight-line list of `llaccel.isa.*` ops on
// `!llaccel.buf` values produced by `llaccel.isa.alloc`:
//
//   * KV cache: one zero-initialized DRAM K and V region per layer (Hkv * kv_stride
//     bytes, kv_stride = max_seq * D), identical in both programs.
//   * resident constants: only the POS-indexed RoPE table is
//     packed once (module level) into `@consts` (64-B aligned members) and
//     DMA'd into one `const` buffer at the start of every program. Gammas,
//     requant tables and biases stream into transient per-op/chunk buffers.
//   * weights are streamed per linear chunk into `weight-buffers` rotating
//     256-B aligned buffers (sized for the largest chunk).
//   * activations are `act` buffers, one per tensor value; the input rows are
//     DMA'd from `@dram_input`. Final linear chunks are DMA'd directly to
//     `@dram_logits` without a full vocabulary tensor allocation in SRAM.
//   * a linear split into N-chunks by llaccel-tile writes dense [M][n_size]
//     blocks (the GEMM output row stride is fixed to the instruction's N). For
//     M == 1 the blocks are contiguous slices of the row, so the chunks write
//     straight into the output buffer. For M > 1 the blocks are re-laid out
//     through `@dram_scratch` with strided DMAs (chunk block -> scratch columns,
//     then one gather load of the full [M][N] row-major tensor); an `aux`
//     operand of a chunked fused linear takes the reverse path (scatter the
//     row-major aux once, gather dense blocks per chunk).
//   * positions come only from the device POS register (RoPE table row and
//     KV slot are POS-relative in the ISA).
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"

namespace mlir::llaccel {
#define GEN_PASS_DEF_LLACCELLOWERTOISA
#include "llaccel/Transforms/Passes.h.inc"
} // namespace mlir::llaccel

using namespace mlir;
using namespace mlir::llaccel;

namespace {

struct ConstLayout {
  llvm::StringMap<int64_t> offset;
  int64_t total = 0;
};

/// Pack the POS-indexed rope_table into `@consts` (64-B aligned members).
ConstLayout packConstants(ModuleOp module) {
  ConstLayout cl;
  std::vector<char> bytes;
  for (auto w : module.getOps<WeightOp>()) {
    StringRef k = kindOf(w);
    if (k != kind::ROPE_TABLE)
      continue;
    bytes.resize(size_t(roundUp(int64_t(bytes.size()), 64)), 0);
    cl.offset[w.getSymName()] = int64_t(bytes.size());
    ArrayRef<char> raw = w.getRawData();
    bytes.insert(bytes.end(), raw.begin(), raw.end());
  }
  bytes.resize(size_t(roundUp(int64_t(bytes.size()), 64)), 0);
  if (bytes.empty())
    bytes.resize(64, 0);
  cl.total = int64_t(bytes.size());
  createWeight(module, sym::CONSTS, {cl.total}, 8, bytes, kind::CONSTS);
  return cl;
}

WeightOp createDramRegion(ModuleOp module, StringRef name, int64_t bytes, StringRef kindName) {
  OpBuilder b(module.getContext());
  Operation *last = nullptr;
  for (Operation &op : module.getBody()->getOperations())
    if (isa<WeightOp>(op)) last = &op;
  if (last)
    b.setInsertionPointAfter(last);
  else
    b.setInsertionPointToStart(module.getBody());
  auto type = RankedTensorType::get({bytes}, IntegerType::get(module.getContext(), 8));
  auto w = b.create<WeightOp>(module.getLoc(), b.getStringAttr(name), TypeAttr::get(type),
                              /*data=*/nullptr, b.getStringAttr(kindName), /*exp=*/nullptr,
                              /*orig_n=*/nullptr, b.getI64IntegerAttr(bytes));
  return w;
}

struct Chunk {
  LinearOp op;
  int64_t off, size;
};

class ProgramLowerer {
public:
  ProgramLowerer(ModuleOp module, const ModelInfo &mi, func::FuncOp src, int64_t M,
                 int64_t nWbuf, const ConstLayout &consts, int64_t inRowStride,
                 int64_t lgRowStride, int64_t scratchRowStride)
      : module(module), ctx(module.getContext()), mi(mi), src(src), M(M), nWbuf(nWbuf),
        consts(consts), inRowStride(inRowStride), lgRowStride(lgRowStride),
        scratchStride(scratchRowStride), b(module.getContext()), loc(src.getLoc()) {}

  LogicalResult run(StringRef fnName);
  bool usedScratch() const { return scratchUsed; }

private:
  IntegerAttr i64(int64_t v) { return IntegerAttr::get(IntegerType::get(ctx, 64), v); }
  FlatSymbolRefAttr symRef(StringRef s) { return FlatSymbolRefAttr::get(ctx, s); }
  Value alloc(int64_t size, int64_t align, const Twine &name, StringRef reg) {
    return b.create<AllocOp>(loc, BufType::get(ctx), i64(size), i64(align),
                             StringAttr::get(ctx, name.str()), StringAttr::get(ctx, reg),
                             IntegerAttr());
  }
  std::string nameOf(Value v) { return getValueName(v).value_or("t").str(); }
  FailureOr<Value> bufOf(Value v, Operation *at) {
    auto it = bufs.find(v);
    if (it == bufs.end())
      return at->emitError("operand has not been lowered (use before def?)");
    return it->second;
  }
  FailureOr<int64_t> constOff(StringRef symName, Operation *at) {
    auto it = consts.offset.find(symName);
    if (it == consts.offset.end())
      return at->emitError("constant `") << symName << "` is not a resident constant";
    return it->second;
  }
  void dmaLoad(Value dst, int64_t dstOff, StringRef sym, int64_t srcOff, int64_t rows,
               int64_t rowBytes, int64_t srcStride, int64_t dstStride) {
    b.create<DmaLoadOp>(loc, dst, i64(dstOff), symRef(sym), i64(srcOff), i64(rows), i64(rowBytes),
                        i64(srcStride), i64(dstStride), IntegerAttr(), IntegerAttr(),
                        IntegerAttr());
  }
  void dmaStore(Value srcBuf, int64_t srcOff, StringRef sym, int64_t dstOff, int64_t rows,
                int64_t rowBytes, int64_t srcStride, int64_t dstStride) {
    b.create<DmaStoreOp>(loc, srcBuf, i64(srcOff), symRef(sym), i64(dstOff), i64(rows),
                         i64(rowBytes), i64(srcStride), i64(dstStride), IntegerAttr(),
                         IntegerAttr(), IntegerAttr());
  }

  LogicalResult lower(RmsNormOp op);
  LogicalResult lower(QuantOp op);
  LogicalResult lower(RopeOp op);
  LogicalResult lower(AttentionOp op);
  LogicalResult lower(SiluOp op);
  LogicalResult lower(MulOp op);
  LogicalResult lower(AddOp op);
  LogicalResult lowerLinearChain(LinearOp first);
  LogicalResult emitGemm(const Chunk &c, Value a, Value out, int64_t outOff, Value aux,
                         int64_t auxOff, int64_t K, bool outI8);

  ModuleOp module;
  MLIRContext *ctx;
  const ModelInfo &mi;
  func::FuncOp src;
  int64_t M, nWbuf;
  const ConstLayout &consts;
  int64_t inRowStride, lgRowStride, scratchStride;
  OpBuilder b;
  Location loc;
  func::FuncOp fn;
  Value constBuf;
  SmallVector<std::string> kvK, kvV;
  SmallVector<Value> wbufs;
  int64_t wbufSize = 0, wbufNext = 0;
  DenseMap<Value, Value> bufs;
  DenseSet<Operation *> handled;
  bool scratchUsed = false;
  DenseSet<Value> storedLogits;
};

LogicalResult ProgramLowerer::run(StringRef fnName) {
  fn = func::FuncOp::create(loc, fnName, FunctionType::get(ctx, {}, {}));
  fn->setAttr(kProgramMAttr, i64(M));
  Block *body = fn.addEntryBlock();
  module.push_back(fn);
  b.setInsertionPointToEnd(body);

  // ---- regions that exist in every program ----

  for (int64_t l = 0; l < mi.nLayers; ++l) {
    kvK.push_back(("kv" + Twine(l) + ".k").str());
    kvV.push_back(("kv" + Twine(l) + ".v").str());
  }
  constBuf = alloc(consts.total, 64, "consts", region::CONST);
  for (auto lin : src.getBody().front().getOps<LinearOp>())
    wbufSize = std::max<int64_t>(wbufSize, lin.getNSize().value_or(lin.getN()) * lin.getK());
  wbufSize = roundUp(std::max<int64_t>(wbufSize, 256), 256);
  for (int64_t i = 0; i < nWbuf; ++i)
    wbufs.push_back(alloc(wbufSize, 256, "wbuf" + Twine(i), region::WEIGHT));

  dmaLoad(constBuf, 0, sym::CONSTS, 0, 1, consts.total, consts.total, consts.total);

  // ---- input rows ----
  BlockArgument arg = src.getArgument(0);
  int64_t inBytes = mi.dim * 2;
  Value x0 = alloc(M * inBytes, 32, nameOf(arg), region::ACT);
  dmaLoad(x0, 0, sym::DRAM_INPUT, 0, M, inBytes, inRowStride, inBytes);
  bufs[arg] = x0;

  // ---- ops ----
  for (Operation &op : src.getBody().front()) {
    if (handled.contains(&op))
      continue;
    LogicalResult r =
        llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            .Case<RmsNormOp, QuantOp, RopeOp, AttentionOp, SiluOp, MulOp, AddOp>(
                [&](auto o) { return lower(o); })
            .Case<LinearOp>([&](LinearOp o) { return lowerLinearChain(o); })
            .Case<func::ReturnOp>([&](func::ReturnOp ret) -> LogicalResult {
              if (storedLogits.contains(ret.getOperand(0)))
                return success();
              auto lg = bufOf(ret.getOperand(0), ret);
              if (failed(lg))
                return failure();
              int64_t rowBytes = tensorWidth(ret.getOperand(0)) * 2;
              dmaStore(*lg, 0, sym::DRAM_LOGITS, 0, M, rowBytes, rowBytes, lgRowStride);
              return success();
            })
            .Default([&](Operation *o) { return o->emitError("cannot lower op"); });
    if (failed(r))
      return failure();
  }
  b.create<IsaHaltOp>(loc, IntegerAttr(), IntegerAttr(), IntegerAttr());
  b.create<func::ReturnOp>(loc);
  return success();
}

LogicalResult ProgramLowerer::lower(RmsNormOp op) {
  auto srcBuf = bufOf(op.getInput(), op);
  if (failed(srcBuf))
    return failure();
  int64_t heads = op.getHeads(), K = tensorWidth(op.getInput()) / heads;
  Value gamma = alloc(K * 2, 64, nameOf(op.getOutput()) + ".gamma", region::TMP);
  dmaLoad(gamma, 0, op.getGamma(), 0, 1, K * 2, K * 2, K * 2);
  Value dst = alloc(M * heads * K * 2, 32, nameOf(op.getOutput()), region::ACT);
  for (int64_t row = 0; row < M * heads; row += ::llaccel::kGemmTM) {
    int64_t rows = std::min<int64_t>(::llaccel::kGemmTM, M * heads - row);
    b.create<IsaRmsNormOp>(loc, *srcBuf, i64(row * K * 2), gamma, i64(0), dst,
                         i64(row * K * 2), i64(rows), i64(K),
                         i64(op.getEpsT().value_or(0)), i64(op.getC().value_or(0)),
                         i64(op.getShPost().value_or(0)), IntegerAttr(), IntegerAttr(),
                         IntegerAttr());
  }
  bufs[op.getOutput()] = dst;
  return success();
}

LogicalResult ProgramLowerer::lower(QuantOp op) {
  auto srcBuf = bufOf(op.getInput(), op);
  if (failed(srcBuf))
    return failure();
  int64_t N = tensorWidth(op.getInput());
  Value dst = alloc(M * N, 32, nameOf(op.getOutput()), region::ACT);
  b.create<IsaQuantOp>(loc, *srcBuf, i64(0), dst, i64(0), i64(M * N), i64(int64_t(op.getM())),
                       i64(int64_t(op.getS())), IntegerAttr(), IntegerAttr(), IntegerAttr());
  bufs[op.getOutput()] = dst;
  return success();
}

LogicalResult ProgramLowerer::lower(RopeOp op) {
  auto srcBuf = bufOf(op.getInput(), op);
  auto tOff = constOff(sym::ROPE_TABLE, op);
  if (failed(srcBuf) || failed(tOff))
    return failure();
  int64_t W = tensorWidth(op.getInput()), H = op.getHeads(), D = W / H;
  Value dst = alloc(M * W * 2, 32, nameOf(op.getOutput()), region::ACT);
  b.create<IsaRopeOp>(loc, *srcBuf, i64(0), dst, i64(0), constBuf, i64(*tOff), i64(M), i64(H),
                      i64(D), i64(2 * D), i64(mi.maxSeq), IntegerAttr(), IntegerAttr(),
                      IntegerAttr());
  bufs[op.getOutput()] = dst;
  return success();
}

LogicalResult ProgramLowerer::lower(AttentionOp op) {
  auto q = bufOf(op.getQ(), op), k = bufOf(op.getK(), op), v = bufOf(op.getV(), op);
  if (failed(q) || failed(k) || failed(v))
    return failure();
  int64_t l = op.getLayer(), H = op.getHeads(), Hkv = op.getKvHeads(), D = op.getHeadDim();
  if (l < 0 || l >= mi.nLayers)
    return op.emitError("layer index out of range");
  int64_t kvStride = mi.maxSeq * D;
  b.create<IsaKvWriteOp>(loc, *k, i64(0), symRef(kvK[l]), i64(0), i64(M), i64(Hkv), i64(D), i64(kvStride),
                         IntegerAttr(), IntegerAttr(), IntegerAttr());
  b.create<IsaKvWriteOp>(loc, *v, i64(0), symRef(kvV[l]), i64(0), i64(M), i64(Hkv), i64(D), i64(kvStride),
                         IntegerAttr(), IntegerAttr(), IntegerAttr());
  Value out = alloc(M * H * D, 32, nameOf(op.getOutput()), region::ACT);
  b.create<IsaAttnOp>(loc, *q, i64(0), out, i64(0), symRef(kvK[l]), i64(0), symRef(kvV[l]), i64(0), i64(M), i64(H),
                      i64(Hkv), i64(D), i64(kvStride), i64(op.getMs().value_or(0)),
                      i64(op.getSs().value_or(0)), i64(op.getMo().value_or(0)),
                      i64(op.getSo().value_or(0)), IntegerAttr(), IntegerAttr(), IntegerAttr());
  bufs[op.getOutput()] = out;
  return success();
}

LogicalResult ProgramLowerer::lower(SiluOp op) {
  auto srcBuf = bufOf(op.getInput(), op);
  if (failed(srcBuf))
    return failure();
  int64_t N = tensorWidth(op.getInput());
  Value dst = alloc(M * N * 2, 32, nameOf(op.getOutput()), region::ACT);
  b.create<IsaSiluOp>(loc, *srcBuf, i64(0), dst, i64(0), i64(M * N), i64(op.getMi().value_or(0)),
                      i64(op.getSi().value_or(0)), i64(op.getShOut().value_or(0)), IntegerAttr(),
                      IntegerAttr(), IntegerAttr());
  bufs[op.getOutput()] = dst;
  return success();
}

LogicalResult ProgramLowerer::lower(MulOp op) {
  auto a = bufOf(op.getA(), op), bb = bufOf(op.getB(), op);
  if (failed(a) || failed(bb))
    return failure();
  int64_t N = tensorWidth(op.getA());
  Value dst = alloc(M * N * 2, 32, nameOf(op.getOutput()), region::ACT);
  b.create<IsaMulOp>(loc, *a, i64(0), *bb, i64(0), dst, i64(0), i64(M * N),
                     i64(op.getSh().value_or(0)), IntegerAttr(), IntegerAttr(), IntegerAttr());
  bufs[op.getOutput()] = dst;
  return success();
}

LogicalResult ProgramLowerer::lower(AddOp op) {
  auto a = bufOf(op.getA(), op), bb = bufOf(op.getB(), op);
  if (failed(a) || failed(bb))
    return failure();
  int64_t N = tensorWidth(op.getA());
  Value dst = alloc(M * N * 2, 32, nameOf(op.getOutput()), region::ACT);
  b.create<IsaAddOp>(loc, *a, i64(0), *bb, i64(0), dst, i64(0), i64(M * N),
                     i64(op.getShB().value_or(0)), IntegerAttr(), IntegerAttr(), IntegerAttr());
  bufs[op.getOutput()] = dst;
  return success();
}

LogicalResult ProgramLowerer::emitGemm(const Chunk &c, Value a, Value out, int64_t outOff,
                                       Value aux, int64_t auxOff, int64_t K, bool outI8) {
  LinearOp op = c.op;
  Value wbuf = wbufs[size_t(wbufNext++ % nWbuf)];
  int64_t wBytes = c.size * K;
  dmaLoad(wbuf, 0, op.getWeight(), c.off * K, 1, wBytes, wBytes, wBytes);
  if (!op.getRq())
    return op.emitError("linear is missing requantization parameters");
  // Per-chunk metadata is transient. Keeping whole-model RQ/bias tables resident
  // would consume more SRAM than the complete device for common vocabularies.
  Value rq = alloc(c.size * 8, 64, nameOf(op.getOutput()) + ".rq", region::TMP);
  dmaLoad(rq, 0, *op.getRq(), c.off * 8, 1, c.size * 8, c.size * 8, c.size * 8);
  Value bias;
  IntegerAttr biasOff;
  if (op.getBias()) {
    bias = alloc(c.size * 4, 64, nameOf(op.getOutput()) + ".bias", region::TMP);
    dmaLoad(bias, 0, *op.getBias(), c.off * 4, 1, c.size * 4, c.size * 4, c.size * 4);
    biasOff = i64(0);
  }
  b.create<GemmOp>(loc, a, i64(0), wbuf, i64(0), out, i64(outOff), rq,
                   i64(0), bias, biasOff, aux, aux ? i64(auxOff) : IntegerAttr(),
                   i64(M), i64(c.size), i64(K), op.getEpilogueAttr(), BoolAttr::get(ctx, outI8),
                   i64(op.getAuxShift().value_or(0)), i64(op.getSiluMi().value_or(0)),
                   i64(op.getSiluSi().value_or(0)), i64(op.getSiluShOut().value_or(0)),
                   IntegerAttr(), IntegerAttr(), IntegerAttr());
  return success();
}

LogicalResult ProgramLowerer::lowerLinearChain(LinearOp first) {
  if (first.getDest())
    return first.emitError("chunk chain does not start at its first chunk");
  // Collect the chain first -> ... -> last (dest links).
  SmallVector<Chunk> chain;
  LinearOp cur = first;
  while (true) {
    int64_t off = cur.getNOffset().value_or(0), size = cur.getNSize().value_or(cur.getN());
    chain.push_back({cur, off, size});
    handled.insert(cur);
    LinearOp next;
    for (Operation *u : cur.getOutput().getUsers())
      if (auto l = dyn_cast<LinearOp>(u); l && l.getDest() == cur.getOutput())
        next = l;
    if (!next)
      break;
    if (!cur.getOutput().hasOneUse())
      return cur.emitError("intermediate chunk result has uses other than the next chunk");
    cur = next;
  }
  LinearOp last = chain.back().op;
  Value result = last.getOutput();
  int64_t N = last.getN(), K = last.getK();
  bool outI8 = isI8Tensor(result);
  int64_t elem = outI8 ? 1 : 2;
  auto aBuf = bufOf(first.getInput(), first);
  if (failed(aBuf))
    return failure();
  Value auxBuf;
  if (first.getAux()) {
    auto ab = bufOf(first.getAux(), first);
    if (failed(ab))
      return failure();
    auxBuf = *ab;
    if (tensorWidth(first.getAux()) != N)
      return first.emitError("aux width does not match N");
  }
  std::string name = nameOf(result);
  // A returned linear is the final projection. Stream its dense chunk blocks
  // directly to DRAM columns; even one 16-row vocabulary tensor may exceed SRAM.
  bool finalLogits = result.hasOneUse() &&
                     isa<func::ReturnOp>(*result.getUsers().begin());
  if (finalLogits && !outI8 && !auxBuf) {
    for (const Chunk &c : chain) {
      Value blk = alloc(M * c.size * elem, 32, name + ".logits.c" + Twine(c.off),
                        region::TMP);
      if (failed(emitGemm(c, *aBuf, blk, 0, Value(), 0, K, outI8)))
        return failure();
      dmaStore(blk, 0, sym::DRAM_LOGITS, c.off * elem, M, c.size * elem,
               c.size * elem, lgRowStride);
    }
    storedLogits.insert(result);
    return success();
  }
  Value out = alloc(M * N * elem, 32, name, region::ACT);

  if (chain.size() == 1) {
    if (failed(emitGemm(chain[0], *aBuf, out, 0, auxBuf, 0, K, outI8)))
      return failure();
  } else if (M == 1) {
    // Dense [1][n_size] blocks are contiguous slices of the single output row.
    for (const Chunk &c : chain)
      if (failed(emitGemm(c, *aBuf, out, c.off * elem, auxBuf, c.off * 2, K, outI8)))
        return failure();
  } else {
    // Re-layout through DRAM scratch (see file comment).
    scratchUsed = true;
    int64_t S = scratchStride, auxArea = 16 * S;
    if (auxBuf)
      dmaStore(auxBuf, 0, sym::DRAM_SCRATCH, auxArea, M, N * 2, N * 2, S);
    for (const Chunk &c : chain) {
      Value auxBlk;
      if (auxBuf) {
        auxBlk = alloc(M * c.size * 2, 32, name + ".aux.c" + Twine(c.off / c.size), region::TMP);
        dmaLoad(auxBlk, 0, sym::DRAM_SCRATCH, auxArea + c.off * 2, M, c.size * 2, S, c.size * 2);
      }
      Value blk = alloc(M * c.size * elem, 32, name + ".c" + Twine(c.off / c.size), region::TMP);
      if (failed(emitGemm(c, *aBuf, blk, 0, auxBlk, 0, K, outI8)))
        return failure();
      dmaStore(blk, 0, sym::DRAM_SCRATCH, c.off * elem, M, c.size * elem, c.size * elem, S);
    }
    dmaLoad(out, 0, sym::DRAM_SCRATCH, 0, M, N * elem, S, N * elem);
  }
  bufs[result] = out;
  return success();
}

struct LowerToISAPass : public mlir::llaccel::impl::LLAccelLowerToISABase<LowerToISAPass> {
  using LLAccelLowerToISABase::LLAccelLowerToISABase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto mi = ModelInfo::read(module);
    if (failed(mi))
      return signalPassFailure();
    if (!mi->quantized) {
      module.emitError("llaccel-lower-to-isa needs a quantized module");
      return signalPassFailure();
    }
    if (prefillM < 1 || prefillM > 16 || weightBuffers < 1) {
      module.emitError("prefill-m must be in [1,16] and weight-buffers >= 1");
      return signalPassFailure();
    }
    if (mi->maxSeq % prefillM != 0) {
      module.emitError("max_seq must be divisible by prefill-m so padded launches fit KV/RoPE storage");
      return signalPassFailure();
    }
    if (mi->maxSeq > int64_t(::llaccel::kAttnTMax)) {
      module.emitError("max_seq exceeds ATTN_TMAX");
      return signalPassFailure();
    }
    func::FuncOp fwd = getForwardFunc(module);
    if (!fwd) {
      module.emitError("no high-level function to lower");
      return signalPassFailure();
    }
    for (int64_t l = 0; l < mi->nLayers; ++l) {
      const int64_t bytes = mi->nKvHeads * mi->maxSeq * mi->headDim;
      createDramRegion(module, ("kv" + Twine(l) + ".k").str(), bytes, kind::DRAM_KV);
      createDramRegion(module, ("kv" + Twine(l) + ".v").str(), bytes, kind::DRAM_KV);
    }
    ConstLayout consts = packConstants(module);
    int64_t inRowStride = roundUp(mi->dim * 2, 64);
    int64_t lgRowStride = roundUp(mi->vocabPadded * 2, 64);
    createDramRegion(module, sym::DRAM_INPUT, 16 * inRowStride, kind::DRAM_INPUT);
    createDramRegion(module, sym::DRAM_LOGITS, 16 * lgRowStride, kind::DRAM_LOGITS);
    // Scratch row stride: widest tensor that may be re-laid out (chunked outputs and their aux).
    int64_t scratchStride = 64;
    for (auto lin : fwd.getBody().front().getOps<LinearOp>())
      if (lin.getNOffset()) {
        int64_t elem = isI8Tensor(lin.getOutput()) ? 1 : 2;
        scratchStride = std::max(scratchStride, roundUp(lin.getN() * elem, 64));
        if (lin.getAux())
          scratchStride = std::max(scratchStride, roundUp(lin.getN() * 2, 64));
      }
    bool scratch = false;
    for (int64_t M : {int64_t(prefillM), int64_t(1)}) {
      std::string name = M == 1 ? "decode_m1" : ("prefill_m" + std::to_string(M));
      if (M == 1 && prefillM == 1)
        name = "prefill_m1";
      ProgramLowerer pl(module, *mi, fwd, M, weightBuffers, consts, inRowStride, lgRowStride,
                        scratchStride);
      if (failed(pl.run(name)))
        return signalPassFailure();
      scratch |= pl.usedScratch();
      if (prefillM == 1)
        break;
    }
    if (scratch)
      createDramRegion(module, sym::DRAM_SCRATCH, 2 * 16 * scratchStride, kind::DRAM_SCRATCH);
    fwd.erase();
  }
};

} // namespace
