//===- Tile.cpp - pad to 16 and split linears into weight chunks ----------===//
//
// * Every linear's N is padded to a multiple of 16 (zero weight rows, {M,S}
//   entries for scale 1, zero bias). Padding changes the tensor width, so it
//   is only legal for a linear whose result is the function result (the
//   lm_head, normally already padded to vocab_padded by llaccel-quantize);
//   any other N % 16 != 0 is an error, as is K % 16 != 0 (K padding would
//   require padding the producing activation).
// * A linear whose tiled weight (N*K bytes, ISA.md layout) exceeds
//   `weight-chunk-bytes` is split along N into chunks of `n_size` columns:
//     %c0 = linear %x, @w {n_offset = 0,   n_size = Nc}
//     %c1 = linear %x, @w into(%c0) {n_offset = Nc, n_size = Nc}   ... (last chunk keeps the name)
//   Chunk j's tiled weight bytes are the contiguous range [n_offset*K, (n_offset+n_size)*K)
//   of the full tiled weight (tiles are nt-major), so the weight symbol is
//   shared and the DMA just offsets into it. Chunk widths are chosen so the
//   chunk's output rows (n_size * elem bytes) are DMA-friendly for the prefill
//   re-layout (docs: README "chunking"): 16 columns, or a multiple of 64 bytes.
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"
#include "llaccel/Support/QuantParams.h"

namespace mlir::llaccel {
#define GEN_PASS_DEF_LLACCELTILE
#include "llaccel/Transforms/Passes.h.inc"
} // namespace mlir::llaccel

using namespace mlir;
using namespace mlir::llaccel;

namespace {

/// Largest legal chunk width <= maxCols (multiple of 16; rows < 64 B or a multiple of 64 B).
int64_t chunkWidth(int64_t maxCols, int64_t elemBytes) {
  int64_t best = 0;
  for (int64_t nc = 16; nc <= maxCols; nc += 16) {
    int64_t rowBytes = nc * elemBytes;
    if (rowBytes < 64 || rowBytes % 64 == 0)
      best = nc;
  }
  return best;
}

LogicalResult padLinear(LinearOp op, int64_t nPad, Value returned) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  MLIRContext *ctx = op.getContext();
  int64_t N = op.getN(), K = op.getK();
  if (op.getOutput() != returned)
    return op.emitError("output width ") << N << " is not a multiple of 16 and the linear is "
                                            "not the function result (cannot pad)";
  WeightOp w = lookupWeight(op, op.getWeightAttr());
  WeightOp rq = lookupWeight(op, op.getRqAttr());
  if (!w || !rq)
    return op.emitError("weight/rq symbol not found");
  {
    auto src = weightDataAs<int8_t>(w);
    std::vector<int8_t> q(size_t(nPad * K), 0);
    std::copy(src.begin(), src.end(), q.begin());
    setWeightData(w, {nPad, K}, 8,
                  ArrayRef<char>(reinterpret_cast<const char *>(q.data()), q.size()));
    if (!w.getOrigN())
      w.setOrigNAttr(IntegerAttr::get(IntegerType::get(ctx, 64), N));
  }
  {
    auto src = weightDataAs<int32_t>(rq);
    std::vector<int32_t> q(size_t(nPad * 2));
    std::copy(src.begin(), src.end(), q.begin());
    // Padded channels: zero weight rows have scale 1 (NUMERICS.md), so their
    // entry is s_a * 1 / denom exactly like llaccel-quantize would compute it.
    auto sA = getValueScale(op.getInput());
    if (!sA)
      return op.emitError("input has no llaccel.scale");
    bool outI8 = isI8Tensor(op.getOutput());
    auto ms = ::llaccel::qp::gemmRq(*sA, 1.0, outI8, getValueExp(op.getOutput()).value_or(0),
                                    getValueScale(op.getOutput()).value_or(1.0));
    if (!ms)
      return op.emitError(ms.error());
    for (int64_t n = N; n < nPad; ++n) {
      q[size_t(2 * n)] = int32_t(ms->M);
      q[size_t(2 * n + 1)] = int32_t(ms->S);
    }
    setWeightData(rq, {nPad, 2}, 32,
                  ArrayRef<char>(reinterpret_cast<const char *>(q.data()), q.size() * 4));
  }
  if (op.getBiasAttr()) {
    WeightOp b = lookupWeight(op, op.getBiasAttr());
    if (!b)
      return op.emitError("bias symbol not found");
    auto src = weightDataAs<int32_t>(b);
    std::vector<int32_t> q(size_t(nPad), 0);
    std::copy(src.begin(), src.end(), q.begin());
    setWeightData(b, {nPad}, 32,
                  ArrayRef<char>(reinterpret_cast<const char *>(q.data()), q.size() * 4));
  }
  auto newType = actType(ctx, nPad, isI8Tensor(op.getOutput()) ? 8 : 16);
  op.getOutput().setType(newType);
  auto fn = op->getParentOfType<func::FuncOp>();
  fn.setType(FunctionType::get(ctx, fn.getArgumentTypes(), {newType}));
  (void)module;
  return success();
}

struct TilePass : public mlir::llaccel::impl::LLAccelTileBase<TilePass> {
  using LLAccelTileBase::LLAccelTileBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    func::FuncOp fn = getForwardFunc(module);
    if (!fn)
      return;
    if (weightChunkBytes < 256 || weightChunkBytes % 256) {
      module.emitError("weight-chunk-bytes must be a positive multiple of 256");
      return signalPassFailure();
    }
    Value returned = fn.getBody().front().getTerminator()->getOperand(0);
    SmallVector<LinearOp> linears(fn.getBody().front().getOps<LinearOp>());
    for (LinearOp op : linears) {
      if (op.getNOffset() || op.getDest()) {
        op.emitError("linear is already tiled");
        return signalPassFailure();
      }
      int64_t K = op.getK();
      if (K % 16) {
        op.emitError("input width ") << K << " is not a multiple of 16";
        return signalPassFailure();
      }
      int64_t N = op.getN();
      int64_t nPad = roundUp(N, 16);
      if (nPad != N && failed(padLinear(op, nPad, returned)))
        return signalPassFailure();
      N = nPad;
      if (N * K <= weightChunkBytes)
        continue;
      int64_t elem = isI8Tensor(op.getOutput()) ? 1 : 2;
      int64_t nc = chunkWidth(weightChunkBytes / K, elem);
      if (nc == 0) {
        op.emitError("weight-chunk-bytes = ") << int64_t(weightChunkBytes) << " is too small for K = " << K
                                              << " (one 16-column chunk needs " << 16 * K
                                              << " bytes)";
        return signalPassFailure();
      }
      // Emit the chain in place of `op`.
      OpBuilder b(op);
      auto i64 = [&](int64_t v) { return IntegerAttr::get(IntegerType::get(ctx, 64), v); };
      auto name = getValueName(op.getOutput());
      auto exp = getValueExp(op.getOutput());
      auto scale = getValueScale(op.getOutput());
      Value prev;
      int64_t nChunks = (N + nc - 1) / nc, idx = 0;
      for (int64_t off = 0; off < N; off += nc, ++idx) {
        int64_t size = std::min(nc, N - off);
        auto chunk = b.create<LinearOp>(
            op.getLoc(), op.getOutput().getType(), op.getInput(), op.getAux(), prev,
            op.getWeightAttr(), op.getBiasAttr(), op.getRqAttr(), op.getEpilogueAttr(),
            op.getAuxShiftAttr(), op.getSiluMiAttr(), op.getSiluSiAttr(), op.getSiluShOutAttr(),
            i64(off), i64(size));
        for (NamedAttribute a : op->getDiscardableAttrs())
          chunk->setAttr(a.getName(), a.getValue());
        if (idx + 1 < nChunks && name)
          setValueName(chunk.getOutput(), (*name + ".c" + Twine(idx)).str());
        if (exp)
          setValueExp(chunk.getOutput(), *exp);
        if (scale)
          setValueScale(chunk.getOutput(), *scale);
        prev = chunk.getOutput();
      }
      op.getOutput().replaceAllUsesWith(prev);
      op.erase();
    }
  }
};

} // namespace
