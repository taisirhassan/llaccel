//===- Fuse.cpp - GEMM epilogue fusion (target llaccel-v2) ----------------===//
//
//   linear -> add(x, .)      =>  linear {epilogue = resadd, aux = x}
//   linear -> silu           =>  linear {epilogue = silu, silu_*}
//   linear(u), mul(sg, u)    =>  linear(u) {epilogue = mul, aux = sg, aux_shift = sh}
//
// A fusion is applied only when the numerics are bit-identical to the unfused
// form (NUMERICS.md GEMM epilogue): the linear must be unfused, produce i16,
// have this consumer as its only use, and for resadd every exponent involved
// must be equal (sh_b == 0). The fused linear's result carries the consumer's
// name/exponent; the requant table still targets the linear's own exponent
// (kept as `llaccel.pre_exp` for readability of --dump-mlir).
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::llaccel {
#define GEN_PASS_DEF_LLACCELFUSE
#include "llaccel/Transforms/Passes.h.inc"
} // namespace mlir::llaccel

using namespace mlir;
using namespace mlir::llaccel;

namespace {

constexpr llvm::StringLiteral kPreExpAttr = "llaccel.pre_exp";

/// A linear that can absorb an epilogue: unfused, i16 result, single use.
LinearOp fusibleProducer(Value v) {
  auto lin = v.getDefiningOp<LinearOp>();
  if (!lin || lin.getEpilogue() != EpilogueMode::none || lin.getAux() || lin.getDest() ||
      lin.getNOffset() || !isI16Tensor(lin.getOutput()) || !lin.getOutput().hasOneUse())
    return nullptr;
  return lin;
}

/// Rebuild `lin` with an epilogue; the result takes over `consumer`'s value.
LinearOp rebuild(PatternRewriter &rw, LinearOp lin, Value consumerResult, EpilogueMode mode,
                 Value aux, std::optional<int64_t> auxShift, std::optional<int64_t> mi,
                 std::optional<int64_t> si, std::optional<int64_t> shOut) {
  MLIRContext *ctx = rw.getContext();
  auto i64 = [&](std::optional<int64_t> v) -> IntegerAttr {
    return v ? IntegerAttr::get(IntegerType::get(ctx, 64), *v) : IntegerAttr();
  };
  auto fused = rw.create<LinearOp>(
      lin.getLoc(), consumerResult.getType(), lin.getInput(), aux, /*dest=*/Value(),
      lin.getWeightAttr(), lin.getBiasAttr(), lin.getRqAttr(), EpilogueModeAttr::get(ctx, mode),
      i64(auxShift), i64(mi), i64(si), i64(shOut), /*n_offset=*/IntegerAttr(),
      /*n_size=*/IntegerAttr());
  if (auto n = getValueName(consumerResult))
    setValueName(fused.getOutput(), *n);
  if (auto e = getValueExp(consumerResult))
    setValueExp(fused.getOutput(), e.value());
  if (auto e = getValueExp(lin.getOutput()))
    fused->setAttr(kPreExpAttr, IntegerAttr::get(IntegerType::get(ctx, 64), *e));
  return fused;
}

struct FuseResAdd : OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp add, PatternRewriter &rw) const override {
    if (add.getShB().value_or(0) != 0)
      return failure();
    LinearOp lin = fusibleProducer(add.getB());
    Value other = add.getA();
    if (!lin) {
      lin = fusibleProducer(add.getA());
      other = add.getB();
    }
    if (!lin)
      return failure();
    auto eLin = getValueExp(lin.getOutput()), eOther = getValueExp(other),
         eOut = getValueExp(add.getOutput());
    if (!eLin || !eOther || !eOut || *eLin != *eOther || *eLin != *eOut)
      return failure();
    if (other.getType() != lin.getOutput().getType())
      return failure();
    // aux must be available before the fused linear: `other` must dominate it.
    if (Operation *def = other.getDefiningOp(); def && !def->isBeforeInBlock(lin))
      return failure();
    rw.setInsertionPoint(add);
    LinearOp fused = rebuild(rw, lin, add.getOutput(), EpilogueMode::resadd, other, 0,
                             std::nullopt, std::nullopt, std::nullopt);
    rw.replaceOp(add, fused.getOutput());
    rw.eraseOp(lin);
    return success();
  }
};

struct FuseSilu : OpRewritePattern<SiluOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SiluOp silu, PatternRewriter &rw) const override {
    LinearOp lin = fusibleProducer(silu.getInput());
    if (!lin || !silu.getMi() || !silu.getSi() || !silu.getShOut())
      return failure();
    rw.setInsertionPoint(silu);
    LinearOp fused = rebuild(rw, lin, silu.getOutput(), EpilogueMode::silu, Value(), std::nullopt,
                             silu.getMi(), silu.getSi(), silu.getShOut());
    rw.replaceOp(silu, fused.getOutput());
    rw.eraseOp(lin);
    return success();
  }
};

struct FuseMul : OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp mul, PatternRewriter &rw) const override {
    if (!mul.getSh())
      return failure();
    // Prefer absorbing into the plain (non-silu) linear: mul(sg, u) fuses into u.
    LinearOp lin = fusibleProducer(mul.getB());
    Value other = mul.getA();
    if (!lin) {
      lin = fusibleProducer(mul.getA());
      other = mul.getB();
    }
    if (!lin)
      return failure();
    if (Operation *def = other.getDefiningOp(); def && !def->isBeforeInBlock(lin))
      return failure();
    rw.setInsertionPoint(mul);
    LinearOp fused = rebuild(rw, lin, mul.getOutput(), EpilogueMode::mul, other, mul.getSh(),
                             std::nullopt, std::nullopt, std::nullopt);
    rw.replaceOp(mul, fused.getOutput());
    rw.eraseOp(lin);
    return success();
  }
};

struct FusePass : public mlir::llaccel::impl::LLAccelFuseBase<FusePass> {
  using LLAccelFuseBase::LLAccelFuseBase;

  void runOnOperation() override {
    if (!enable)
      return;
    ModuleOp module = getOperation();
    func::FuncOp fn = getForwardFunc(module);
    if (!fn)
      return;
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseSilu, FuseMul, FuseResAdd>(&getContext());
    GreedyRewriteConfig cfg;
    cfg.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(fn, std::move(patterns), cfg)))
      return signalPassFailure();
  }
};

} // namespace
