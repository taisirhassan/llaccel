//===- AllocSram.cpp - liveness-interval first-fit SRAM allocation --------===//
//
// Per program (function with `llaccel.program_m`):
//   1. liveness: a buffer is live from the first to the last instruction that
//      names it (instruction index in block order); `kv`, `const` and `weight`
//      buffers are live for the whole program.
//   2. placement order: KV regions first (same order and sizes in every
//      program, so they land at the same addresses -- checked), then the
//      resident constants, the rotating weight buffers (256-B aligned, the
//      GEMM `w` operand requirement), then activations/staging blocks by
//      increasing start index, first-fit into the lowest gap not used by any
//      buffer with an overlapping live range.
//   3. bank rotation: the SRAM has 16 banks of 16 B (bank = addr[7:4]) with a
//      fixed-priority per-bank arbiter. Activation buffers are 32-B aligned;
//      inside the chosen gap an `act` buffer is placed at the first 32-B
//      address whose bank equals a rotating target (0, 2, 4, ... 14), so
//      consecutive activation buffers -- typically the operands of one VEC
//      instruction or the A/out of one GEMM -- start in different banks and
//      the streaming engines do not collide on the same bank every cycle.
//      When the rotated address does not fit the gap, the plain aligned
//      address is used.
// Reports the high-water mark and the peak of simultaneously live bytes as
// function attributes; fails with a clear message when a buffer does not fit.
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"

namespace mlir::llaccel {
#define GEN_PASS_DEF_LLACCELALLOCSRAM
#include "llaccel/Transforms/Passes.h.inc"
} // namespace mlir::llaccel

using namespace mlir;
using namespace mlir::llaccel;

namespace {

constexpr int64_t kInf = std::numeric_limits<int64_t>::max() / 4;

struct Buf {
  AllocOp op;
  int64_t size, align;
  StringRef region;
  int64_t start = kInf, end = -1;  // instruction index range
  int64_t addr = -1;
  bool wholeProgram() const { return region == region::KV || region == region::CONST ||
                                     region == region::WEIGHT; }
  bool overlapsLife(const Buf &o) const {
    int64_t s = wholeProgram() ? 0 : start, e = wholeProgram() ? kInf : end;
    int64_t os = o.wholeProgram() ? 0 : o.start, oe = o.wholeProgram() ? kInf : o.end;
    return s <= oe && os <= e;
  }
};

struct Allocator {
  int64_t sramBytes;
  std::vector<Buf *> placed;
  int64_t rot = 0;
  int64_t highWater = 0;

  bool fitInGap(Buf &b, int64_t gapStart, int64_t gapEnd) {
    int64_t a = roundUp(gapStart, b.align);
    if (b.region == region::ACT && b.align == 32) {
      int64_t bank = (a >> 4) & 15;
      int64_t r = a + (((rot - bank) & 15) << 4);
      if (r + b.size <= gapEnd) {
        b.addr = r;
        return true;
      }
    }
    if (a + b.size <= gapEnd) {
      b.addr = a;
      return true;
    }
    return false;
  }

  LogicalResult place(Buf &b) {
    std::vector<Buf *> conflicts;
    for (Buf *p : placed)
      if (p->overlapsLife(b))
        conflicts.push_back(p);
    llvm::sort(conflicts, [](Buf *x, Buf *y) { return x->addr < y->addr; });
    int64_t cursor = 0;
    bool ok = false;
    for (Buf *c : conflicts) {
      if (c->addr > cursor && fitInGap(b, cursor, c->addr)) {
        ok = true;
        break;
      }
      cursor = std::max(cursor, c->addr + c->size);
    }
    if (!ok)
      ok = fitInGap(b, cursor, sramBytes);
    if (!ok)
      return b.op.emitError("SRAM allocation failed: buffer `")
             << b.op.getName() << "` (" << b.size << " bytes, region " << b.region
             << ") does not fit in " << sramBytes << " bytes (high-water mark so far "
             << highWater << ")";
    if (b.region == region::ACT)
      rot = (rot + 2) & 15;
    placed.push_back(&b);
    highWater = std::max(highWater, b.addr + b.size);
    return success();
  }
};

struct AllocSramPass : public mlir::llaccel::impl::LLAccelAllocSramBase<AllocSramPass> {
  using LLAccelAllocSramBase::LLAccelAllocSramBase;

  LogicalResult allocFunction(func::FuncOp fn, llvm::StringMap<int64_t> &kvAddrs) {
    MLIRContext *ctx = fn.getContext();
    std::vector<Buf> bufs;
    DenseMap<Value, size_t> index;
    int64_t instrIdx = 0;
    for (Operation &op : fn.getBody().front()) {
      if (auto a = dyn_cast<AllocOp>(op)) {
        index[a.getBuf()] = bufs.size();
        bufs.push_back({a, int64_t(a.getSize()), int64_t(a.getAlign()), a.getRegion()});
        continue;
      }
      if (!isa<IsaOp>(op))
        continue;
      for (Value v : op.getOperands()) {
        auto it = index.find(v);
        if (it == index.end())
          return op.emitError("buffer operand is not an isa.alloc result");
        Buf &b = bufs[it->second];
        b.start = std::min(b.start, instrIdx);
        b.end = std::max(b.end, instrIdx);
      }
      ++instrIdx;
    }
    Allocator alloc{sramBytes};
    auto placeRegion = [&](StringRef reg) -> LogicalResult {
      for (Buf &b : bufs)
        if (b.region == reg && failed(alloc.place(b)))
          return failure();
      return success();
    };
    if (failed(placeRegion(region::KV)) || failed(placeRegion(region::CONST)) ||
        failed(placeRegion(region::WEIGHT)))
      return failure();
    std::vector<Buf *> acts;
    for (Buf &b : bufs)
      if (!b.wholeProgram())
        acts.push_back(&b);
    llvm::stable_sort(acts, [](Buf *x, Buf *y) { return x->start < y->start; });
    for (Buf *b : acts) {
      if (b->end < 0) {  // never referenced: give it no space
        b->addr = 0;
        b->size = 0;
        continue;
      }
      if (failed(alloc.place(*b)))
        return failure();
    }
    // Peak of simultaneously live bytes.
    int64_t livePeak = 0;
    for (int64_t i = 0; i < instrIdx; ++i) {
      int64_t live = 0;
      for (Buf &b : bufs)
        if (b.wholeProgram() || (b.start <= i && i <= b.end))
          live += b.size;
      livePeak = std::max(livePeak, live);
    }
    for (Buf &b : bufs) {
      b.op.setAddrAttr(IntegerAttr::get(IntegerType::get(ctx, 64), b.addr));
      if (b.region == region::KV) {
        auto it = kvAddrs.find(b.op.getName());
        if (it != kvAddrs.end() && it->second != b.addr)
          return b.op.emitError("KV region `")
                 << b.op.getName() << "` placed at " << b.addr << " here but at " << it->second
                 << " in another program";
        kvAddrs[b.op.getName()] = b.addr;
      }
    }
    fn->setAttr(kSramPeakAttr, IntegerAttr::get(IntegerType::get(ctx, 64), alloc.highWater));
    fn->setAttr(kSramLivePeakAttr, IntegerAttr::get(IntegerType::get(ctx, 64), livePeak));
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::StringMap<int64_t> kvAddrs;
    for (auto fn : module.getOps<func::FuncOp>()) {
      if (!fn->hasAttr(kProgramMAttr))
        continue;
      if (failed(allocFunction(fn, kvAddrs)))
        return signalPassFailure();
    }
  }
};

} // namespace
