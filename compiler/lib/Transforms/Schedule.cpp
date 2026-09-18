//===- Schedule.cpp - engine deps, prefetch hoisting, semaphores ----------===//
//
// Dependency model. Every instruction runs on one engine (dma/gemm/vec/attn);
// each engine executes its queue in order, engines run concurrently, and the
// command processor issues in program order, stalling on `wait_sem >= wait_val`.
// Two instructions on different engines conflict when their byte ranges
// (SRAM, resolved to absolute addresses after llaccel-alloc-sram, or DRAM
// symbol ranges) overlap and at least one writes (RAW / WAR / WAW). Same-engine
// ordering is implicit.
//
// Semaphore scheme (4 of the 32 counting semaphores): semaphore e counts the
// completed instructions of engine e -- every instruction signals its engine's
// semaphore. "Instruction i depends on instruction j (engine e, the k-th
// instruction issued to e)" is therefore exactly `wait sem[e] >= k+1`, and
// because the counts are monotonic a consumer waiting for the count that
// includes its producer also covers every earlier producer on that engine.
// Waits already enforced by an earlier instruction in issue order are implied
// (issue is in order) and dropped; when an instruction needs waits on several
// engines the extra ones ride on `isa.nop` instructions placed in front of it
// (a NOP's wait blocks issue, hence everything behind it).
//
// Modes:
//   inorder : instruction order = lowering order and every instruction waits
//             for the completion of its predecessor when that runs on another
//             engine -- the program executes as if on a single in-order engine
//             (the no-concurrency baseline).
//   overlap : minimal RAW/WAR/WAW semaphores; the weight-chunk DMA for GEMM i+1
//             is hoisted to just after its last dependency -- the WAR on the
//             rotating buffer's previous reader (GEMM i+1-nbuf) and the previous
//             DMA-engine instruction (DMA queue order is kept) -- so with
//             `weight-buffers` = 2 it is issued above GEMM i and the DMA engine
//             stays one chunk ahead of the GEMM engine.
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"

namespace mlir::llaccel {
#define GEN_PASS_DEF_LLACCELSCHEDULE
#include "llaccel/Transforms/Passes.h.inc"
} // namespace mlir::llaccel

using namespace mlir;
using namespace mlir::llaccel;

namespace {

struct Range {
  int64_t lo, hi;  // [lo, hi)
  bool write;
  StringRef dram;  // empty: SRAM
};

struct Node {
  Operation *op;
  IsaEngine eng;
  SmallVector<Range, 8> ranges;
  int64_t ordinal = -1;  // index among its engine's instructions (final order)
  bool weightDma = false;
};

bool conflicts(const Node &a, const Node &b) {
  for (const Range &x : a.ranges)
    for (const Range &y : b.ranges) {
      if (!(x.write || y.write) || x.dram != y.dram)
        continue;
      if (x.lo < y.hi && y.lo < x.hi)
        return true;
    }
  return false;
}

int semOf(IsaEngine e) {
  switch (e) {
  case IsaEngine::DMA: return 0;
  case IsaEngine::GEMM: return 1;
  case IsaEngine::VEC: return 2;
  case IsaEngine::ATTN: return 3;
  case IsaEngine::CP: return -1;
  }
  return -1;
}

struct SchedulePass : public mlir::llaccel::impl::LLAccelScheduleBase<SchedulePass> {
  using LLAccelScheduleBase::LLAccelScheduleBase;

  LogicalResult buildNodes(func::FuncOp fn, std::vector<Node> &nodes) {
    for (Operation &op : fn.getBody().front()) {
      auto isa = dyn_cast<IsaOp>(op);
      if (!isa)
        continue;
      Node n;
      n.op = &op;
      n.eng = isa.getEngine();
      SmallVector<BufAccess, 8> acc;
      isa.getAccesses(acc);
      for (const BufAccess &a : acc) {
        if (a.buf) {
          auto alloc = a.buf.getDefiningOp<AllocOp>();
          if (!alloc || !alloc.getAddr())
            return op.emitError("buffer has no SRAM address (run llaccel-alloc-sram first)");
          int64_t base = *alloc.getAddr() + a.offset;
          n.ranges.push_back({base, base + a.size, a.write, StringRef()});
        } else {
          n.ranges.push_back({a.offset, a.offset + a.size, a.write, a.dram.getValue()});
        }
      }
      if (auto dl = dyn_cast<DmaLoadOp>(op))
        if (auto alloc = dl.getDst().getDefiningOp<AllocOp>())
          n.weightDma = alloc.getRegion() == region::WEIGHT;
      nodes.push_back(std::move(n));
    }
    return success();
  }

  void schedule(func::FuncOp fn, std::vector<Node> &nodes, bool overlap) {
    // ---- ordering ----
    std::vector<Node *> order;
    order.reserve(nodes.size());
    for (Node &n : nodes) {
      if (overlap && n.weightDma) {
        size_t pos = 0;
        for (size_t k = 0; k < order.size(); ++k)
          if (order[k]->eng == IsaEngine::DMA || conflicts(n, *order[k]))
            pos = k + 1;
        order.insert(order.begin() + long(pos), &n);
      } else {
        order.push_back(&n);
      }
    }
    Operation *term = fn.getBody().front().getTerminator();
    for (Node *n : order)
      n->op->moveBefore(term);

    // ---- semaphores ----
    OpBuilder b(fn.getContext());
    auto i64 = [&](int64_t v) { return IntegerAttr::get(IntegerType::get(fn.getContext(), 64), v); };
    int64_t ord[4] = {0, 0, 0, 0}, known[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < order.size(); ++i) {
      Node &n = *order[i];
      int64_t need[4] = {0, 0, 0, 0};
      if (!overlap) {
        if (i > 0) {
          Node &p = *order[i - 1];
          int ps = semOf(p.eng);
          if (ps >= 0 && p.eng != n.eng)
            need[ps] = std::max(need[ps], p.ordinal + 1);
        }
      } else {
        for (size_t j = 0; j < i; ++j) {
          Node &p = *order[j];
          int ps = semOf(p.eng);
          if (ps < 0 || p.eng == n.eng || !conflicts(n, p))
            continue;
          need[ps] = std::max(need[ps], p.ordinal + 1);
        }
      }
      auto isa = cast<IsaOp>(n.op);
      bool primary = false;
      for (int e = 0; e < 4; ++e) {
        if (need[e] <= known[e])
          continue;
        known[e] = need[e];
        if (!primary) {
          isa.setWait(e, need[e]);
          primary = true;
        } else {
          b.setInsertionPoint(n.op);
          b.create<IsaNopOp>(n.op->getLoc(), i64(e), i64(need[e]), IntegerAttr());
        }
      }
      int s = semOf(n.eng);
      if (s >= 0) {
        isa.setSignal(s);
        n.ordinal = ord[s]++;
      }
    }
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (mode != "inorder" && mode != "overlap") {
      module.emitError("llaccel-schedule: mode must be inorder or overlap");
      return signalPassFailure();
    }
    for (auto fn : module.getOps<func::FuncOp>()) {
      if (!fn->hasAttr(kProgramMAttr))
        continue;
      std::vector<Node> nodes;
      if (failed(buildNodes(fn, nodes)))
        return signalPassFailure();
      schedule(fn, nodes, mode == "overlap");
    }
    module->setAttr(kScheduleAttr, StringAttr::get(&getContext(), mode));
  }
};

} // namespace
