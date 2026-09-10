//===- LLAccelInterfaces.td - interfaces for llaccel.isa ops -*- tablegen -*-===//
#ifndef LLACCEL_INTERFACES_TD
#define LLACCEL_INTERFACES_TD

include "mlir/IR/OpBase.td"

def LLAccel_IsaOpInterface : OpInterface<"IsaOp"> {
  let cppNamespace = "::mlir::llaccel";
  let description = [{
    Implemented by every `llaccel.isa.*` op. Exposes the engine queue the
    instruction is issued to and the byte ranges it reads/writes, which the
    scheduler (dependency analysis), the SRAM allocator (liveness) and the
    emitter (instruction encoding) all consume.
  }];
  let methods = [
    InterfaceMethod<"Engine that executes this instruction.",
      "::mlir::llaccel::IsaEngine", "getEngine", (ins)>,
    InterfaceMethod<"Append every SRAM/DRAM byte range this instruction touches.",
      "void", "getAccesses", (ins "::llvm::SmallVectorImpl<::mlir::llaccel::BufAccess>&":$out)>,
    InterfaceMethod<"Semaphore this instruction waits on (or nullopt).",
      "std::optional<int64_t>", "getWaitSem", (ins), [{}], [{
        return $_op.getWaitSemAttr() ? std::optional<int64_t>($_op.getWaitSemAttr().getInt()) : std::nullopt;
      }]>,
    InterfaceMethod<"Value waited for.",
      "int64_t", "getWaitValue", (ins), [{}], [{
        return $_op.getWaitValAttr() ? $_op.getWaitValAttr().getInt() : 0;
      }]>,
    InterfaceMethod<"Semaphore signalled on completion (or nullopt).",
      "std::optional<int64_t>", "getSignalSem", (ins), [{}], [{
        return $_op.getSignalSemAttr() ? std::optional<int64_t>($_op.getSignalSemAttr().getInt()) : std::nullopt;
      }]>,
    InterfaceMethod<"Set/clear the wait annotation.",
      "void", "setWait", (ins "std::optional<int64_t>":$sem, "int64_t":$val), [{}], [{
        auto *ctx = $_op.getContext();
        if (sem) {
          $_op.setWaitSemAttr(::mlir::IntegerAttr::get(::mlir::IntegerType::get(ctx, 64), *sem));
          $_op.setWaitValAttr(::mlir::IntegerAttr::get(::mlir::IntegerType::get(ctx, 64), val));
        } else {
          $_op.removeWaitSemAttr();
          $_op.removeWaitValAttr();
        }
      }]>,
    InterfaceMethod<"Set/clear the signal annotation.",
      "void", "setSignal", (ins "std::optional<int64_t>":$sem), [{}], [{
        if (sem)
          $_op.setSignalSemAttr(::mlir::IntegerAttr::get(::mlir::IntegerType::get($_op.getContext(), 64), *sem));
        else
          $_op.removeSignalSemAttr();
      }]>,
  ];
}

#endif // LLACCEL_INTERFACES_TD
