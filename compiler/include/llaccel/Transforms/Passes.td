//===- Passes.td - llaccel compiler passes ---------------------*- tablegen -*-===//
#ifndef LLACCEL_PASSES_TD
#define LLACCEL_PASSES_TD

include "mlir/Pass/PassBase.td"

def LLAccelQuantize : Pass<"llaccel-quantize", "::mlir::ModuleOp"> {
  let summary = "f32 graph -> integer graph with every NUMERICS.md parameter computed";
  let description = [{
    Types become i8/i16, requant tables / biases / gammas / embedding / RoPE
    tables become new `llaccel.weight` symbols with attached data, and every
    op gets its integer parameters (docs/NUMERICS.md "Compiler:" lines).
  }];
  let options = [
    Option<"calib", "calib", "std::string", "\"\"", "calib.json (per-tensor abs-max)">,
    Option<"weights", "weights", "std::string", "\"\"", "weights.bin (raw f32)">,
    Option<"weightsJson", "weights-json", "std::string", "\"\"", "weights.json (name/shape/offset)">,
  ];
}

def LLAccelFuse : Pass<"llaccel-fuse", "::mlir::ModuleOp"> {
  let summary = "fuse linear+add / linear+silu / linear+mul into GEMM epilogues (target v2)";
  let options = [
    Option<"enable", "enable", "bool", "true", "perform fusion (false = no-op)">,
  ];
}

def LLAccelTile : Pass<"llaccel-tile", "::mlir::ModuleOp"> {
  let summary = "pad N to 16 and split linears along N into weight chunks";
  let options = [
    Option<"weightChunkBytes", "weight-chunk-bytes", "int64_t", "65536",
           "maximum tiled weight bytes per chunk (multiple of 256)">,
  ];
}

def LLAccelLowerToISA : Pass<"llaccel-lower-to-isa", "::mlir::ModuleOp"> {
  let summary = "high-level ops -> llaccel.isa.* programs (prefill M rows, decode M=1)";
  let options = [
    Option<"prefillM", "prefill-m", "int64_t", "16", "rows per prefill chunk (<= 16)">,
    Option<"weightBuffers", "weight-buffers", "int64_t", "2", "rotating SRAM weight buffers">,
  ];
}

def LLAccelAllocSram : Pass<"llaccel-alloc-sram", "::mlir::ModuleOp"> {
  let summary = "liveness-based first-fit SRAM allocation of isa.alloc buffers";
  let options = [
    Option<"sramBytes", "sram-bytes", "int64_t", "1048576", "SRAM size">,
  ];
}

def LLAccelSchedule : Pass<"llaccel-schedule", "::mlir::ModuleOp"> {
  let summary = "engine dependency analysis, DMA prefetch hoisting, semaphore assignment";
  let options = [
    Option<"mode", "mode", "std::string", "\"inorder\"", "inorder | overlap">,
  ];
}

def LLAccelEmit : Pass<"llaccel-emit", "::mlir::ModuleOp"> {
  let summary = "write the .llbin container (DRAM image + programs + META_JSON)";
  let options = [
    Option<"output", "output", "std::string", "\"\"", "output .llbin path">,
    Option<"target", "target", "std::string", "\"llaccel-v1\"", "target name for META_JSON">,
    Option<"fusion", "fusion", "bool", "false", "fusion flag for META_JSON">,
    Option<"schedule", "schedule", "std::string", "\"inorder\"", "schedule mode for META_JSON">,
    Option<"printStats", "print-stats", "bool", "false", "print per-program statistics">,
  ];
}

#endif // LLACCEL_PASSES_TD
