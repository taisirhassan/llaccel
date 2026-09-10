//===- LLAccelDialect.td - llaccel dialect definition ------*- tablegen -*-===//
//
// The `llaccel` dialect: high-level transformer ops (docs/DIALECT.md section 1)
// and ISA-level ops (`llaccel.isa.*`, docs/ISA.md) in one dialect.
//
//===----------------------------------------------------------------------===//
#ifndef LLACCEL_DIALECT_TD
#define LLACCEL_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def LLAccel_Dialect : Dialect {
  let name = "llaccel";
  let summary = "llaccel transformer-inference accelerator dialect";
  let description = [{
    High-level ops operate on `tensor<?xNxT>` values where `?` is the token
    (row) dimension. Before `llaccel-quantize` the element type is `f32`;
    afterwards it is `i8` (per-tensor float scale, attribute `llaccel.scale`
    on the defining op) or `i16` (per-tensor power-of-two exponent, attribute
    `llaccel.exp`). Every value carries `llaccel.name`, the key into calib.json
    and into the qgraph.json dump.

    ISA-level ops (`llaccel.isa.*`) map 1:1 onto docs/ISA.md instructions.
    SRAM buffers are `!llaccel.buf` values produced by `llaccel.isa.alloc`;
    every buffer operand is accompanied by a byte-offset attribute.
  }];
  let cppNamespace = "::mlir::llaccel";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    /// Discardable attribute names shared by all passes.
    static constexpr ::llvm::StringLiteral kNameAttr = "llaccel.name";
    static constexpr ::llvm::StringLiteral kExpAttr = "llaccel.exp";
    static constexpr ::llvm::StringLiteral kScaleAttr = "llaccel.scale";
    static constexpr ::llvm::StringLiteral kModelAttr = "llaccel.model";
  }];
}

class LLAccel_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LLAccel_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def LLAccel_BufType : LLAccel_Type<"Buf", "buf"> {
  let summary = "an SRAM buffer (symbolic before llaccel-alloc-sram, placed after)";
}

// ---- GEMM epilogue mode -------------------------------------------------------
def LLAccel_EpNone   : I64EnumAttrCase<"none", 0>;
def LLAccel_EpResAdd : I64EnumAttrCase<"resadd", 1>;
def LLAccel_EpSilu   : I64EnumAttrCase<"silu", 2>;
def LLAccel_EpMul    : I64EnumAttrCase<"mul", 3>;

def LLAccel_EpilogueMode : I64EnumAttr<"EpilogueMode", "GEMM epilogue mode (ISA.md `ep.mode`)",
    [LLAccel_EpNone, LLAccel_EpResAdd, LLAccel_EpSilu, LLAccel_EpMul]> {
  let cppNamespace = "::mlir::llaccel";
  let genSpecializedAttr = 0;
}

def LLAccel_EpilogueAttr : EnumAttr<LLAccel_Dialect, LLAccel_EpilogueMode, "epilogue"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // LLACCEL_DIALECT_TD
