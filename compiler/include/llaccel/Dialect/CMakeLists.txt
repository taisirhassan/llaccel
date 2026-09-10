# Ops + dialect + types (standard add_mlir_dialect layout).
add_mlir_dialect(LLAccelOps llaccel)

# Attributes (enum attr) and enums live in the same .td; generate them too.
set(LLVM_TARGET_DEFINITIONS LLAccelOps.td)
mlir_tablegen(LLAccelEnums.h.inc -gen-enum-decls)
mlir_tablegen(LLAccelEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(LLAccelAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=llaccel)
mlir_tablegen(LLAccelAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=llaccel)
add_public_tablegen_target(MLIRLLAccelAttrsIncGen)

# Op interface implemented by every llaccel.isa.* op.
add_mlir_interface(LLAccelInterfaces)
