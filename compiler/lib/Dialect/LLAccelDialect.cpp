//===- LLAccelDialect.cpp - llaccel dialect registration ------------------===//
#include "llaccel/Dialect/LLAccelDialect.h"
#include "llaccel/Dialect/LLAccelOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::llaccel;

#include "llaccel/Dialect/LLAccelOpsDialect.cpp.inc"
#include "llaccel/Dialect/LLAccelEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "llaccel/Dialect/LLAccelOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "llaccel/Dialect/LLAccelAttrs.cpp.inc"

void LLAccelDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "llaccel/Dialect/LLAccelOpsTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "llaccel/Dialect/LLAccelAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "llaccel/Dialect/LLAccelOps.cpp.inc"
      >();
}
