//===- LLAccelDialect.h - llaccel dialect ------------------------*- C++ -*-===//
#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include "llaccel/Dialect/LLAccelOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "llaccel/Dialect/LLAccelOpsTypes.h.inc"

#include "llaccel/Dialect/LLAccelEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "llaccel/Dialect/LLAccelAttrs.h.inc"
