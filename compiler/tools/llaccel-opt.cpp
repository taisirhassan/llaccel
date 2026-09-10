//===- llaccel-opt.cpp - mlir-opt with the llaccel dialect and passes -----===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "llaccel/Dialect/LLAccelDialect.h"
#include "llaccel/Dialect/LLAccelOps.h"
#ifndef LLACCEL_NO_PASSES
#include "llaccel/Transforms/Passes.h"
#endif

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llaccel::LLAccelDialect, mlir::func::FuncDialect>();
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
#ifndef LLACCEL_NO_PASSES
  mlir::llaccel::registerLLAccelPasses();
#endif
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "llaccel optimizer\n", registry));
}
