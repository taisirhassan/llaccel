add_mlir_dialect_library(LLAccelDialect
  LLAccelDialect.cpp
  LLAccelOps.cpp

  DEPENDS
  MLIRLLAccelOpsIncGen
  MLIRLLAccelAttrsIncGen
  MLIRLLAccelInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFunctionInterfaces
  MLIRSideEffectInterfaces
  MLIRBytecodeOpInterface
)
target_compile_options(LLAccelDialect PRIVATE ${LLACCEL_COMPILER_WARNINGS})
