add_mlir_library(LLAccelSupport
  ModelInfo.cpp
  ConstantData.cpp

  LINK_LIBS PUBLIC
  LLAccelDialect
  MLIRIR
  MLIRSupport
)
target_compile_options(LLAccelSupport PRIVATE ${LLACCEL_COMPILER_WARNINGS})
