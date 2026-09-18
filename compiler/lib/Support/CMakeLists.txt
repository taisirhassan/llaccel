add_mlir_library(LLAccelSupport
  ModelInfo.cpp
  ConstantData.cpp

  LINK_LIBS PUBLIC
  LLAccelDialect
)
target_compile_options(LLAccelSupport PRIVATE ${LLACCEL_COMPILER_WARNINGS})
mlir_target_link_libraries(LLAccelSupport PUBLIC MLIRIR MLIRSupport)
