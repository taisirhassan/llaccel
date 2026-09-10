add_mlir_library(LLAccelTransforms
  Quantize.cpp
  Fuse.cpp
  Tile.cpp
  LowerToISA.cpp
  AllocSram.cpp
  Schedule.cpp
  Emit.cpp
  QGraphDump.cpp

  DEPENDS
  MLIRLLAccelPassIncGen

  LINK_LIBS PUBLIC
  LLAccelDialect
  LLAccelSupport
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRTransformUtils
)
target_compile_options(LLAccelTransforms PRIVATE ${LLACCEL_COMPILER_WARNINGS})
