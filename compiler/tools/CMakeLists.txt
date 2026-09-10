set(LLACCEL_TOOL_LIBS LLAccelDialect LLAccelSupport LLAccelTransforms)

add_llvm_executable(llaccel-opt llaccel-opt.cpp)
target_link_libraries(llaccel-opt PRIVATE ${LLACCEL_TOOL_LIBS})
mlir_target_link_libraries(llaccel-opt PRIVATE MLIROptLib MLIRFuncDialect MLIRTransforms)

add_llvm_executable(llaccel-compile llaccel-compile.cpp)
target_link_libraries(llaccel-compile PRIVATE ${LLACCEL_TOOL_LIBS})
mlir_target_link_libraries(llaccel-compile PRIVATE MLIRParser MLIRPass MLIRFuncDialect MLIRTransforms)

# The disassembler only needs isa.h + LLVM support (cl::opt); no MLIR.
add_llvm_executable(llaccel-disasm llaccel-disasm.cpp)
target_link_libraries(llaccel-disasm PRIVATE LLVMSupport)

foreach(t llaccel-opt llaccel-compile llaccel-disasm)
  target_compile_options(${t} PRIVATE ${LLACCEL_COMPILER_WARNINGS})
endforeach()
