find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME llaccel-compiler-regressions
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_compiler.py
          ${LLACCEL_COMPILER_BINARY_DIR}/bin)
add_test(NAME llaccel-compiler-streaming
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming.py
          ${LLACCEL_COMPILER_BINARY_DIR}/bin)

if(TARGET llaccel-sim)
  add_test(NAME llaccel-compiler-streaming-functional
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming.py
            ${LLACCEL_COMPILER_BINARY_DIR}/bin $<TARGET_FILE:llaccel-sim>)
endif()
add_test(NAME llaccel-compiler-dram-kv
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_dram_kv.py
          ${LLACCEL_COMPILER_BINARY_DIR}/bin)

add_test(NAME llaccel-compiler-hf-geometry
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_hf_geometry.py
          ${LLACCEL_COMPILER_BINARY_DIR}/bin)
