# llaccel MLIR compiler. Builds standalone (cmake -S compiler -B build/compiler)
# or as a subdirectory of the top-level project (add_subdirectory(compiler)).
cmake_minimum_required(VERSION 3.28)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(llaccel-compiler LANGUAGES C CXX)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
  enable_testing()
  set(LLACCEL_COMPILER_STANDALONE ON)
else()
  set(LLACCEL_COMPILER_STANDALONE OFF)
endif()

set(LLACCEL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
# Tools end up in <compiler binary dir>/bin (build/cmake/compiler/bin from the
# top level, build/compiler/bin standalone) — the top-level Makefile expects that.
set(LLACCEL_COMPILER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Locate MLIR/LLVM unless the enclosing project already did. Homebrew LLVM is
# the default location on this machine; override with -DMLIR_DIR/-DLLVM_DIR.
if(NOT MLIR_FOUND)
  if(NOT MLIR_DIR AND EXISTS /opt/homebrew/opt/llvm/lib/cmake/mlir)
    set(MLIR_DIR /opt/homebrew/opt/llvm/lib/cmake/mlir)
  endif()
  if(NOT LLVM_DIR AND EXISTS /opt/homebrew/opt/llvm/lib/cmake/llvm)
    set(LLVM_DIR /opt/homebrew/opt/llvm/lib/cmake/llvm)
  endif()
  find_package(MLIR REQUIRED CONFIG)
endif()
message(STATUS "llaccel-compiler: MLIR ${MLIR_DIR} (LLVM ${LLVM_PACKAGE_VERSION})")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# add_llvm_executable/add_mlir_library honour these when set.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${LLACCEL_COMPILER_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${LLACCEL_COMPILER_BINARY_DIR}/lib)

# Homebrew LLVM ships libMLIR.dylib and sets MLIR_LINK_MLIR_DYLIB=1; the
# per-component static archives are also present. We link the dylib (via
# mlir_target_link_libraries) so every tool shares one copy of the MLIR
# context/registry machinery.
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${LLACCEL_ROOT}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

# LLVM's HandleLLVMOptions turns RTTI/exceptions off (matches the MLIR dylib).
set(LLACCEL_COMPILER_WARNINGS -Wall -Wextra -Wno-unused-parameter)

option(LLACCEL_DIALECT_ONLY "Build only the dialect + support libraries (development aid)" OFF)

add_subdirectory(include/llaccel)
if(LLACCEL_DIALECT_ONLY)
  add_subdirectory(lib/Dialect)
  add_subdirectory(lib/Support)
else()
  add_subdirectory(lib)
  add_subdirectory(tools)
  add_subdirectory(test)
endif()

if(LLACCEL_DIALECT_ONLY)
  add_llvm_executable(llaccel-opt-dialect tools/llaccel-opt.cpp)
  target_compile_definitions(llaccel-opt-dialect PRIVATE LLACCEL_NO_PASSES=1)
  target_link_libraries(llaccel-opt-dialect PRIVATE LLAccelDialect)
  mlir_target_link_libraries(llaccel-opt-dialect PRIVATE MLIROptLib MLIRFuncDialect MLIRTransforms)
endif()
