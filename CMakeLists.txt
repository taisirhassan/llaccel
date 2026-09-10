cmake_minimum_required(VERSION 3.28)
project(llaccel LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(LLACCEL_RTL "Build the Verilated RTL backend into llaccel-sim" ON)
option(LLACCEL_COMPILER "Build the MLIR compiler" ON)

find_path(NLOHMANN_JSON_INCLUDE nlohmann/json.hpp HINTS /opt/homebrew/include REQUIRED)

# ---- runtime -------------------------------------------------------------------
add_library(llaccel_runtime STATIC
  runtime/src/llbin.cpp
  runtime/src/funcsim.cpp
  runtime/src/host.cpp
  runtime/src/rtlsim.cpp)
target_include_directories(llaccel_runtime PUBLIC include runtime/include ${NLOHMANN_JSON_INCLUDE})
target_compile_options(llaccel_runtime PRIVATE -Wall -Wextra)

if(LLACCEL_RTL)
  find_package(verilator HINTS $ENV{VERILATOR_ROOT} /opt/homebrew/opt/verilator)
  if(verilator_FOUND)
    file(STRINGS rtl/filelist.f RTL_LIST)
    set(RTL_SOURCES)
    foreach(f IN LISTS RTL_LIST)
      string(STRIP "${f}" f)
      if(f AND NOT f MATCHES "^[#+]")
        list(APPEND RTL_SOURCES ${CMAKE_SOURCE_DIR}/${f})
      endif()
    endforeach()
    foreach(variant IN ITEMS 1 2)
      math(EXPR fusion "${variant} - 1")
      add_library(rtl_v${variant} STATIC)
      target_include_directories(rtl_v${variant} PUBLIC ${CMAKE_SOURCE_DIR}/rtl)
      verilate(rtl_v${variant}
        SOURCES ${RTL_SOURCES}
        TOP_MODULE llaccel_top
        PREFIX Vtop_v${variant}
        INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/rtl
        TRACE_FST
        VERILATOR_ARGS -GEPILOGUE_FUSION=${fusion} -Wno-fatal -O3 --x-assign fast --x-initial fast
                       -CFLAGS "-O2 -std=c++26")
      target_link_libraries(llaccel_runtime PUBLIC rtl_v${variant})
    endforeach()
    target_compile_definitions(llaccel_runtime PRIVATE LLACCEL_HAVE_RTL=1)
    message(STATUS "llaccel: RTL backend enabled (Verilator ${verilator_VERSION})")
  else()
    message(WARNING "llaccel: Verilator not found; llaccel-sim will be func-sim only")
  endif()
endif()

add_executable(llaccel-sim runtime/src/main.cpp)
target_link_libraries(llaccel-sim PRIVATE llaccel_runtime)

# ---- compiler ---------------------------------------------------------------------
if(LLACCEL_COMPILER AND EXISTS ${CMAKE_SOURCE_DIR}/compiler/CMakeLists.txt)
  add_subdirectory(compiler)
endif()

# ---- tests ------------------------------------------------------------------------------
enable_testing()
add_executable(numerics_selftest tests/cpp/numerics_selftest.cpp)
target_include_directories(numerics_selftest PRIVATE include)
add_test(NAME numerics_selftest COMMAND numerics_selftest)
