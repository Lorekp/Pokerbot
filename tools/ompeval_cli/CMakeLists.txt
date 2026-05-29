cmake_minimum_required(VERSION 3.16)
project(ompeval_cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

option(USE_OMPEVAL "Link vendored OMPEval" OFF)
option(BUILD_BENCH_NATIVE "Build Google Benchmark harness" ON)

set(OMPEVAL_VENDOR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/OMPEval")
set(OMPEVAL_HEADER "${OMPEVAL_VENDOR}/omp/HandEvaluator.h")

if(USE_OMPEVAL)
  if(NOT EXISTS "${OMPEVAL_HEADER}")
    message(FATAL_ERROR "Run: git submodule update --init tools/ompeval_cli/vendor/OMPEval")
  endif()
endif()

if(USE_OMPEVAL)
  add_library(ompeval_core STATIC
    vendor/OMPEval/omp/HandEvaluator.cpp
  )
  set_target_properties(ompeval_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(ompeval_core PUBLIC USE_OMPEVAL=1)
  target_include_directories(ompeval_core PUBLIC "${OMPEVAL_VENDOR}")

  find_package(OpenMP REQUIRED)

  add_library(hu_score2_core STATIC
    hu_score2_core.cpp
    hu_score2_batch.cpp
  )
  set_target_properties(hu_score2_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(hu_score2_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(hu_score2_core PUBLIC ompeval_core OpenMP::OpenMP_CXX)

  add_executable(hu_table_build
    hu_table_build.cpp
    hu_table_io.cpp
  )
  target_link_libraries(hu_table_build PRIVATE hu_score2_core)
endif()

add_executable(enumerate_hu_equity enumerate_hu_equity.cpp)

if(USE_OMPEVAL)
  target_compile_definitions(enumerate_hu_equity PRIVATE USE_OMPEVAL=1)
  target_link_libraries(enumerate_hu_equity PRIVATE ompeval_core)
else()
  message(STATUS "Building enumerate_hu_equity without OMPEval (stub eval7 exits unless USE_OMPEVAL)")
endif()

if(USE_OMPEVAL)
  find_package(Python COMPONENTS Interpreter Development REQUIRED)

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE pybind11_DIR
    RESULT_VARIABLE _pybind11_rc
  )
  if(_pybind11_rc OR NOT pybind11_DIR)
    message(FATAL_ERROR "pybind11 not found; pip install pybind11")
  endif()
  find_package(pybind11 CONFIG REQUIRED PATHS "${pybind11_DIR}")

  pybind11_add_module(hu_table_native hu_table_native.cpp)
  target_link_libraries(hu_table_native PRIVATE hu_score2_core)
endif()

if(USE_OMPEVAL AND BUILD_BENCH_NATIVE)
  include(FetchContent)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.5
  )
  FetchContent_MakeAvailable(googlebenchmark)

  add_executable(bench_native bench_native.cpp hu_table_io.cpp)
  target_link_libraries(bench_native PRIVATE hu_score2_core benchmark::benchmark)
endif()
