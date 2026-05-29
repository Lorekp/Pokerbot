cmake_minimum_required(VERSION 3.15)
project(texas_solver_cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED TEXAS_SOLVER_ROOT)
  set(TEXAS_SOLVER_ROOT "$ENV{TEXAS_SOLVER_ROOT}" CACHE PATH "TexasSolver install dir (console_solver + resources/)")
endif()

add_executable(texas_solver_cli
  src/main.cpp
  src/request_adapter.cpp
  src/response_adapter.cpp
  src/json_minimal.cpp
)

target_include_directories(texas_solver_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(MSVC)
  target_compile_options(texas_solver_cli PRIVATE /W4 /O2)
else()
  target_compile_options(texas_solver_cli PRIVATE -Wall -Wextra -O3)
endif()

install(TARGETS texas_solver_cli RUNTIME DESTINATION bin)
