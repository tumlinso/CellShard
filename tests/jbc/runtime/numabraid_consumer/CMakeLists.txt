cmake_minimum_required(VERSION 3.28)
project(cellshard_numabraid_consumer LANGUAGES CXX)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
elseif(CMAKE_CXX_STANDARD LESS 20)
    message(FATAL_ERROR "CellShard requires C++20 or newer.")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CELLSHARD_ROOT)
    message(FATAL_ERROR "CELLSHARD_ROOT is required")
endif()

add_executable(
    cellshard_numabraid_consumer
    "${CMAKE_CURRENT_LIST_DIR}/../numabraid_capability_test.cc"
)
target_include_directories(
    cellshard_numabraid_consumer PRIVATE "${CELLSHARD_ROOT}/include"
)
include("${CELLSHARD_ROOT}/src/runtime/v2/numabraid_build.cmake")
cellshard_runtime_v2_link_numabraid(cellshard_numabraid_consumer)
