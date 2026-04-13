cmake_minimum_required(VERSION 3.24)

project(CellShardInspectConsumer LANGUAGES CXX)

option(CELLSHARD_EXPECT_RUNTIME "Expect the installed CellShard package to export CellShard::runtime." OFF)

find_package(CellShard REQUIRED COMPONENTS inspect)

if(CELLSHARD_EXPECT_RUNTIME)
    if(NOT TARGET CellShard::runtime)
        message(FATAL_ERROR "Expected CellShard::runtime in this installed package.")
    endif()
else()
    if(TARGET CellShard::runtime)
        message(FATAL_ERROR "Did not expect CellShard::runtime in this installed package.")
    endif()
endif()

add_executable(cellshard_inspect_consumer
    inspect_consumer.cc
)
target_link_libraries(cellshard_inspect_consumer PRIVATE CellShard::inspect)
set_target_properties(cellshard_inspect_consumer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
