cmake_minimum_required(VERSION 3.24)

project(CellShardInspectConsumer LANGUAGES CXX)

option(CELLSHARD_EXPECT_RUNTIME "Expect the installed CellShard package to export CellShard::runtime." OFF)
option(CELLSHARD_EXPECT_INGEST "Expect the installed CellShard package to export CellShard::ingest." OFF)
option(CELLSHARD_EXPECT_PREPROCESS "Expect the install prefix to include CellShardPreprocess." OFF)

find_package(CellShard REQUIRED COMPONENTS inspect)

if(CELLSHARD_EXPECT_RUNTIME)
    if(NOT TARGET CellShard::runtime)
        message(FATAL_ERROR "Expected CellShard::runtime in this installed package.")
    endif()
    enable_language(CUDA)
else()
    if(TARGET CellShard::runtime)
        message(FATAL_ERROR "Did not expect CellShard::runtime in this installed package.")
    endif()
endif()

if(CELLSHARD_EXPECT_INGEST)
    find_package(CellShard REQUIRED COMPONENTS ingest)
    if(NOT TARGET CellShard::ingest)
        message(FATAL_ERROR "Expected CellShard::ingest in this installed package.")
    endif()
else()
    if(TARGET CellShard::ingest)
        message(FATAL_ERROR "Did not expect CellShard::ingest in this installed package.")
    endif()
endif()

if(CELLSHARD_EXPECT_PREPROCESS)
    if(NOT CELLSHARD_EXPECT_RUNTIME)
        message(FATAL_ERROR "CELLSHARD_EXPECT_PREPROCESS requires CELLSHARD_EXPECT_RUNTIME=ON.")
    endif()
    find_package(CellShard REQUIRED COMPONENTS runtime)
    find_package(CellShardPreprocess REQUIRED)
    if(NOT TARGET CellShardPreprocess::runtime)
        message(FATAL_ERROR "Expected CellShardPreprocess::runtime in this installed package.")
    endif()
else()
    find_package(CellShardPreprocess QUIET CONFIG PATHS ${CMAKE_PREFIX_PATH} NO_DEFAULT_PATH)
    if(CellShardPreprocess_FOUND OR TARGET CellShardPreprocess::runtime)
        message(FATAL_ERROR "Did not expect CellShardPreprocess in this installed package.")
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

if(CELLSHARD_EXPECT_RUNTIME)
    add_executable(cellshard_runtime_consumer
        runtime_consumer.cu
    )
    target_link_libraries(cellshard_runtime_consumer PRIVATE CellShard::runtime)
    set_target_properties(cellshard_runtime_consumer PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )
endif()

if(CELLSHARD_EXPECT_PREPROCESS)
    add_executable(cellshard_preprocess_consumer
        preprocess_consumer.cu
    )
    target_link_libraries(cellshard_preprocess_consumer PRIVATE CellShardPreprocess::runtime)
    set_target_properties(cellshard_preprocess_consumer PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )
endif()

if(CELLSHARD_EXPECT_INGEST)
    add_executable(cellshard_ingest_consumer
        ingest_consumer.cu
    )
    target_link_libraries(cellshard_ingest_consumer PRIVATE CellShard::ingest)
    set_target_properties(cellshard_ingest_consumer PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )
endif()
