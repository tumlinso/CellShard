cmake_minimum_required(VERSION 3.28)

project(CellShardInspectConsumer LANGUAGES C CXX)

option(CELLSHARD_EXPECT_RUNTIME "Expect the installed CellShard package to export CellShard::runtime." OFF)
option(CELLSHARD_EXPECT_INGEST "Expect the installed CellShard package to export CellShard::ingest." OFF)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
elseif(CMAKE_CXX_STANDARD LESS 20)
    message(FATAL_ERROR "The installed CellShard package requires C++20 or newer.")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
elseif(CELLSHARD_EXPECT_RUNTIME AND CMAKE_CUDA_STANDARD LESS 20)
    message(FATAL_ERROR "The installed CellShard runtime requires CUDA C++20 or newer.")
endif()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CellShard REQUIRED COMPONENTS inspect anndata_ingest)
if(NOT TARGET CellShard::anndata_ingest)
    message(FATAL_ERROR "Expected CellShard::anndata_ingest in every installed CellShard package.")
endif()

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

add_executable(cellshard_anndata_ingest_consumer
    ../anndata_ingest_test.cc
)
target_link_libraries(cellshard_anndata_ingest_consumer PRIVATE CellShard::anndata_ingest)
set_target_properties(cellshard_anndata_ingest_consumer PROPERTIES
    CXX_STANDARD ${CMAKE_CXX_STANDARD}
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

set(cellshard_anndatapp_importer_source
    "${CMAKE_CURRENT_BINARY_DIR}/anndatapp_module_importer.cc")
file(WRITE "${cellshard_anndatapp_importer_source}"
    "import anndatapp.h5ad;\nint main() { return 0; }\n")
foreach(cellshard_importer IN ITEMS one two)
    add_executable(cellshard_anndatapp_module_importer_${cellshard_importer}
        "${cellshard_anndatapp_importer_source}")
    target_link_libraries(cellshard_anndatapp_module_importer_${cellshard_importer}
        PRIVATE anndatapp::anndatapp)
    set_target_properties(cellshard_anndatapp_module_importer_${cellshard_importer}
        PROPERTIES CXX_STANDARD ${CMAKE_CXX_STANDARD} CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS OFF)
endforeach()

add_executable(cellshard_inspect_consumer
    inspect_consumer.cc
)
target_link_libraries(cellshard_inspect_consumer PRIVATE CellShard::inspect)
set_target_properties(cellshard_inspect_consumer PROPERTIES
    CXX_STANDARD ${CMAKE_CXX_STANDARD}
    CXX_STANDARD_REQUIRED YES
)

if(CELLSHARD_EXPECT_RUNTIME)
    add_executable(cellshard_runtime_consumer
        runtime_consumer.cu
    )
    target_link_libraries(cellshard_runtime_consumer PRIVATE CellShard::runtime)
    set_target_properties(cellshard_runtime_consumer PROPERTIES
        CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
        CUDA_STANDARD_REQUIRED YES
    )
endif()

if(CELLSHARD_EXPECT_INGEST)
    add_executable(cellshard_ingest_consumer
        ingest_consumer.cu
    )
    target_link_libraries(cellshard_ingest_consumer PRIVATE CellShard::ingest)
    set_target_properties(cellshard_ingest_consumer PROPERTIES
        CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
        CUDA_STANDARD_REQUIRED YES
    )
endif()
