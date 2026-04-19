cmake_minimum_required(VERSION 3.24)

option(CELLSHARD_ENABLE_CUDA "Build the CellShard CUDA runtime and staging surfaces." ON)

if(CELLSHARD_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        if(DEFINED ENV{CUDACXX} AND NOT "$ENV{CUDACXX}" STREQUAL "")
            set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}" CACHE FILEPATH "CUDA compiler" FORCE)
        elseif(EXISTS "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc")
            set(CMAKE_CUDA_COMPILER
                "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc"
                CACHE FILEPATH "CUDA compiler" FORCE)
            message(STATUS "Defaulting CellShard CUDA compiler to the local HPC SDK nvcc")
        endif()
    endif()

    if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
        if(DEFINED ENV{CUDAHOSTCXX} AND NOT "$ENV{CUDAHOSTCXX}" STREQUAL "")
            set(CMAKE_CUDA_HOST_COMPILER "$ENV{CUDAHOSTCXX}" CACHE FILEPATH "CUDA host compiler" FORCE)
        elseif(EXISTS "/usr/bin/g++-12")
            set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12" CACHE FILEPATH "CUDA host compiler" FORCE)
            message(STATUS "Defaulting CellShard CUDA host compiler to /usr/bin/g++-12 for nvcc compatibility")
        endif()
    endif()

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70)
    endif()

    project(CellShard LANGUAGES CXX CUDA)
else()
    project(CellShard LANGUAGES CXX)
    message(WARNING
        "Building CellShard without CUDA. "
        "This enables a slow host-only inspect/materialize mode for portability, not the normal GPU runtime path.")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CELLSHARD_VERSION 0.1.0)

option(CELLSHARD_BUILD_EXPORT "Build the CellShard export helpers." ON)
option(CELLSHARD_ENABLE_PYTHON "Build the optional CellShard Python bindings and H5AD export app." OFF)
option(CELLSHARD_BUILD_APPS "Build optional CellShard applications." ON)
option(CELLSHARD_BUILD_TESTS "Build CellShard-local test targets." OFF)
option(CELLSHARD_ENABLE_CUDA_LINEINFO "Emit CUDA line mapping for Nsight profiling." OFF)
set(CELLSHARD_CUDA_MODE "generic" CACHE STRING
    "CellShard CUDA optimization mode: generic, native, or native-extreme")
set_property(CACHE CELLSHARD_CUDA_MODE PROPERTY STRINGS generic native native-extreme)

if(NOT CELLSHARD_CUDA_MODE STREQUAL "generic"
   AND NOT CELLSHARD_CUDA_MODE STREQUAL "native"
   AND NOT CELLSHARD_CUDA_MODE STREQUAL "native-extreme")
    message(FATAL_ERROR "CELLSHARD_CUDA_MODE must be one of: generic, native, native-extreme")
endif()

if(CELLSHARD_ENABLE_CUDA AND CELLSHARD_CUDA_MODE STREQUAL "native-extreme")
    set(cellshard_extreme_arch_count 0)
    foreach(cellshard_cuda_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        math(EXPR cellshard_extreme_arch_count "${cellshard_extreme_arch_count} + 1")
        string(STRIP "${cellshard_cuda_arch}" cellshard_cuda_arch_clean)
        if(NOT cellshard_cuda_arch_clean MATCHES "^(70|7\\.0)(-real|-virtual)?$")
            message(FATAL_ERROR
                "CELLSHARD_CUDA_MODE=native-extreme is a Volta-only path and currently requires "
                "CMAKE_CUDA_ARCHITECTURES to contain only sm_70 entries.")
        endif()
    endforeach()
    if(cellshard_extreme_arch_count EQUAL 0)
        message(FATAL_ERROR
            "CELLSHARD_CUDA_MODE=native-extreme requires at least one explicit sm_70 architecture entry.")
    endif()
endif()

set(CELLSHARD_CUDA_MODE_GENERIC 0)
set(CELLSHARD_CUDA_MODE_NATIVE 0)
set(CELLSHARD_CUDA_MODE_NATIVE_EXTREME 0)
if(CELLSHARD_CUDA_MODE STREQUAL "generic")
    set(CELLSHARD_CUDA_MODE_GENERIC 1)
elseif(CELLSHARD_CUDA_MODE STREQUAL "native")
    set(CELLSHARD_CUDA_MODE_NATIVE 1)
else()
    set(CELLSHARD_CUDA_MODE_NATIVE_EXTREME 1)
endif()

message(STATUS "CellShard CUDA mode: ${CELLSHARD_CUDA_MODE}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CELLSHARD_ENABLE_CUDA AND NOT TARGET CUDA::cudart)
    find_package(CUDAToolkit REQUIRED)
endif()

if(CELLSHARD_ENABLE_CUDA)
    find_library(CELLSHARD_NCCL_LIBRARY
        NAMES nccl
        HINTS
            /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/12.9/nccl/lib
    )
endif()

if(NOT HDF5_FOUND)
    find_package(HDF5 REQUIRED COMPONENTS C)
endif()

function(cellshard_enable_perf target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG -march=native -mtune=native>
    )
    if(CELLSHARD_ENABLE_CUDA)
        set(cellshard_cuda_perf_flags
            -O3
            --use_fast_math
            --expt-relaxed-constexpr
            --expt-extended-lambda
            -Xptxas=-O3,-dlcm=ca
        )
        if(CELLSHARD_ENABLE_CUDA_LINEINFO)
            list(APPEND cellshard_cuda_perf_flags -lineinfo)
        endif()
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${cellshard_cuda_perf_flags}>
        )
    endif()
    set_target_properties(${target} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE YES
    )
endfunction()

function(cellshard_assert_no_torch_dependency target)
    foreach(cellshard_link_property IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
        get_target_property(cellshard_links ${target} ${cellshard_link_property})
        if(NOT cellshard_links OR cellshard_links STREQUAL "cellshard_links-NOTFOUND")
            continue()
        endif()
        foreach(cellshard_link IN LISTS cellshard_links)
            if("${cellshard_link}" MATCHES "(^|::)(torch|Torch|ATen|c10)($|::)")
                message(FATAL_ERROR
                    "CellShard target ${target} must not depend on libtorch/Torch. "
                    "Keep Torch interop in Cellerator's src/torch/ boundary.")
            endif()
        endforeach()
    endforeach()
endfunction()

add_library(cellshard_headers INTERFACE)
add_library(CellShard::headers ALIAS cellshard_headers)
set_target_properties(cellshard_headers PROPERTIES EXPORT_NAME headers)
target_include_directories(cellshard_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(cellshard_headers INTERFACE cxx_std_17)
if(CELLSHARD_ENABLE_CUDA)
    target_compile_definitions(cellshard_headers INTERFACE CELLSHARD_ENABLE_CUDA=1)
else()
    target_compile_definitions(cellshard_headers INTERFACE CELLSHARD_ENABLE_CUDA=0)
endif()
target_compile_definitions(cellshard_headers INTERFACE
    CELLSHARD_CUDA_MODE_GENERIC=${CELLSHARD_CUDA_MODE_GENERIC}
    CELLSHARD_CUDA_MODE_NATIVE=${CELLSHARD_CUDA_MODE_NATIVE}
    CELLSHARD_CUDA_MODE_NATIVE_EXTREME=${CELLSHARD_CUDA_MODE_NATIVE_EXTREME}
)

set(cellshard_inspect_sources
    src/io/pack/packfile.cu
    src/io/csh5/create.cc
    src/io/csh5/metadata.cc
    src/io/csh5/finalize_preprocess.cc
    src/io/csh5/write.cc
    src/io/csh5/runtime/bind.cc
    src/io/csh5/runtime/headers.cc
    src/io/csh5/runtime/fetch.cc
    src/io/csh5/runtime/warm.cc
    src/bucket/blocked_ell_bipartite_optimize_bridge.cu
    src/runtime/storage/shard_storage.cu
)
set_source_files_properties(${cellshard_inspect_sources} PROPERTIES LANGUAGE CXX)
if(CELLSHARD_ENABLE_CUDA)
    set_source_files_properties(src/bucket/blocked_ell_bipartite_optimize_bridge.cu PROPERTIES LANGUAGE CUDA)
endif()

add_library(cellshard_inspect STATIC
    ${cellshard_inspect_sources}
)
add_library(CellShard::inspect ALIAS cellshard_inspect)
set_target_properties(cellshard_inspect PROPERTIES EXPORT_NAME inspect)
target_include_directories(cellshard_inspect
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/export
        ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(cellshard_inspect
    PUBLIC
        cellshard_headers
        ${HDF5_LIBRARIES}
)
if(CELLSHARD_ENABLE_CUDA)
    target_link_libraries(cellshard_inspect PUBLIC CUDA::cudart)
endif()
set_target_properties(cellshard_inspect PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellshard_enable_perf(cellshard_inspect)
cellshard_assert_no_torch_dependency(cellshard_inspect)

set(CELLSHARD_EXPORT_TARGETS
    cellshard_headers
    cellshard_inspect
)
set(CELLSHARD_INSTALL_TARGETS
    cellshard_inspect
)
set(CELLSHARD_PACKAGE_COMPONENTS
    headers
    inspect
)

if(CELLSHARD_ENABLE_CUDA)
    add_library(cellshard_runtime INTERFACE)
    add_library(CellShard::runtime ALIAS cellshard_runtime)
    set_target_properties(cellshard_runtime PROPERTIES EXPORT_NAME runtime)
    target_link_libraries(cellshard_runtime INTERFACE
        cellshard_inspect
        CUDA::cudart
    )
    if(CELLSHARD_NCCL_LIBRARY)
        target_link_libraries(cellshard_runtime INTERFACE ${CELLSHARD_NCCL_LIBRARY})
    endif()
    target_compile_features(cellshard_runtime INTERFACE cxx_std_17)
    cellshard_assert_no_torch_dependency(cellshard_runtime)
    list(APPEND CELLSHARD_EXPORT_TARGETS cellshard_runtime)
    list(APPEND CELLSHARD_PACKAGE_COMPONENTS runtime)
endif()

if(CELLSHARD_BUILD_EXPORT)
    add_library(cellshard_export STATIC
        export/summary/load_summary.cc
        export/materialize/csr_export.cc
        export/snapshot/global_metadata_snapshot.cc
    )
    add_library(CellShard::export ALIAS cellshard_export)
    set_target_properties(cellshard_export PROPERTIES EXPORT_NAME export)
    target_include_directories(cellshard_export
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/export
            ${HDF5_INCLUDE_DIRS}
    )
    target_link_libraries(cellshard_export
        PUBLIC
            cellshard_inspect
            ${HDF5_LIBRARIES}
    )
    set_target_properties(cellshard_export PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )
    cellshard_assert_no_torch_dependency(cellshard_export)
    list(APPEND CELLSHARD_EXPORT_TARGETS cellshard_export)
    list(APPEND CELLSHARD_INSTALL_TARGETS cellshard_export)
    list(APPEND CELLSHARD_PACKAGE_COMPONENTS export)
endif()

if(CELLSHARD_ENABLE_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module Development.Embed)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE CELLSHARD_PYBIND11_CMAKE_DIR
            RESULT_VARIABLE CELLSHARD_PYBIND11_CMAKE_DIR_STATUS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(CELLSHARD_PYBIND11_CMAKE_DIR_STATUS EQUAL 0 AND NOT "${CELLSHARD_PYBIND11_CMAKE_DIR}" STREQUAL "")
            list(PREPEND CMAKE_PREFIX_PATH "${CELLSHARD_PYBIND11_CMAKE_DIR}")
            find_package(pybind11 CONFIG QUIET)
        endif()
    endif()
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.6
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    if(NOT CELLSHARD_BUILD_EXPORT)
        message(FATAL_ERROR "CELLSHARD_ENABLE_PYTHON requires CELLSHARD_BUILD_EXPORT=ON")
    endif()

    add_library(cellshard_h5ad_python STATIC
        export/h5ad_writer.cc
    )
    add_library(CellShard::h5ad_python ALIAS cellshard_h5ad_python)
    set_target_properties(cellshard_h5ad_python PROPERTIES EXPORT_NAME h5ad_python)
    target_include_directories(cellshard_h5ad_python
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/export
            ${Python3_INCLUDE_DIRS}
    )
    target_link_libraries(cellshard_h5ad_python
        PUBLIC
            cellshard_export
        PRIVATE
            pybind11::embed
            Python3::Python
    )
    set_target_properties(cellshard_h5ad_python PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )
    cellshard_assert_no_torch_dependency(cellshard_h5ad_python)
    list(APPEND CELLSHARD_EXPORT_TARGETS cellshard_h5ad_python)
    list(APPEND CELLSHARD_INSTALL_TARGETS cellshard_h5ad_python)
    list(APPEND CELLSHARD_PACKAGE_COMPONENTS h5ad_python)

    pybind11_add_module(_cellshard
        python/module.cc
        python/bind_types.cc
        python/bind_handles.cc
    )
    target_include_directories(_cellshard PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/export
    )
    target_link_libraries(_cellshard PRIVATE
        cellshard_export
        cellshard_h5ad_python
    )
    set_target_properties(_cellshard PROPERTIES
        OUTPUT_NAME _cellshard
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )
    cellshard_assert_no_torch_dependency(_cellshard)

    if(CELLSHARD_BUILD_APPS)
        add_executable(cellshardH5adExport
            export/cellshard_h5ad_export.cc
        )
        target_include_directories(cellshardH5adExport PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/export
        )
        target_link_libraries(cellshardH5adExport PRIVATE
            cellshard_export
            cellshard_h5ad_python
            pybind11::embed
        )
        set_target_properties(cellshardH5adExport PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED YES
        )
        cellshard_assert_no_torch_dependency(cellshardH5adExport)
    endif()
endif()

if(CELLSHARD_BUILD_TESTS AND CELLSHARD_BUILD_EXPORT)
    if(CELLSHARD_ENABLE_CUDA)
        set(cellshard_export_runtime_test_source tests/cellshard_export_runtime_test.cu)
    else()
        set(cellshard_export_runtime_test_source tests/cellshard_export_runtime_test.cc)
    endif()
    add_executable(cellShardExportRuntimeTest
        ${cellshard_export_runtime_test_source}
    )
    target_include_directories(cellShardExportRuntimeTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/export
    )
    target_link_libraries(cellShardExportRuntimeTest PRIVATE
        cellshard_export
        ${HDF5_LIBRARIES}
    )
    set_target_properties(cellShardExportRuntimeTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )
    if(CELLSHARD_ENABLE_CUDA)
        set_target_properties(cellShardExportRuntimeTest PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED YES
        )
    endif()
endif()

if(CELLSHARD_BUILD_TESTS)
    set(cellshard_package_test_root ${CMAKE_CURRENT_BINARY_DIR}/tests/package_consumer)
    set(cellshard_package_test_stage ${cellshard_package_test_root}/install)
    set(cellshard_package_test_build ${cellshard_package_test_root}/build)
    if(CELLSHARD_ENABLE_CUDA)
        set(cellshard_package_test_expect_runtime ON)
    else()
        set(cellshard_package_test_expect_runtime OFF)
    endif()

    add_custom_target(cellShardInspectPackageTest
        DEPENDS
            cellshard_headers
            ${CELLSHARD_INSTALL_TARGETS}
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${cellshard_package_test_root}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${cellshard_package_test_root}
        COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} --prefix ${cellshard_package_test_stage}
        COMMAND ${CMAKE_COMMAND}
            -S ${CMAKE_CURRENT_SOURCE_DIR}/tests/package_consumer
            -B ${cellshard_package_test_build}
            -DCMAKE_PREFIX_PATH=${cellshard_package_test_stage}
            -DCELLSHARD_EXPECT_RUNTIME=${cellshard_package_test_expect_runtime}
            -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
            -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}
            -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
        COMMAND ${CMAKE_COMMAND} --build ${cellshard_package_test_build} -j 1
        USES_TERMINAL
    )
endif()

install(TARGETS ${CELLSHARD_INSTALL_TARGETS}
    EXPORT CellShardTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(TARGETS cellshard_headers
    EXPORT CellShardTargets
)

if(TARGET cellshard_runtime)
    install(TARGETS cellshard_runtime
        EXPORT CellShardTargets
    )
endif()

if(TARGET _cellshard)
    install(TARGETS _cellshard
        LIBRARY DESTINATION cellshard
        RUNTIME DESTINATION cellshard
    )
    install(FILES
        python/cellshard/__init__.py
        DESTINATION cellshard
    )
endif()

if(TARGET cellshardH5adExport)
    install(TARGETS cellshardH5adExport
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
        PATTERN "*.hh"
        PATTERN "*.cuh"
)

install(EXPORT CellShardTargets
    FILE CellShardTargets.cmake
    NAMESPACE CellShard::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CellShard
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/CellShardConfigVersion.cmake
    VERSION ${CELLSHARD_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CellShardConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CellShardConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CellShard
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CellShardConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CellShardConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CellShard
)
