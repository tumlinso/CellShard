# Opt-in bridge for CellShard runtime-v2 targets. The root build may include
# this fragment only when numaBraid is provisioned; default CellShard builds
# remain independent of the experimental forwarding package.
function(cellshard_runtime_v2_link_numabraid target)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "Unknown CellShard runtime-v2 target: ${target}")
    endif()
    find_package(numabraid CONFIG REQUIRED)
    target_link_libraries("${target}" PRIVATE numabraid::numabraid)
    target_compile_definitions(
        "${target}" PRIVATE CELLSHARD_RUNTIME_V2_HAS_NUMABRAID=1
    )
endfunction()
