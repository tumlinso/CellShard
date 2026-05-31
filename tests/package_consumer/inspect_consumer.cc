#include <CellShard/access.hh>
#include <CellShard/core/cuda_compat.cuh>
#include <CellShard/runtime/layout/sharded.cuh>

int main() {
    ::cellshard::sharded<::cellshard::sparse::compressed> matrix;
    ::cellshard::init(&matrix);
    static_assert(::cellshard::access::is_archive_adapter<::cellshard::access::dense_fallback_binding>::value,
                  "installed access fallback headers should compile for package consumers");
    return 0;
}
