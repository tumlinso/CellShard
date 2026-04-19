#include <CellShard/core/cuda_compat.cuh>
#include <CellShard/runtime/layout/sharded.cuh>

int main() {
    ::cellshard::sharded<::cellshard::sparse::compressed> matrix;
    ::cellshard::init(&matrix);
    return 0;
}
