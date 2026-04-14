#include "src/cuda_compat.cuh"
#include "src/sharded/sharded.cuh"

int main() {
    ::cellshard::sharded<::cellshard::sparse::compressed> matrix;
    ::cellshard::init(&matrix);
    return 0;
}
