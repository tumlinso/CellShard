#pragma once

#include "../types.cuh"

namespace cellshard {
namespace sparse {

struct alignas(16) dia {
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;

    int *offsets;
    real::storage_t *val;
    types::idx_t num_diagonals;
};

__host__ __device__ __forceinline__ void init(
    dia * __restrict__ m,
    types::dim_t rows = 0,
    types::dim_t cols = 0,
    types::nnz_t nnz = 0
) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->offsets = 0;
    m->val = 0;
    m->num_diagonals = 0;
}

__host__ __device__ __forceinline__ std::size_t bytes(const dia * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) m->num_diagonals * sizeof(int)
        + (std::size_t) m->nnz * sizeof(real::storage_t);
}

__host__ __device__ __forceinline__ const real::storage_t *at(const dia * __restrict__ m, types::dim_t r, types::dim_t c) {
    const int offset = (int) c - (int) r;
    for (types::idx_t i = 0; i < m->num_diagonals; ++i) {
        if (m->offsets[i] == offset) return m->val + i * m->rows + r;
    }
    return 0;
}

__host__ __device__ __forceinline__ real::storage_t *at(dia * __restrict__ m, types::dim_t r, types::dim_t c) {
    const int offset = (int) c - (int) r;
    for (types::idx_t i = 0; i < m->num_diagonals; ++i) {
        if (m->offsets[i] == offset) return m->val + i * m->rows + r;
    }
    return 0;
}

} // namespace sparse
} // namespace cellshard
