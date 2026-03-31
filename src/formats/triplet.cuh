#pragma once

#include "../types.cuh"

namespace cellshard {
namespace sparse {

struct alignas(16) coo {
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;

    types::idx_t *rowIdx;
    types::idx_t *colIdx;
    real::storage_t *val;
};

__host__ __device__ __forceinline__ void init(
    coo * __restrict__ m,
    types::dim_t rows = 0,
    types::dim_t cols = 0,
    types::nnz_t nnz = 0
) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->rowIdx = 0;
    m->colIdx = 0;
    m->val = 0;
}

__host__ __device__ __forceinline__ std::size_t bytes(const coo * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) m->nnz * sizeof(types::idx_t)
        + (std::size_t) m->nnz * sizeof(types::idx_t)
        + (std::size_t) m->nnz * sizeof(real::storage_t);
}

__host__ __device__ __forceinline__ const real::storage_t *at(const coo * __restrict__ m, types::idx_t r, types::idx_t c) {
    for (types::nnz_t i = 0; i < m->nnz; ++i) {
        if (m->rowIdx[i] == r && m->colIdx[i] == c) return m->val + i;
    }
    return 0;
}

__host__ __device__ __forceinline__ real::storage_t *at(coo * __restrict__ m, types::idx_t r, types::idx_t c) {
    for (types::nnz_t i = 0; i < m->nnz; ++i) {
        if (m->rowIdx[i] == r && m->colIdx[i] == c) return m->val + i;
    }
    return 0;
}

} // namespace sparse
} // namespace cellshard
