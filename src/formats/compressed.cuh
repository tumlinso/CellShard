#pragma once

#include "../types.cuh"

namespace cellshard {
namespace sparse {

struct alignas(16) csr {
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;

    types::ptr_t *rowPtr;
    types::idx_t *colIdx;
    real::storage_t *val;
};

__host__ __device__ __forceinline__ void init(
    csr * __restrict__ m,
    types::dim_t rows = 0,
    types::dim_t cols = 0,
    types::nnz_t nnz = 0
) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->rowPtr = 0;
    m->colIdx = 0;
    m->val = 0;
}

__host__ __device__ __forceinline__ std::size_t bytes(const csr * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) (m->rows + 1) * sizeof(types::ptr_t)
        + (std::size_t) m->nnz * sizeof(types::idx_t)
        + (std::size_t) m->nnz * sizeof(real::storage_t);
}

__host__ __device__ __forceinline__ const real::storage_t *at(const csr * __restrict__ m, types::dim_t r, types::idx_t c) {
    const types::ptr_t begin = m->rowPtr[r];
    const types::ptr_t end = m->rowPtr[r + 1];
    for (types::ptr_t i = begin; i < end; ++i) {
        if (m->colIdx[i] == c) return m->val + i;
    }
    return 0;
}

__host__ __device__ __forceinline__ real::storage_t *at(csr * __restrict__ m, types::dim_t r, types::idx_t c) {
    const types::ptr_t begin = m->rowPtr[r];
    const types::ptr_t end = m->rowPtr[r + 1];
    for (types::ptr_t i = begin; i < end; ++i) {
        if (m->colIdx[i] == c) return m->val + i;
    }
    return 0;
}

struct alignas(16) csc {
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;

    types::ptr_t *colPtr;
    types::idx_t *rowIdx;
    real::storage_t *val;
};

__host__ __device__ __forceinline__ void init(
    csc * __restrict__ m,
    types::dim_t rows = 0,
    types::dim_t cols = 0,
    types::nnz_t nnz = 0
) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->colPtr = 0;
    m->rowIdx = 0;
    m->val = 0;
}

__host__ __device__ __forceinline__ std::size_t bytes(const csc * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) (m->cols + 1) * sizeof(types::ptr_t)
        + (std::size_t) m->nnz * sizeof(types::idx_t)
        + (std::size_t) m->nnz * sizeof(real::storage_t);
}

__host__ __device__ __forceinline__ const real::storage_t *at(const csc * __restrict__ m, types::idx_t r, types::dim_t c) {
    const types::ptr_t begin = m->colPtr[c];
    const types::ptr_t end = m->colPtr[c + 1];
    for (types::ptr_t i = begin; i < end; ++i) {
        if (m->rowIdx[i] == r) return m->val + i;
    }
    return 0;
}

__host__ __device__ __forceinline__ real::storage_t *at(csc * __restrict__ m, types::idx_t r, types::dim_t c) {
    const types::ptr_t begin = m->colPtr[c];
    const types::ptr_t end = m->colPtr[c + 1];
    for (types::ptr_t i = begin; i < end; ++i) {
        if (m->rowIdx[i] == r) return m->val + i;
    }
    return 0;
}

struct alignas(16) csx {
    types::dim_t cDim;
    types::dim_t uDim;
    types::nnz_t nnz;

    types::ptr_t *cAxPtr;
    types::idx_t *uAxIdx;
    real::storage_t *val;
};

__host__ __device__ __forceinline__ void init(
    csx * __restrict__ m,
    types::dim_t cDim = 0,
    types::dim_t uDim = 0
) {
    m->cDim = cDim;
    m->uDim = uDim;
    m->nnz = 0;
    m->cAxPtr = 0;
    m->uAxIdx = 0;
    m->val = 0;
}

__host__ __device__ __forceinline__ std::size_t bytes(const csx * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) (m->cDim + 1) * sizeof(types::ptr_t)
        + (std::size_t) m->nnz * sizeof(types::idx_t)
        + (std::size_t) m->nnz * sizeof(real::storage_t);
}

} // namespace sparse
} // namespace cellshard
