#pragma once

#include "../types.cuh"

#include <cstdlib>

namespace cellshard {
namespace sparse {

enum {
    compressed_by_row = 0,
    compressed_by_col = 1
};

struct alignas(16) compressed {
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;
    types::u32 axis;

    types::ptr_t *majorPtr;
    types::idx_t *minorIdx;
    real::storage_t *val;
};

__host__ __device__ __forceinline__ types::dim_t major_dim(const compressed * __restrict__ m) {
    return m->axis == compressed_by_col ? m->cols : m->rows;
}

__host__ __device__ __forceinline__ types::dim_t minor_dim(const compressed * __restrict__ m) {
    return m->axis == compressed_by_col ? m->rows : m->cols;
}

__host__ __device__ __forceinline__ void init(
    compressed * __restrict__ m,
    types::dim_t rows = 0,
    types::dim_t cols = 0,
    types::nnz_t nnz = 0,
    types::u32 axis = compressed_by_row
) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->axis = axis;
    m->majorPtr = 0;
    m->minorIdx = 0;
    m->val = 0;
}

__host__ __device__ __forceinline__ std::size_t bytes(const compressed * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) (major_dim(m) + 1) * sizeof(types::ptr_t)
        + (std::size_t) m->nnz * sizeof(types::idx_t)
        + (std::size_t) m->nnz * sizeof(real::storage_t);
}

__host__ __device__ __forceinline__ const real::storage_t *at(const compressed * __restrict__ m, types::dim_t r, types::idx_t c) {
    const types::dim_t major = m->axis == compressed_by_col ? c : r;
    const types::idx_t minor = m->axis == compressed_by_col ? r : c;
    const types::ptr_t begin = m->majorPtr[major];
    const types::ptr_t end = m->majorPtr[major + 1];
    for (types::ptr_t i = begin; i < end; ++i) {
        if (m->minorIdx[i] == minor) return m->val + i;
    }
    return 0;
}

__host__ __device__ __forceinline__ real::storage_t *at(compressed * __restrict__ m, types::dim_t r, types::idx_t c) {
    const types::dim_t major = m->axis == compressed_by_col ? c : r;
    const types::idx_t minor = m->axis == compressed_by_col ? r : c;
    const types::ptr_t begin = m->majorPtr[major];
    const types::ptr_t end = m->majorPtr[major + 1];
    for (types::ptr_t i = begin; i < end; ++i) {
        if (m->minorIdx[i] == minor) return m->val + i;
    }
    return 0;
}

__host__ __forceinline__ void clear(compressed * __restrict__ m) {
    std::free(m->majorPtr);
    std::free(m->minorIdx);
    std::free(m->val);
    m->majorPtr = 0;
    m->minorIdx = 0;
    m->val = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->axis = compressed_by_row;
}

__host__ __forceinline__ int allocate(compressed * __restrict__ m) {
    const std::size_t ptr_count = (std::size_t) major_dim(m) + 1;

    std::free(m->majorPtr);
    std::free(m->minorIdx);
    std::free(m->val);
    m->majorPtr = 0;
    m->minorIdx = 0;
    m->val = 0;

    if (ptr_count != 0) m->majorPtr = (types::ptr_t *) std::malloc(ptr_count * sizeof(types::ptr_t));
    if (m->nnz != 0) {
        m->minorIdx = (types::idx_t *) std::malloc((std::size_t) m->nnz * sizeof(types::idx_t));
        m->val = (real::storage_t *) std::malloc((std::size_t) m->nnz * sizeof(real::storage_t));
    }

    if (ptr_count != 0 && m->majorPtr == 0) return 0;
    if (m->nnz != 0 && (m->minorIdx == 0 || m->val == 0)) {
        std::free(m->majorPtr);
        std::free(m->minorIdx);
        std::free(m->val);
        m->majorPtr = 0;
        m->minorIdx = 0;
        m->val = 0;
        return 0;
    }

    return 1;
}

} // namespace sparse
} // namespace cellshard
