#pragma once

#include "compressed.cuh"

#include <cstdlib>

namespace cellshard {
namespace sparse {

__host__ __forceinline__ void clear(csr * __restrict__ m) {
    std::free(m->rowPtr);
    std::free(m->colIdx);
    std::free(m->val);
    m->rowPtr = 0;
    m->colIdx = 0;
    m->val = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
}

__host__ __forceinline__ int allocate(csr * __restrict__ m) {
    std::free(m->rowPtr);
    std::free(m->colIdx);
    std::free(m->val);
    m->rowPtr = 0;
    m->colIdx = 0;
    m->val = 0;

    if (m->rows != 0) m->rowPtr = (types::ptr_t *) std::malloc((std::size_t) (m->rows + 1) * sizeof(types::ptr_t));
    if (m->nnz != 0) {
        m->colIdx = (types::idx_t *) std::malloc((std::size_t) m->nnz * sizeof(types::idx_t));
        m->val = (real::storage_t *) std::malloc((std::size_t) m->nnz * sizeof(real::storage_t));
    }

    if (m->rows != 0 && m->rowPtr == 0) return 0;
    if (m->nnz != 0 && (m->colIdx == 0 || m->val == 0)) {
        std::free(m->rowPtr);
        std::free(m->colIdx);
        std::free(m->val);
        m->rowPtr = 0;
        m->colIdx = 0;
        m->val = 0;
        return 0;
    }

    return 1;
}

__host__ __forceinline__ void clear(csc * __restrict__ m) {
    std::free(m->colPtr);
    std::free(m->rowIdx);
    std::free(m->val);
    m->colPtr = 0;
    m->rowIdx = 0;
    m->val = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
}

__host__ __forceinline__ int allocate(csc * __restrict__ m) {
    std::free(m->colPtr);
    std::free(m->rowIdx);
    std::free(m->val);
    m->colPtr = 0;
    m->rowIdx = 0;
    m->val = 0;
    if (m->cols != 0) m->colPtr = (types::ptr_t *) std::malloc((std::size_t) (m->cols + 1) * sizeof(types::ptr_t));
    if (m->nnz != 0) {
        m->rowIdx = (types::idx_t *) std::malloc((std::size_t) m->nnz * sizeof(types::idx_t));
        m->val = (real::storage_t *) std::malloc((std::size_t) m->nnz * sizeof(real::storage_t));
    }
    if (m->cols != 0 && m->colPtr == 0) return 0;
    if (m->nnz != 0 && (m->rowIdx == 0 || m->val == 0)) {
        std::free(m->colPtr);
        std::free(m->rowIdx);
        std::free(m->val);
        m->colPtr = 0;
        m->rowIdx = 0;
        m->val = 0;
        return 0;
    }
    return 1;
}

__host__ __forceinline__ void clear(csx * __restrict__ m) {
    std::free(m->val);
    std::free(m->uAxIdx);
    std::free(m->cAxPtr);
    m->cAxPtr = 0;
    m->uAxIdx = 0;
    m->val = 0;
    m->cDim = 0;
    m->uDim = 0;
    m->nnz = 0;
}

__host__ __forceinline__ int allocate(csx * __restrict__ m) {
    std::free(m->val);
    std::free(m->uAxIdx);
    std::free(m->cAxPtr);
    m->cAxPtr = 0;
    m->uAxIdx = 0;
    m->val = 0;

    if (m->cDim != 0) {
        m->cAxPtr = (types::ptr_t *) std::malloc((std::size_t) (m->cDim + 1) * sizeof(types::ptr_t));
        if (m->cAxPtr == 0) return 0;
    }

    if (m->nnz != 0) {
        m->uAxIdx = (types::idx_t *) std::malloc((std::size_t) m->nnz * sizeof(types::idx_t));
        if (m->uAxIdx == 0) return 0;
        m->val = (real::storage_t *) std::malloc((std::size_t) m->nnz * sizeof(real::storage_t));
        if (m->val == 0) return 0;
    }

    return 1;
}

} // namespace sparse
} // namespace cellshard
