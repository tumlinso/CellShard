#pragma once

#include "../../matrix.cuh"

namespace cellshard {
namespace sparse {

struct alignas(16) csr {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;

    unsigned int *rowPtr;
    unsigned int *colIdx;
    __half *val;
};

__host__ __device__ __forceinline__ void init(
    csr * __restrict__ m,
    unsigned int rows = 0,
    unsigned int cols = 0,
    unsigned int nnz = 0
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
        + (std::size_t) (m->rows + 1) * sizeof(unsigned int)
        + (std::size_t) m->nnz * sizeof(unsigned int)
        + (std::size_t) m->nnz * sizeof(__half);
}

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

    if (m->rows != 0) m->rowPtr = (unsigned int *) std::malloc((std::size_t) (m->rows + 1) * sizeof(unsigned int));
    if (m->nnz != 0) {
        m->colIdx = (unsigned int *) std::malloc((std::size_t) m->nnz * sizeof(unsigned int));
        m->val = (__half *) std::malloc((std::size_t) m->nnz * sizeof(__half));
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

__host__ __device__ __forceinline__ const __half *at(const csr * __restrict__ m, unsigned int r, unsigned int c) {
    const unsigned int begin = m->rowPtr[r];
    const unsigned int end = m->rowPtr[r + 1];
    for (unsigned int i = begin; i < end; ++i) {
        if (m->colIdx[i] == c) return m->val + i;
    }
    return 0;
}

__host__ __device__ __forceinline__ __half *at(csr * __restrict__ m, unsigned int r, unsigned int c) {
    const unsigned int begin = m->rowPtr[r];
    const unsigned int end = m->rowPtr[r + 1];
    for (unsigned int i = begin; i < end; ++i) {
        if (m->colIdx[i] == c) return m->val + i;
    }
    return 0;
}

} // namespace sparse
} // namespace cellshard
