#pragma once

#include <cstddef>
#include <cstdlib>

namespace cellshard {

struct alignas(16) dense {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;

    __half *val;
    unsigned int ld;
};

__host__ __device__ __forceinline__ void init(dense * __restrict__ m, unsigned int rows = 0, unsigned int cols = 0) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = rows * cols;
    m->val = 0;
    m->ld = cols;
}

__host__ __device__ __forceinline__ std::size_t bytes(const dense * __restrict__ m) {
    return sizeof(*m) + (std::size_t) m->nnz * sizeof(__half);
}

__host__ __forceinline__ void clear(dense * __restrict__ m) {
    std::free(m->val);
    m->val = 0;
    m->ld = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
}

__host__ __forceinline__ int allocate(dense * __restrict__ m) {
    std::free(m->val);
    m->val = 0;
    m->nnz = m->rows * m->cols;
    m->ld = m->cols;
    if (m->nnz == 0) return 1;
    m->val = (__half *) std::malloc((std::size_t) m->nnz * sizeof(__half));
    return m->val != 0;
}

__host__ __device__ __forceinline__ const __half *at(const dense * __restrict__ m, unsigned int r, unsigned int c) {
    return m->val + r * m->ld + c;
}

__host__ __device__ __forceinline__ __half *at(dense * __restrict__ m, unsigned int r, unsigned int c) {
    return m->val + r * m->ld + c;
}

} // namespace cellshard
