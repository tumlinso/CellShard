#pragma once

#include "../../matrix.cuh"

namespace cellshard {
namespace sparse {

struct alignas(16) dia {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned char format;

    int *offsets;
    __half *val;
    unsigned int num_diagonals;
};

__host__ __device__ __forceinline__ void init(dia * __restrict__ m, unsigned int rows = 0, unsigned int cols = 0, unsigned int nnz = 0) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->format = format_dia;
    m->offsets = 0;
    m->val = 0;
    m->num_diagonals = 0;
}

__host__ __device__ __forceinline__ std::size_t bytes(const dia * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) m->num_diagonals * sizeof(int)
        + (std::size_t) m->nnz * sizeof(__half);
}

__host__ __forceinline__ void clear(dia * __restrict__ m) {
    std::free(m->offsets);
    std::free(m->val);
    m->offsets = 0;
    m->val = 0;
    m->num_diagonals = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->format = format_dia;
}

__host__ __forceinline__ int allocate(dia * __restrict__ m) {
    std::free(m->offsets);
    std::free(m->val);
    m->offsets = 0;
    m->val = 0;
    if (m->num_diagonals != 0) m->offsets = (int *) std::malloc((std::size_t) m->num_diagonals * sizeof(int));
    if (m->nnz != 0) m->val = (__half *) std::malloc((std::size_t) m->nnz * sizeof(__half));
    if (m->num_diagonals != 0 && m->offsets == 0) return 0;
    if (m->nnz != 0 && m->val == 0) {
        std::free(m->offsets);
        m->offsets = 0;
        return 0;
    }
    return 1;
}

__host__ __device__ __forceinline__ const __half *at(const dia * __restrict__ m, unsigned int r, unsigned int c) {
    const int offset = (int) c - (int) r;
    for (unsigned int i = 0; i < m->num_diagonals; ++i) {
        if (m->offsets[i] == offset) return m->val + i * m->rows + r;
    }
    return 0;
}

__host__ __device__ __forceinline__ __half *at(dia * __restrict__ m, unsigned int r, unsigned int c) {
    const int offset = (int) c - (int) r;
    for (unsigned int i = 0; i < m->num_diagonals; ++i) {
        if (m->offsets[i] == offset) return m->val + i * m->rows + r;
    }
    return 0;
}

} // namespace sparse
} // namespace cellshard
