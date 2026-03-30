#pragma once

#include "../../matrix.cuh"

namespace cellshard {
namespace sparse {

struct alignas(16) coo {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned char format;

    unsigned int *rowIdx;
    unsigned int *colIdx;
    __half *val;
};

__host__ __device__ __forceinline__ void init(
    coo * __restrict__ m,
    unsigned int rows = 0,
    unsigned int cols = 0,
    unsigned int nnz = 0) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->format = format_coo;
    m->rowIdx = 0;
    m->colIdx = 0;
    m->val = 0;
}

__host__ __device__ __forceinline__ std::size_t bytes(const coo * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) m->nnz * sizeof(unsigned int)
        + (std::size_t) m->nnz * sizeof(unsigned int)
        + (std::size_t) m->nnz * sizeof(__half);
}

__host__ __forceinline__ void clear(coo * __restrict__ m) {
    std::free(m->rowIdx);
    std::free(m->colIdx);
    std::free(m->val);
    m->rowIdx = 0;
    m->colIdx = 0;
    m->val = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->format = format_coo;
}

__host__ __forceinline__ int allocate(coo * __restrict__ m) {
    std::free(m->rowIdx);
    std::free(m->colIdx);
    std::free(m->val);
    m->rowIdx = 0;
    m->colIdx = 0;
    m->val = 0;
    if (m->nnz == 0) return 1;
    m->rowIdx = (unsigned int *) std::malloc((std::size_t) m->nnz * sizeof(unsigned int));
    m->colIdx = (unsigned int *) std::malloc((std::size_t) m->nnz * sizeof(unsigned int));
    m->val = (__half *) std::malloc((std::size_t) m->nnz * sizeof(__half));
    if (m->rowIdx == 0 || m->colIdx == 0 || m->val == 0) {
        std::free(m->rowIdx);
        std::free(m->colIdx);
        std::free(m->val);
        m->rowIdx = 0;
        m->colIdx = 0;
        m->val = 0;
        return 0;
    }
    return 1;
}

__host__ __device__ __forceinline__ const __half *at(const coo * __restrict__ m, unsigned int r, unsigned int c) {
    for (unsigned int i = 0; i < m->nnz; ++i) {
        if (m->rowIdx[i] == r && m->colIdx[i] == c) return m->val + i;
    }
    return 0;
}

__host__ __device__ __forceinline__ __half *at(coo * __restrict__ m, unsigned int r, unsigned int c) {
    for (unsigned int i = 0; i < m->nnz; ++i) {
        if (m->rowIdx[i] == r && m->colIdx[i] == c) return m->val + i;
    }
    return 0;
}

__host__ __forceinline__ int concatenate_rows(coo * __restrict__ dst, const coo * __restrict__ top, const coo * __restrict__ bottom) {
    if (top->cols != 0 && bottom->cols != 0 && top->cols != bottom->cols) {
        std::fprintf(stderr, "Error: cannot concatenate coo matrices with different column counts\n");
        return 0;
    }

    unsigned int oldRows = top->rows;
    dst->rows = top->rows + bottom->rows;
    dst->cols = top->cols != 0 ? top->cols : bottom->cols;
    dst->nnz = top->nnz + bottom->nnz;
    dst->format = format_coo;
    dst->rowIdx = 0;
    dst->colIdx = 0;
    dst->val = 0;
    if (!allocate(dst)) return 0;

    if (top->nnz != 0) {
        std::memcpy(dst->rowIdx, top->rowIdx, top->nnz * sizeof(unsigned int));
        std::memcpy(dst->colIdx, top->colIdx, top->nnz * sizeof(unsigned int));
        std::memcpy(dst->val, top->val, top->nnz * sizeof(__half));
    }
    for (unsigned int i = 0; i < bottom->nnz; ++i) dst->rowIdx[top->nnz + i] = bottom->rowIdx[i] + oldRows;
    if (bottom->nnz != 0) {
        std::memcpy(dst->colIdx + top->nnz, bottom->colIdx, bottom->nnz * sizeof(unsigned int));
        std::memcpy(dst->val + top->nnz, bottom->val, bottom->nnz * sizeof(__half));
    }
    return 1;
}

__host__ __forceinline__ int append_rows(coo * __restrict__ dst, const coo * __restrict__ src) {
    if (dst->cols != 0 && src->cols != 0 && dst->cols != src->cols) {
        std::fprintf(stderr, "Error: cannot concatenate coo matrices with different column counts\n");
        return 0;
    }

    unsigned int oldRows = dst->rows;
    unsigned int oldNnz = dst->nnz;
    unsigned int newNnz = dst->nnz + src->nnz;
    unsigned int *rowIdx = 0;
    unsigned int *colIdx = 0;
    __half *val = 0;

    dst->rows += src->rows;
    if (dst->cols == 0) dst->cols = src->cols;
    dst->nnz = newNnz;
    dst->format = format_coo;
    if (newNnz != 0) {
        rowIdx = (unsigned int *) std::malloc((std::size_t) newNnz * sizeof(unsigned int));
        colIdx = (unsigned int *) std::malloc((std::size_t) newNnz * sizeof(unsigned int));
        val = (__half *) std::malloc((std::size_t) newNnz * sizeof(__half));
        if (rowIdx == 0 || colIdx == 0 || val == 0) {
            std::free(rowIdx);
            std::free(colIdx);
            std::free(val);
            dst->rows = oldRows;
            dst->nnz = oldNnz;
            return 0;
        }
    }

    if (oldNnz != 0) {
        std::memcpy(rowIdx, dst->rowIdx, (std::size_t) oldNnz * sizeof(unsigned int));
        std::memcpy(colIdx, dst->colIdx, (std::size_t) oldNnz * sizeof(unsigned int));
        std::memcpy(val, dst->val, (std::size_t) oldNnz * sizeof(__half));
    }
    std::free(dst->rowIdx);
    std::free(dst->colIdx);
    std::free(dst->val);
    dst->rowIdx = rowIdx;
    dst->colIdx = colIdx;
    dst->val = val;

    for (unsigned int i = 0; i < src->nnz; ++i) dst->rowIdx[oldNnz + i] = src->rowIdx[i] + oldRows;
    if (src->nnz != 0) {
        std::memcpy(dst->colIdx + oldNnz, src->colIdx, src->nnz * sizeof(unsigned int));
        std::memcpy(dst->val + oldNnz, src->val, src->nnz * sizeof(__half));
    }
    return 1;
}

} // namespace sparse
} // namespace cellshard
