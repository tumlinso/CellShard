#pragma once

#include "../types.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cellshard {
namespace sparse {

// COO triplet layout.
// This is the most natural ingest layout and the worst random-access layout.
struct alignas(16) coo {
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;

    types::idx_t *rowIdx;
    types::idx_t *colIdx;
    real::storage_t *val;
};

// Metadata-only init.
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

// Total resident host bytes for one materialized COO part.
__host__ __device__ __forceinline__ std::size_t bytes(const coo * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) m->nnz * sizeof(types::idx_t)
        + (std::size_t) m->nnz * sizeof(types::idx_t)
        + (std::size_t) m->nnz * sizeof(real::storage_t);
}

// O(nnz) random access helper. Useful for validation, not throughput.
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

// Release all host arrays and reset metadata.
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
}

// allocate() discards any previous payload and allocates all three arrays.
__host__ __forceinline__ int allocate(coo * __restrict__ m) {
    std::free(m->rowIdx);
    std::free(m->colIdx);
    std::free(m->val);
    m->rowIdx = 0;
    m->colIdx = 0;
    m->val = 0;
    if (m->nnz == 0) return 1;
    m->rowIdx = (types::idx_t *) std::malloc((std::size_t) m->nnz * sizeof(types::idx_t));
    m->colIdx = (types::idx_t *) std::malloc((std::size_t) m->nnz * sizeof(types::idx_t));
    m->val = (real::storage_t *) std::malloc((std::size_t) m->nnz * sizeof(real::storage_t));
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

// Fresh destination concatenate. This is a full host-side copy of both inputs.
__host__ __forceinline__ int concatenate_rows(coo * __restrict__ dst, const coo * __restrict__ top, const coo * __restrict__ bottom) {
    if (top->cols != 0 && bottom->cols != 0 && top->cols != bottom->cols) {
        std::fprintf(stderr, "Error: cannot concatenate coo matrices with different column counts\n");
        return 0;
    }

    const types::dim_t oldRows = top->rows;
    dst->rows = top->rows + bottom->rows;
    dst->cols = top->cols != 0 ? top->cols : bottom->cols;
    dst->nnz = top->nnz + bottom->nnz;
    dst->rowIdx = 0;
    dst->colIdx = 0;
    dst->val = 0;
    if (!allocate(dst)) return 0;

    if (top->nnz != 0) {
        std::memcpy(dst->rowIdx, top->rowIdx, (std::size_t) top->nnz * sizeof(types::idx_t));
        std::memcpy(dst->colIdx, top->colIdx, (std::size_t) top->nnz * sizeof(types::idx_t));
        std::memcpy(dst->val, top->val, (std::size_t) top->nnz * sizeof(real::storage_t));
    }
    for (types::nnz_t i = 0; i < bottom->nnz; ++i) dst->rowIdx[top->nnz + i] = bottom->rowIdx[i] + oldRows;
    if (bottom->nnz != 0) {
        std::memcpy(dst->colIdx + top->nnz, bottom->colIdx, (std::size_t) bottom->nnz * sizeof(types::idx_t));
        std::memcpy(dst->val + top->nnz, bottom->val, (std::size_t) bottom->nnz * sizeof(real::storage_t));
    }
    return 1;
}

// In-place append by allocate-copy-append. This is intentionally explicit
// because it is expensive.
__host__ __forceinline__ int append_rows(coo * __restrict__ dst, const coo * __restrict__ src) {
    if (dst->cols != 0 && src->cols != 0 && dst->cols != src->cols) {
        std::fprintf(stderr, "Error: cannot concatenate coo matrices with different column counts\n");
        return 0;
    }

    const types::dim_t oldRows = dst->rows;
    const types::nnz_t oldNnz = dst->nnz;
    const types::nnz_t newNnz = dst->nnz + src->nnz;
    types::idx_t *rowIdx = 0;
    types::idx_t *colIdx = 0;
    real::storage_t *val = 0;

    dst->rows += src->rows;
    if (dst->cols == 0) dst->cols = src->cols;
    dst->nnz = newNnz;
    if (newNnz != 0) {
        rowIdx = (types::idx_t *) std::malloc((std::size_t) newNnz * sizeof(types::idx_t));
        colIdx = (types::idx_t *) std::malloc((std::size_t) newNnz * sizeof(types::idx_t));
        val = (real::storage_t *) std::malloc((std::size_t) newNnz * sizeof(real::storage_t));
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
        std::memcpy(rowIdx, dst->rowIdx, (std::size_t) oldNnz * sizeof(types::idx_t));
        std::memcpy(colIdx, dst->colIdx, (std::size_t) oldNnz * sizeof(types::idx_t));
        std::memcpy(val, dst->val, (std::size_t) oldNnz * sizeof(real::storage_t));
    }
    std::free(dst->rowIdx);
    std::free(dst->colIdx);
    std::free(dst->val);
    dst->rowIdx = rowIdx;
    dst->colIdx = colIdx;
    dst->val = val;

    for (types::nnz_t i = 0; i < src->nnz; ++i) dst->rowIdx[oldNnz + i] = src->rowIdx[i] + oldRows;
    if (src->nnz != 0) {
        std::memcpy(dst->colIdx + oldNnz, src->colIdx, (std::size_t) src->nnz * sizeof(types::idx_t));
        std::memcpy(dst->val + oldNnz, src->val, (std::size_t) src->nnz * sizeof(real::storage_t));
    }
    return 1;
}

} // namespace sparse
} // namespace cellshard
