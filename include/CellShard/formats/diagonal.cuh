#pragma once

#include "../core/types.cuh"

#include <cstdlib>

namespace cellshard {
namespace sparse {

// DIA sparse layout: one offset per diagonal plus a packed diagonal-value slab.
struct alignas(16) dia {
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;

    // When non-null, this owns the packed host allocation containing offsets
    // and values. offsets/val point inside it.
    void *storage;
    int *offsets;
    real::storage_t *val;
    types::idx_t num_diagonals;
};

// Metadata-only init.
__host__ __device__ __forceinline__ void init(
    dia * __restrict__ m,
    types::dim_t rows = 0,
    types::dim_t cols = 0,
    types::nnz_t nnz = 0
) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->storage = 0;
    m->offsets = 0;
    m->val = 0;
    m->num_diagonals = 0;
}

// Total resident host bytes for one materialized DIA part.
__host__ __device__ __forceinline__ std::size_t bytes(const dia * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) m->num_diagonals * sizeof(int)
        + (std::size_t) m->nnz * sizeof(real::storage_t);
}

// Random access walks the diagonal-offset list and then indexes into the
// matching diagonal payload.
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

// Release host arrays and reset metadata.
__host__ __forceinline__ void clear(dia * __restrict__ m) {
    if (m->storage != 0) std::free(m->storage);
    else {
        std::free(m->offsets);
        std::free(m->val);
    }
    m->storage = 0;
    m->offsets = 0;
    m->val = 0;
    m->num_diagonals = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
}

// allocate() rebuilds offsets and values from current metadata.
__host__ __forceinline__ int allocate(dia * __restrict__ m) {
    const std::size_t offsets_bytes = (std::size_t) m->num_diagonals * sizeof(int);
    const std::size_t val_offset = ((offsets_bytes + alignof(real::storage_t) - 1u) / alignof(real::storage_t)) * alignof(real::storage_t);
    const std::size_t total_bytes = val_offset + (std::size_t) m->nnz * sizeof(real::storage_t);
    void *storage = 0;

    if (m->storage != 0) std::free(m->storage);
    else {
        std::free(m->offsets);
        std::free(m->val);
    }
    m->storage = 0;
    m->offsets = 0;
    m->val = 0;
    if (total_bytes == 0) return 1;
    storage = std::malloc(total_bytes);
    if (storage == 0) return 0;
    m->storage = storage;
    m->offsets = m->num_diagonals != 0 ? (int *) storage : 0;
    m->val = m->nnz != 0 ? (real::storage_t *) ((char *) storage + val_offset) : 0;
    return 1;
}

} // namespace sparse
} // namespace cellshard
