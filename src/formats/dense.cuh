#pragma once

#include "../types.cuh"

#include <cstdlib>

namespace cellshard {

// Flat row-major dense part layout with one heap allocation for values.
struct alignas(16) dense {
    types::dim_t rows;
    types::dim_t cols;
    real::storage_t *val;
};

// Metadata-only init. No allocation happens here.
__host__ __device__ __forceinline__ void init(
    dense * __restrict__ m,
    types::dim_t rows = 0,
    types::dim_t cols = 0
) {
    m->rows = rows;
    m->cols = cols;
    m->val = 0;
}

// Total resident host bytes for one materialized dense part.
__host__ __device__ __forceinline__ std::size_t bytes(const dense * __restrict__ m) {
    return sizeof(*m) + (std::size_t) m->rows * (std::size_t) m->cols * sizeof(real::storage_t);
}

// Direct row-major addressing with no bounds checks.
__host__ __device__ __forceinline__ const real::storage_t *at(const dense * __restrict__ m, types::dim_t r, types::dim_t c) {
    return m->val + (std::size_t) r * (std::size_t) m->cols + c;
}

__host__ __device__ __forceinline__ real::storage_t *at(dense * __restrict__ m, types::dim_t r, types::dim_t c) {
    return m->val + (std::size_t) r * (std::size_t) m->cols + c;
}

// Release host storage and reset metadata.
__host__ __forceinline__ void clear(dense * __restrict__ m) {
    std::free(m->val);
    m->val = 0;
    m->rows = 0;
    m->cols = 0;
}

// allocate() discards any previous payload and allocates a fresh host buffer
// sized from rows*cols.
__host__ __forceinline__ int allocate(dense * __restrict__ m) {
    const std::size_t count = (std::size_t) m->rows * (std::size_t) m->cols;

    std::free(m->val);
    m->val = 0;
    if (count == 0) return 1;
    m->val = (real::storage_t *) std::malloc(count * sizeof(real::storage_t));
    return m->val != 0;
}

} // namespace cellshard
