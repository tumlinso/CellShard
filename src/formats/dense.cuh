#pragma once

#include "../types.cuh"

namespace cellshard {

struct alignas(16) dense {
    types::dim_t rows;
    types::dim_t cols;
    real::storage_t *val;
};

__host__ __device__ __forceinline__ void init(
    dense * __restrict__ m,
    types::dim_t rows = 0,
    types::dim_t cols = 0
) {
    m->rows = rows;
    m->cols = cols;
    m->val = 0;
}

__host__ __device__ __forceinline__ std::size_t bytes(const dense * __restrict__ m) {
    return sizeof(*m) + (std::size_t) m->rows * (std::size_t) m->cols * sizeof(real::storage_t);
}

__host__ __device__ __forceinline__ const real::storage_t *at(const dense * __restrict__ m, types::dim_t r, types::dim_t c) {
    return m->val + (std::size_t) r * (std::size_t) m->cols + c;
}

__host__ __device__ __forceinline__ real::storage_t *at(dense * __restrict__ m, types::dim_t r, types::dim_t c) {
    return m->val + (std::size_t) r * (std::size_t) m->cols + c;
}

} // namespace cellshard
