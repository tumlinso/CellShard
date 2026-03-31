#pragma once

#include "dense.cuh"

#include <cstdlib>

namespace cellshard {

__host__ __forceinline__ void clear(dense * __restrict__ m) {
    std::free(m->val);
    m->val = 0;
    m->rows = 0;
    m->cols = 0;
}

__host__ __forceinline__ int allocate(dense * __restrict__ m) {
    const std::size_t count = (std::size_t) m->rows * (std::size_t) m->cols;

    std::free(m->val);
    m->val = 0;
    if (count == 0) return 1;
    m->val = (real::storage_t *) std::malloc(count * sizeof(real::storage_t));
    return m->val != 0;
}

} // namespace cellshard
