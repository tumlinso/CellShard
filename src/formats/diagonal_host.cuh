#pragma once

#include "diagonal.cuh"

#include <cstdlib>

namespace cellshard {
namespace sparse {

__host__ __forceinline__ void clear(dia * __restrict__ m) {
    std::free(m->offsets);
    std::free(m->val);
    m->offsets = 0;
    m->val = 0;
    m->num_diagonals = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
}

__host__ __forceinline__ int allocate(dia * __restrict__ m) {
    std::free(m->offsets);
    std::free(m->val);
    m->offsets = 0;
    m->val = 0;
    if (m->num_diagonals != 0) m->offsets = (int *) std::malloc((std::size_t) m->num_diagonals * sizeof(int));
    if (m->nnz != 0) m->val = (real::storage_t *) std::malloc((std::size_t) m->nnz * sizeof(real::storage_t));
    if (m->num_diagonals != 0 && m->offsets == 0) return 0;
    if (m->nnz != 0 && m->val == 0) {
        std::free(m->offsets);
        m->offsets = 0;
        return 0;
    }
    return 1;
}

} // namespace sparse
} // namespace cellshard
