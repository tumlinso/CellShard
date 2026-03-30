#pragma once

#include "../matrix.cuh"

#include <cstddef>
#include <cstdlib>

namespace cellshard {
namespace sparse {

struct alignas(16) csx {
    types::dim_t cDim;
    types::dim_t uDim;
    types::nnz_t nnz;
    unsigned char format;

    types::ptr_t *cAxPtr;
    types::idx_t *uAxIdx;
    real::storage_t *val;
};

__host__ __device__ __forceinline__ void init(
    csx * __restrict__ m,
    types::dim_t cDim = 0,
    types::dim_t uDim = 0,
    unsigned char format = format_csr
) {
    m->cDim = cDim;
    m->uDim = uDim;
    m->nnz = 0;
    m->format = format;
    m->cAxPtr = 0;
    m->uAxIdx = 0;
    m->val = 0;
}

__host__ __device__ __forceinline__ std::size_t bytes(const csx * __restrict__ m) {
    return sizeof(*m)
         + (std::size_t) (m->cDim + 1) * sizeof(types::ptr_t)
         + (std::size_t) m->nnz * sizeof(types::idx_t)
         + (std::size_t) m->nnz * sizeof(real::storage_t);
}

__host__ __forceinline__ void clear(csx * __restrict__ m) {
    std::free(m->val);
    std::free(m->uAxIdx);
    std::free(m->cAxPtr);
    m->cAxPtr = 0;
    m->uAxIdx = 0;
    m->val = 0;
    m->cDim = 0;
    m->uDim = 0;
    m->nnz = 0;
    m->format = format_csr;
}

__host__ __forceinline__ int allocate(csx * __restrict__ m) {
    std::free(m->val);
    std::free(m->uAxIdx);
    std::free(m->cAxPtr);
    m->cAxPtr = 0;
    m->uAxIdx = 0;
    m->val = 0;

    if (m->cDim != 0) {
        m->cAxPtr = (types::ptr_t *) std::malloc((std::size_t) (m->cDim + 1) * sizeof(types::ptr_t));
        if (m->cAxPtr == 0) return 0;
    }

    if (m->nnz != 0) {
        m->uAxIdx = (types::idx_t *) std::malloc((std::size_t) m->nnz * sizeof(types::idx_t));
        if (m->uAxIdx == 0) return 0;
        m->val = (real::storage_t *) std::malloc((std::size_t) m->nnz * sizeof(real::storage_t));
        if (m->val == 0) return 0;
    }

    return 1;
}

} // namespace sparse
} // namespace cellshard
