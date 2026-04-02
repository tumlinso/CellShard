#pragma once

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cellshard {
namespace convert {
namespace kernels {

// Count COO entries per compressed-axis bucket into ptr[1..cDim].
// Atomic pressure depends directly on how skewed the compressed-axis index
// distribution is.
__global__ static void shift_ptr_idx_count(
    const unsigned int nnz,
    const unsigned int * __restrict__ axIdx,
    unsigned int * __restrict__ axPtr_shifted
) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int i = tid;
    while (i < nnz) {
        atomicAdd(axPtr_shifted + axIdx[i] + 1, 1u);
        i += stride;
    }
}

// Copy scanned pointer starts into mutable scatter heads.
// heads is intentionally mutable because scatter consumes it with atomicAdd.
__global__ static void init_cs_scatter_heads(
    const unsigned int cDim,
    const unsigned int * __restrict__ cAxPtr,
    unsigned int * __restrict__ heads
) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int i = tid;
    while (i < cDim) {
        heads[i] = cAxPtr[i];
        i += stride;
    }
}

// Scatter uncompressed-axis indices and values into CSR/CSC order.
// AtomicAdd on heads chooses the final slot for each COO entry.
__global__ static void csScatter(
    const unsigned int nnz,
    const unsigned int * __restrict__ cAxIdx,
    const unsigned int * __restrict__ uAxIdx,
    const __half * __restrict__ val,
    unsigned int * __restrict__ heads,
    unsigned int * __restrict__ out_uAx,
    __half * __restrict__ out_val
) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int i = tid;
    while (i < nnz) {
        const unsigned int dst = atomicAdd(heads + cAxIdx[i], 1u);
        out_uAx[dst] = uAxIdx[i];
        out_val[dst] = val[i];
        i += stride;
    }
}

} // namespace kernels
} // namespace convert
} // namespace cellshard
