#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cellshard {
namespace convert {
namespace kernels {

// Expand CSR/CSC-style pointer ranges into COO compressed-axis indices.
// One block walks one or more compressed-axis ranges and writes contiguous runs.
__global__ static void csExpandToCoo(
    const unsigned int cDim,
    const unsigned int * __restrict__ cAxPtr,
    const unsigned int * __restrict__ uAxIdx,
    const __half * __restrict__ val,
    unsigned int * __restrict__ out_cAxIdx,
    unsigned int * __restrict__ out_uAxIdx,
    __half * __restrict__ out_val
) {
    unsigned int c = (unsigned int) blockIdx.x;
    const unsigned int cStride = (unsigned int) gridDim.x;

    while (c < cDim) {
        const unsigned int begin = cAxPtr[c];
        const unsigned int end = cAxPtr[c + 1];
        unsigned int i = begin + (unsigned int) threadIdx.x;

        while (i < end) {
            out_cAxIdx[i] = c;
            out_uAxIdx[i] = uAxIdx[i];
            out_val[i] = val[i];
            i += (unsigned int) blockDim.x;
        }
        c += cStride;
    }
}

// Fallback path when the compressed axis is too small to expose enough blocks.
// Each thread resolves its own COO compressed-axis index by upper-bound search.
__global__ static void csSearchToCoo(
    const unsigned int cDim,
    const unsigned int nnz,
    const unsigned int * __restrict__ cAxPtr,
    const unsigned int * __restrict__ uAxIdx,
    const __half * __restrict__ val,
    unsigned int * __restrict__ out_cAxIdx,
    unsigned int * __restrict__ out_uAxIdx,
    __half * __restrict__ out_val
) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int i = tid;

    while (i < nnz) {
        unsigned int lo = 0;
        unsigned int hi = cDim;

        while (lo < hi) {
            const unsigned int mid = lo + ((hi - lo) >> 1);
            if (i >= cAxPtr[mid + 1]) lo = mid + 1;
            else hi = mid;
        }

        out_cAxIdx[i] = lo;
        out_uAxIdx[i] = uAxIdx[i];
        out_val[i] = val[i];
        i += stride;
    }
}

} // namespace kernels
} // namespace convert
} // namespace cellshard