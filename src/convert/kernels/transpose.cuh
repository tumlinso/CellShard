#pragma once

#include <cstddef>
#include <cstdio>

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "csScatter.cuh"

#include "../../types.cuh"

namespace cellshard {
namespace convert {
namespace kernels {

// One warp lane dimension handles entries inside a segment; one warp-count
// dimension handles many segments per block.
enum {
    transpose_lanes = 32,
    transpose_warps = 8
};

// Direct CUDA error reporting helper.
static inline int transpose_cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

// Launch geometry for compressed transpose.
static inline void setup_cs_transpose_launch(
    const types::dim_t cDim,
    dim3 *grid,
    dim3 *block
) {
    block->x = transpose_lanes;
    block->y = transpose_warps;
    block->z = 1;
    grid->x = (unsigned int) ((cDim + transpose_warps - 1u) / transpose_warps);
    grid->y = 1;
    grid->z = 1;
    if (grid->x < 1u) grid->x = 1u;
    if (grid->x > 4096u) grid->x = 4096u;
}

// Count target compressed-axis populations for the transposed output.
__global__ static void count_cs_transpose_targets(
    const types::dim_t cDim,
    const types::ptr_t * __restrict__ cAxPtr,
    const types::idx_t * __restrict__ uAxIdx,
    types::ptr_t * __restrict__ out_uAxPtr_shifted
) {
    types::dim_t seg = (types::dim_t) blockIdx.x * blockDim.y + (types::dim_t) threadIdx.y;
    const types::dim_t seg_stride = (types::dim_t) gridDim.x * blockDim.y;

    while (seg < cDim) {
        const types::ptr_t begin = cAxPtr[seg];
        const types::ptr_t end = cAxPtr[seg + 1];
        types::ptr_t i = begin + (types::ptr_t) threadIdx.x;
        while (i < end) {
            atomicAdd(out_uAxPtr_shifted + uAxIdx[i] + 1u, 1u);
            i += (types::ptr_t) blockDim.x;
        }
        seg += seg_stride;
    }
}

// Scatter transposed entries into the already-scanned output structure.
__global__ static void scatter_cs_transpose(
    const types::dim_t cDim,
    const types::ptr_t * __restrict__ cAxPtr,
    const types::idx_t * __restrict__ uAxIdx,
    const real::storage_t * __restrict__ val,
    types::ptr_t * __restrict__ heads,
    types::idx_t * __restrict__ out_cAxIdx,
    real::storage_t * __restrict__ out_val
) {
    types::dim_t seg = (types::dim_t) blockIdx.x * blockDim.y + (types::dim_t) threadIdx.y;
    const types::dim_t seg_stride = (types::dim_t) gridDim.x * blockDim.y;

    while (seg < cDim) {
        const types::ptr_t begin = cAxPtr[seg];
        const types::ptr_t end = cAxPtr[seg + 1];
        types::ptr_t i = begin + (types::ptr_t) threadIdx.x;
        while (i < end) {
            const types::idx_t dst_seg = uAxIdx[i];
            const types::ptr_t dst = atomicAdd(heads + dst_seg, 1u);
            out_cAxIdx[dst] = (types::idx_t) seg;
            out_val[dst] = val[i];
            i += (types::ptr_t) blockDim.x;
        }
        seg += seg_stride;
    }
}

// Entrywise COO transpose kernel.
template<typename ValueT>
__global__ static void transpose_coo_entries(
    const types::nnz_t nnz,
    const types::idx_t * __restrict__ src_rowIdx,
    const types::idx_t * __restrict__ src_colIdx,
    const ValueT * __restrict__ src_val,
    types::idx_t * __restrict__ dst_rowIdx,
    types::idx_t * __restrict__ dst_colIdx,
    ValueT * __restrict__ dst_val
) {
    const types::nnz_t tid = (types::nnz_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::nnz_t stride = (types::nnz_t) (gridDim.x * blockDim.x);
    types::nnz_t i = tid;

    while (i < nnz) {
        const types::idx_t r = src_rowIdx[i];
        const types::idx_t c = src_colIdx[i];
        dst_rowIdx[i] = c;
        dst_colIdx[i] = r;
        if (dst_val != 0 && dst_val != src_val) dst_val[i] = src_val[i];
        i += stride;
    }
}

} // namespace kernels

} // namespace convert
} // namespace cellshard
