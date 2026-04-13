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
#include "transpose/count_cs_transpose_targets.cuh"

// Scatter transposed entries into the already-scanned output structure.
#include "transpose/scatter_cs_transpose.cuh"

// Entrywise COO transpose kernel.
#include "transpose/transpose_coo_entries.cuh"

} // namespace kernels

} // namespace convert
} // namespace cellshard
