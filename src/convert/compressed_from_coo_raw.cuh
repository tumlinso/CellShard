#pragma once

#include "kernels/csScatter.cuh"

#include <cstdio>

namespace cellshard {
namespace convert {

static inline int compressed_from_coo_cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

// Choose a 256-thread launch grid and clamp it to a sane range.
static inline void setup_compressed_from_coo_launch(
    const unsigned int cDim,
    const unsigned int nnz,
    int *blocks_nnz,
    int *blocks_cdim
) {
    *blocks_nnz = (int) ((nnz + 255u) >> 8);
    *blocks_cdim = (int) ((cDim + 255u) >> 8);

    if (*blocks_nnz < 1) *blocks_nnz = 1;
    if (*blocks_cdim < 1) *blocks_cdim = 1;
    if (*blocks_nnz > 4096) *blocks_nnz = 4096;
    if (*blocks_cdim > 4096) *blocks_cdim = 4096;
}

// Raw host-side launcher over already-allocated device buffers.
// No allocation, no copies, no ownership.
static inline int build_compressed_from_coo_raw(
    const unsigned int cDim,
    const unsigned int nnz,
    const unsigned int *d_cAxIdx,
    const unsigned int *d_uAxIdx,
    const __half *d_val,
    unsigned int *d_cAxPtr,
    unsigned int *d_heads,
    unsigned int *d_out_uAx,
    __half *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream
) {
    int blocks_nnz = 0;
    int blocks_cdim = 0;

    if (cudaMemsetAsync(d_cAxPtr, 0, (std::size_t) (cDim + 1) * sizeof(unsigned int), stream) != cudaSuccess) return 0;
    if (nnz == 0) return 1;

    setup_compressed_from_coo_launch(cDim, nnz, &blocks_nnz, &blocks_cdim);

    cellshard::convert::kernels::shift_ptr_idx_count<<<blocks_nnz, 256, 0, stream>>>(nnz, d_cAxIdx, d_cAxPtr);
    if (cudaGetLastError() != cudaSuccess) return 0;

    if (cub::DeviceScan::ExclusiveSum(
            d_scan_tmp,
            scan_bytes,
            d_cAxPtr,
            d_cAxPtr,
            cDim + 1,
            stream) != cudaSuccess) return 0;

    cellshard::convert::kernels::init_cs_scatter_heads<<<blocks_cdim, 256, 0, stream>>>(cDim, d_cAxPtr, d_heads);
    if (cudaGetLastError() != cudaSuccess) return 0;

    cellshard::convert::kernels::csScatter<<<blocks_nnz, 256, 0, stream>>>(nnz, d_cAxIdx, d_uAxIdx, d_val, d_heads, d_out_uAx, d_out_val);
    if (cudaGetLastError() != cudaSuccess) return 0;

    return 1;
}

static inline int build_cs_from_coo_raw(
    const unsigned int cDim,
    const unsigned int nnz,
    const unsigned int *d_cAxIdx,
    const unsigned int *d_uAxIdx,
    const __half *d_val,
    unsigned int *d_cAxPtr,
    unsigned int *d_heads,
    unsigned int *d_out_uAx,
    __half *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream
) {
    return build_compressed_from_coo_raw(
        cDim,
        nnz,
        d_cAxIdx,
        d_uAxIdx,
        d_val,
        d_cAxPtr,
        d_heads,
        d_out_uAx,
        d_out_val,
        d_scan_tmp,
        scan_bytes,
        stream
    );
}

} // namespace convert
} // namespace cellshard
