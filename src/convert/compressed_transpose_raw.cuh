#pragma once

#include "kernels/transpose.cuh"

namespace cellshard {
namespace convert {

// Raw compressed-transpose launcher over already-allocated device buffers.
// This is device-resident only:
// - no host staging
// - no host/device copies
// - caller owns scan scratch and output buffers
static inline int build_compressed_transpose_raw(
    const types::dim_t cDim,
    const types::dim_t uDim,
    const types::nnz_t nnz,
    const types::ptr_t *d_cAxPtr,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_out_uAxPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_cAxIdx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream
) {
    dim3 grid;
    dim3 block;

    if (!kernels::transpose_cuda_check(
            cudaMemsetAsync(d_out_uAxPtr, 0, (std::size_t) (uDim + 1) * sizeof(types::ptr_t), stream),
            "cudaMemsetAsync transpose out ptr")) return 0;

    if (nnz == 0) return 1;

    kernels::setup_cs_transpose_launch(cDim, &grid, &block);

    kernels::count_cs_transpose_targets<<<grid, block, 0, stream>>>(
        cDim,
        d_cAxPtr,
        d_uAxIdx,
        d_out_uAxPtr
    );
    if (cudaGetLastError() != cudaSuccess) return 0;

    if (!kernels::transpose_cuda_check(
            cub::DeviceScan::ExclusiveSum(
                d_scan_tmp,
                scan_bytes,
                d_out_uAxPtr,
                d_out_uAxPtr,
                uDim + 1,
                stream),
            "cub transpose scan")) return 0;

    cellshard::convert::kernels::init_cs_scatter_heads<<<grid, block, 0, stream>>>(
        uDim,
        d_out_uAxPtr,
        d_heads
    );
    if (cudaGetLastError() != cudaSuccess) return 0;

    kernels::scatter_cs_transpose<<<grid, block, 0, stream>>>(
        cDim,
        d_cAxPtr,
        d_uAxIdx,
        d_val,
        d_heads,
        d_out_cAxIdx,
        d_out_val
    );
    if (cudaGetLastError() != cudaSuccess) return 0;

    return 1;
}

// Alias kept for shorter call sites.
static inline int build_transpose_cs_from_cs_raw(
    const types::dim_t cDim,
    const types::dim_t uDim,
    const types::nnz_t nnz,
    const types::ptr_t *d_cAxPtr,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_out_uAxPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_cAxIdx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream
) {
    return build_compressed_transpose_raw(
        cDim,
        uDim,
        nnz,
        d_cAxPtr,
        d_uAxIdx,
        d_val,
        d_out_uAxPtr,
        d_heads,
        d_out_cAxIdx,
        d_out_val,
        d_scan_tmp,
        scan_bytes,
        stream
    );
}

// Entrywise COO transpose for already-device-resident COO buffers.
static inline int transpose_coo_entries_raw(
    const types::nnz_t nnz,
    const types::idx_t *d_src_rowIdx,
    const types::idx_t *d_src_colIdx,
    const real::storage_t *d_src_val,
    types::idx_t *d_dst_rowIdx,
    types::idx_t *d_dst_colIdx,
    real::storage_t *d_dst_val,
    cudaStream_t stream
) {
    dim3 block;
    dim3 grid;

    if (nnz == 0) return 1;
    block.x = 256;
    block.y = 1;
    block.z = 1;
    grid.x = (unsigned int) ((nnz + 255u) >> 8);
    grid.y = 1;
    grid.z = 1;
    if (grid.x < 1u) grid.x = 1u;
    if (grid.x > 4096u) grid.x = 4096u;

    kernels::transpose_coo_entries<real::storage_t><<<grid, block, 0, stream>>>(
        nnz,
        d_src_rowIdx,
        d_src_colIdx,
        d_src_val,
        d_dst_rowIdx,
        d_dst_colIdx,
        d_dst_val
    );
    return cudaGetLastError() == cudaSuccess;
}

} // namespace convert
} // namespace cellshard
