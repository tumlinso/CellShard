#pragma once

#include "kernels/_transpose.cuh"

namespace cellshard {
namespace convert {

static inline int build_csr_from_csc_raw(
    const types::dim_t cols,
    const types::dim_t rows,
    const types::nnz_t nnz,
    const types::ptr_t *d_colPtr,
    const types::idx_t *d_rowIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_rowPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_colIdx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream
) {
    return build_transpose_cs_from_cs_raw(
        cols,
        rows,
        nnz,
        d_colPtr,
        d_rowIdx,
        d_val,
        d_rowPtr,
        d_heads,
        d_out_colIdx,
        d_out_val,
        d_scan_tmp,
        scan_bytes,
        stream
    );
}

} // namespace convert
} // namespace cellshard
