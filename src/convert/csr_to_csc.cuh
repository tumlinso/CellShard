#pragma once

#include "kernels/_transpose.cuh"

namespace cellshard {
namespace convert {

static inline int build_csc_from_csr_raw(
    const types::dim_t rows,
    const types::dim_t cols,
    const types::nnz_t nnz,
    const types::ptr_t *d_rowPtr,
    const types::idx_t *d_colIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_colPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_rowIdx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream
) {
    return build_transpose_cs_from_cs_raw(
        rows,
        cols,
        nnz,
        d_rowPtr,
        d_colIdx,
        d_val,
        d_colPtr,
        d_heads,
        d_out_rowIdx,
        d_out_val,
        d_scan_tmp,
        scan_bytes,
        stream
    );
}

} // namespace convert
} // namespace cellshard
