#pragma once

#include "../compressed_from_coo_raw.cuh"

namespace cellshard {
namespace convert {

static inline int build_csr_from_coo_raw(
    const unsigned int rows,
    const unsigned int nnz,
    const unsigned int *d_rowIdx,
    const unsigned int *d_colIdx,
    const __half *d_val,
    unsigned int *d_rowPtr,
    unsigned int *d_heads,
    unsigned int *d_out_colIdx,
    __half *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream
) {
    return build_compressed_from_coo_raw(
        rows,
        nnz,
        d_rowIdx,
        d_colIdx,
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

static inline int build_csc_from_coo_raw(
    const unsigned int cols,
    const unsigned int nnz,
    const unsigned int *d_colIdx,
    const unsigned int *d_rowIdx,
    const __half *d_val,
    unsigned int *d_colPtr,
    unsigned int *d_heads,
    unsigned int *d_out_rowIdx,
    __half *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream
) {
    return build_compressed_from_coo_raw(
        cols,
        nnz,
        d_colIdx,
        d_rowIdx,
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
