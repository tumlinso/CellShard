#pragma once

#include "coo_to_csx.cuh"

namespace cellshard {
namespace convert {

using matrix::sparse::convert::csConversion_buffer;

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
    return matrix::sparse::convert::build_cs_from_coo_raw(
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

static inline int csc_conversion_buffer_build_from_coo(
    csConversion_buffer *buf,
    unsigned int cols,
    unsigned int nnz,
    const unsigned int *host_colIdx,
    const unsigned int *host_rowIdx,
    const __half *host_val
) {
    return matrix::sparse::convert::cs_conversion_buffer_build_from_coo(
        buf,
        cols,
        nnz,
        host_colIdx,
        host_rowIdx,
        host_val
    );
}

} // namespace convert
} // namespace cellshard
