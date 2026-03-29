#pragma once

#include "coo_to_csx.cuh"

namespace cellshard {
namespace convert {

using matrix::sparse::convert::csConversion_buffer;

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
    return matrix::sparse::convert::build_cs_from_coo_raw(
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

static inline int csr_conversion_buffer_build_from_coo(
    csConversion_buffer *buf,
    unsigned int rows,
    unsigned int nnz,
    const unsigned int *host_rowIdx,
    const unsigned int *host_colIdx,
    const __half *host_val
) {
    return matrix::sparse::convert::cs_conversion_buffer_build_from_coo(
        buf,
        rows,
        nnz,
        host_rowIdx,
        host_colIdx,
        host_val
    );
}

} // namespace convert
} // namespace cellshard
