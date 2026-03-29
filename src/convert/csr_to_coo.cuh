#pragma once

#include "csx_to_coo.cuh"

namespace cellshard {
namespace convert {

static inline int build_coo_from_csr_raw(
    const unsigned int rows,
    const unsigned int nnz,
    const unsigned int *d_rowPtr,
    const unsigned int *d_colIdx,
    const __half *d_val,
    unsigned int *d_out_rowIdx,
    unsigned int *d_out_colIdx,
    __half *d_out_val,
    cudaStream_t stream
) {
    return build_coo_from_cs_raw(
        rows,
        nnz,
        d_rowPtr,
        d_colIdx,
        d_val,
        d_out_rowIdx,
        d_out_colIdx,
        d_out_val,
        stream
    );
}

static inline int coo_conversion_buffer_build_from_csr(
    cooConversion_buffer *buf,
    unsigned int rows,
    unsigned int nnz,
    const unsigned int *host_rowPtr,
    const unsigned int *host_colIdx,
    const __half *host_val
) {
    return coo_conversion_buffer_build_from_cs(
        buf,
        rows,
        nnz,
        host_rowPtr,
        host_colIdx,
        host_val
    );
}

} // namespace convert
} // namespace cellshard
