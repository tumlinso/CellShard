#pragma once

#include "csx_to_coo.cuh"

namespace cellshard {
namespace convert {

static inline int build_coo_from_csc_raw(
    const unsigned int cols,
    const unsigned int nnz,
    const unsigned int *d_colPtr,
    const unsigned int *d_rowIdx,
    const __half *d_val,
    unsigned int *d_out_colIdx,
    unsigned int *d_out_rowIdx,
    __half *d_out_val,
    cudaStream_t stream
) {
    return build_coo_from_cs_raw(
        cols,
        nnz,
        d_colPtr,
        d_rowIdx,
        d_val,
        d_out_colIdx,
        d_out_rowIdx,
        d_out_val,
        stream
    );
}

static inline int coo_conversion_buffer_build_from_csc(
    cooConversion_buffer *buf,
    unsigned int cols,
    unsigned int nnz,
    const unsigned int *host_colPtr,
    const unsigned int *host_rowIdx,
    const __half *host_val
) {
    return coo_conversion_buffer_build_from_cs(
        buf,
        cols,
        nnz,
        host_colPtr,
        host_rowIdx,
        host_val
    );
}

} // namespace convert
} // namespace cellshard
