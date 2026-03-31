#pragma once

#include "../coo_from_compressed_raw.cuh"

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
    return build_coo_from_compressed_raw(
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
    return build_coo_from_compressed_raw(
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

} // namespace convert
} // namespace cellshard
