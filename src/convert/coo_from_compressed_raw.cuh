#pragma once

#include "cusparse_utils.cuh"
#include "kernels/csExpand.cuh"

#include <cstdio>

namespace cellshard {
namespace convert {

// Error printing stays direct and cheap.
static inline int coo_from_compressed_cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

// Launch setup for the two compressed->COO kernels.
static inline void setup_coo_from_compressed_launch(
    const unsigned int cDim,
    const unsigned int nnz,
    int *blocks_expand,
    int *blocks_search
) {
    *blocks_expand = (int) ((cDim + 255u) >> 8);
    *blocks_search = (int) ((nnz + 255u) >> 8);

    if (*blocks_expand < 1) *blocks_expand = 1;
    if (*blocks_search < 1) *blocks_search = 1;
    if (*blocks_expand > 4096) *blocks_expand = 4096;
    if (*blocks_search > 4096) *blocks_search = 4096;
}

// Raw host-side launcher over already-allocated device buffers.
// Input is CSR/CSC-style compressed sparse storage, output is COO.
// No allocation, no host/device copies, no ownership transfer.
static inline int build_coo_from_compressed_raw(
    const unsigned int cDim,
    const unsigned int nnz,
    const unsigned int *d_cAxPtr,
    const unsigned int *d_uAxIdx,
    const __half *d_val,
    unsigned int *d_out_cAxIdx,
    unsigned int *d_out_uAxIdx,
    __half *d_out_val,
    cudaStream_t stream
) {
    int blocks_expand = 0;
    int blocks_search = 0;
    cusparseHandle_t handle = 0;

    if (nnz == 0) return 1;

    // csr2coo is exactly the operation we need for the compressed axis and it
    // is generally faster than reconstructing those coordinates ourselves.
    // The minor axis and values are already laid out in final COO order, so
    // they only need straight async copies.
    if (cusparse_utils::acquire(stream, &handle) &&
        cusparse_utils::check(
            cusparseXcsr2coo(
                handle,
                reinterpret_cast<const int *>(d_cAxPtr),
                (int) nnz,
                (int) cDim,
                reinterpret_cast<int *>(d_out_cAxIdx),
                CUSPARSE_INDEX_BASE_ZERO),
            "cusparseXcsr2coo")) {
        if (d_out_uAxIdx != d_uAxIdx &&
            !coo_from_compressed_cuda_check(
                cudaMemcpyAsync(
                    d_out_uAxIdx,
                    d_uAxIdx,
                    (std::size_t) nnz * sizeof(unsigned int),
                    cudaMemcpyDeviceToDevice,
                    stream),
                "cudaMemcpyAsync compressed->coo minor idx")) return 0;
        if (d_out_val != d_val &&
            !coo_from_compressed_cuda_check(
                cudaMemcpyAsync(
                    d_out_val,
                    d_val,
                    (std::size_t) nnz * sizeof(__half),
                    cudaMemcpyDeviceToDevice,
                    stream),
                "cudaMemcpyAsync compressed->coo values")) return 0;
        return 1;
    }

    setup_coo_from_compressed_launch(cDim, nnz, &blocks_expand, &blocks_search);

    if (blocks_expand >= 80 || cDim >= (nnz >> 4)) {
        cellshard::convert::kernels::csExpandToCoo<<<blocks_expand, 256, 0, stream>>>(
            cDim,
            d_cAxPtr,
            d_uAxIdx,
            d_val,
            d_out_cAxIdx,
            d_out_uAxIdx,
            d_out_val);
    } else {
        cellshard::convert::kernels::csSearchToCoo<<<blocks_search, 256, 0, stream>>>(
            cDim,
            nnz,
            d_cAxPtr,
            d_uAxIdx,
            d_val,
            d_out_cAxIdx,
            d_out_uAxIdx,
            d_out_val);
    }

    return cudaGetLastError() == cudaSuccess;
}

// Alias kept for shorter call sites.
static inline int build_coo_from_cs_raw(
    const unsigned int cDim,
    const unsigned int nnz,
    const unsigned int *d_cAxPtr,
    const unsigned int *d_uAxIdx,
    const __half *d_val,
    unsigned int *d_out_cAxIdx,
    unsigned int *d_out_uAxIdx,
    __half *d_out_val,
    cudaStream_t stream
) {
    return build_coo_from_compressed_raw(
        cDim,
        nnz,
        d_cAxPtr,
        d_uAxIdx,
        d_val,
        d_out_cAxIdx,
        d_out_uAxIdx,
        d_out_val,
        stream
    );
}

} // namespace convert
} // namespace cellshard
