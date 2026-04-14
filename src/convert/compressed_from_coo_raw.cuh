#pragma once

#include "cusparse_utils.cuh"
#include "kernels/csScatter.cuh"

#include <cstdio>

namespace cellshard {
namespace convert {

namespace kernels {

// Reorder values into the permutation produced by cuSPARSE COO sorting.
// The ingest path feeds unsorted MTX triplets through this kernel, so keeping
// the value gather explicit is cheaper than leaning on a second heavyweight
// library routine for a simple copy pattern.
__global__ static void gather_half_by_permutation(
    const unsigned int nnz,
    const unsigned int * __restrict__ permutation,
    const __half * __restrict__ src_val,
    __half * __restrict__ dst_val
) {
    const unsigned int tid = (unsigned int) ::cellshard::ptx::global_tid_1d();
    const unsigned int stride = (unsigned int) ::cellshard::ptx::global_stride_1d();
    unsigned int i = tid;

    while (i < nnz) {
        dst_val[i] = src_val[permutation[i]];
        i += stride;
    }
}

} // namespace kernels

// Error printing stays direct and cheap.
static inline int compressed_from_coo_cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

// Choose a 256-thread launch grid and clamp it to a sane range.
// This keeps launch behavior predictable without another policy layer.
static inline void setup_compressed_from_coo_launch(
    const unsigned int cDim,
    const unsigned int nnz,
    int *blocks_nnz,
    int *blocks_cdim
) {
    *blocks_nnz = (int) ((nnz + 255u) >> 8);
    *blocks_cdim = (int) ((cDim + 255u) >> 8);

    if (*blocks_nnz < 1) *blocks_nnz = 1;
    if (*blocks_cdim < 1) *blocks_cdim = 1;
    if (*blocks_nnz > 4096) *blocks_nnz = 4096;
    if (*blocks_cdim > 4096) *blocks_cdim = 4096;
}

// Query the temporary storage required for the library-backed sorted COO path.
// The caller can reserve this once and reuse it across many parts.
static inline int compressed_from_coo_sorted_workspace_bytes(
    const unsigned int cDim,
    const unsigned int uDim,
    const unsigned int nnz,
    const unsigned int *d_cAxIdx,
    const unsigned int *d_uAxIdx,
    cudaStream_t stream,
    std::size_t *bytes_out
) {
    cusparseHandle_t handle = 0;
    std::size_t sort_bytes = 0;

    if (bytes_out == 0) return 0;
    *bytes_out = 0;
    if (nnz == 0) return 1;
    if (!cusparse_utils::acquire(stream, &handle)) return 0;
    if (!cusparse_utils::check(
            cusparseXcoosort_bufferSizeExt(
                handle,
                (int) cDim,
                (int) uDim,
                (int) nnz,
                reinterpret_cast<const int *>(d_cAxIdx),
                reinterpret_cast<const int *>(d_uAxIdx),
                &sort_bytes),
            "cusparseXcoosort_bufferSizeExt")) return 0;
    *bytes_out = sort_bytes;
    return 1;
}

// High-performance path for arbitrary COO input.
//
// The old atomic scatter builder was simple but it degrades badly on skewed
// rows and completely arbitrary MTX input order. This sorted path uses
// cuSPARSE to reorder the coordinates once, then emits compressed pointers and
// gathered values in the final device order.
static inline int build_compressed_from_coo_sorted_raw(
    const unsigned int cDim,
    const unsigned int uDim,
    const unsigned int nnz,
    const unsigned int *d_cAxIdx,
    const unsigned int *d_uAxIdx,
    const __half *d_val,
    unsigned int *d_cAxPtr,
    unsigned int *d_sort_cAxIdx,
    unsigned int *d_out_uAx,
    __half *d_out_val,
    unsigned int *d_permutation,
    void *d_sort_tmp,
    std::size_t sort_bytes,
    cudaStream_t stream
) {
    cusparseHandle_t handle = 0;
    int blocks_nnz = 0;
    int blocks_cdim = 0;
    std::size_t required_bytes = 0;

    if (!compressed_from_coo_cuda_check(
            cudaMemsetAsync(d_cAxPtr, 0, (std::size_t) (cDim + 1) * sizeof(unsigned int), stream),
            "cudaMemsetAsync sorted compressed ptr")) return 0;
    if (nnz == 0) return 1;
    if (d_sort_cAxIdx == 0 || d_out_uAx == 0 || d_out_val == 0 || d_permutation == 0 || d_sort_tmp == 0) return 0;
    if (!cusparse_utils::acquire(stream, &handle)) return 0;
    if (!compressed_from_coo_sorted_workspace_bytes(cDim, uDim, nnz, d_sort_cAxIdx, d_out_uAx, stream, &required_bytes)) return 0;
    if (sort_bytes < required_bytes) return 0;

    // Sort rows and columns in place on workspace buffers, with an explicit
    // permutation that lets us gather the values exactly once.
    if (!compressed_from_coo_cuda_check(
            cudaMemcpyAsync(
                d_sort_cAxIdx,
                d_cAxIdx,
                (std::size_t) nnz * sizeof(unsigned int),
                cudaMemcpyDeviceToDevice,
                stream),
            "cudaMemcpyAsync COO sort rows")) return 0;
    if (!compressed_from_coo_cuda_check(
            cudaMemcpyAsync(
                d_out_uAx,
                d_uAxIdx,
                (std::size_t) nnz * sizeof(unsigned int),
                cudaMemcpyDeviceToDevice,
                stream),
            "cudaMemcpyAsync COO sort cols")) return 0;
    if (!cusparse_utils::check(
            cusparseCreateIdentityPermutation(handle, (int) nnz, reinterpret_cast<int *>(d_permutation)),
            "cusparseCreateIdentityPermutation")) return 0;
    if (!cusparse_utils::check(
            cusparseXcoosortByRow(
                handle,
                (int) cDim,
                (int) uDim,
                (int) nnz,
                reinterpret_cast<int *>(d_sort_cAxIdx),
                reinterpret_cast<int *>(d_out_uAx),
                reinterpret_cast<int *>(d_permutation),
                d_sort_tmp),
            "cusparseXcoosortByRow")) return 0;
    if (!cusparse_utils::check(
            cusparseXcoo2csr(
                handle,
                reinterpret_cast<const int *>(d_sort_cAxIdx),
                (int) nnz,
                (int) cDim,
                reinterpret_cast<int *>(d_cAxPtr),
                CUSPARSE_INDEX_BASE_ZERO),
            "cusparseXcoo2csr")) return 0;

    setup_compressed_from_coo_launch(cDim, nnz, &blocks_nnz, &blocks_cdim);
    kernels::gather_half_by_permutation<<<blocks_nnz, 256, 0, stream>>>(
        nnz,
        d_permutation,
        d_val,
        d_out_val
    );
    return cudaGetLastError() == cudaSuccess;
}

// Raw host-side launcher over already-allocated device buffers.
// No allocation, no host/device copies, no ownership transfer.
//
// This compatibility path keeps the old atomic scatter algorithm. It is still
// useful when the caller does not have row-sort scratch or wants the smallest
// possible temporary footprint.
static inline int build_compressed_from_coo_raw(
    const unsigned int cDim,
    const unsigned int nnz,
    const unsigned int *d_cAxIdx,
    const unsigned int *d_uAxIdx,
    const __half *d_val,
    unsigned int *d_cAxPtr,
    unsigned int *d_heads,
    unsigned int *d_out_uAx,
    __half *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream
) {
    int blocks_nnz = 0;
    int blocks_cdim = 0;

    if (cudaMemsetAsync(d_cAxPtr, 0, (std::size_t) (cDim + 1) * sizeof(unsigned int), stream) != cudaSuccess) return 0;
    if (nnz == 0) return 1;

    setup_compressed_from_coo_launch(cDim, nnz, &blocks_nnz, &blocks_cdim);

    cellshard::convert::kernels::shift_ptr_idx_count<<<blocks_nnz, 256, 0, stream>>>(nnz, d_cAxIdx, d_cAxPtr);
    if (cudaGetLastError() != cudaSuccess) return 0;

    if (cub::DeviceScan::ExclusiveSum(
            d_scan_tmp,
            scan_bytes,
            d_cAxPtr,
            d_cAxPtr,
            cDim + 1,
            stream) != cudaSuccess) return 0;

    cellshard::convert::kernels::init_cs_scatter_heads<<<blocks_cdim, 256, 0, stream>>>(cDim, d_cAxPtr, d_heads);
    if (cudaGetLastError() != cudaSuccess) return 0;

    cellshard::convert::kernels::csScatter<<<blocks_nnz, 256, 0, stream>>>(nnz, d_cAxIdx, d_uAxIdx, d_val, d_heads, d_out_uAx, d_out_val);
    if (cudaGetLastError() != cudaSuccess) return 0;

    return 1;
}

// Alias kept for shorter call sites in other code.
static inline int build_cs_from_coo_raw(
    const unsigned int cDim,
    const unsigned int nnz,
    const unsigned int *d_cAxIdx,
    const unsigned int *d_uAxIdx,
    const __half *d_val,
    unsigned int *d_cAxPtr,
    unsigned int *d_heads,
    unsigned int *d_out_uAx,
    __half *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream
) {
    return build_compressed_from_coo_raw(
        cDim,
        nnz,
        d_cAxIdx,
        d_uAxIdx,
        d_val,
        d_cAxPtr,
        d_heads,
        d_out_uAx,
        d_out_val,
        d_scan_tmp,
        scan_bytes,
        stream
    );
}

} // namespace convert
} // namespace cellshard
