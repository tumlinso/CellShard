#pragma once

#include "kernels/csExpand.cuh"

#include <cstdio>
#include <cstring>

namespace matrix {
namespace sparse {
namespace convert {

static inline int cs_to_coo_cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

struct cooConversion_buffer {
    int device;
    cudaStream_t stream;

    unsigned int cDim_capacity;
    unsigned int nnz_capacity;

    unsigned int *d_cAxPtr;
    unsigned int *d_uAxIdx;
    __half *d_val;

    unsigned int *d_out_cAxIdx;
    unsigned int *d_out_uAxIdx;
    __half *d_out_val;

    unsigned int *h_cAxIdx;
    unsigned int *h_uAxIdx;
    __half *h_val;
};

static inline void setup_coo_expand_launch(
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
    int blocks_expand = 0;
    int blocks_search = 0;

    if (nnz == 0) return 1;
    setup_coo_expand_launch(cDim, nnz, &blocks_expand, &blocks_search);

    // On V100, direct range expansion is best when the compressed axis itself can
    // expose enough blocks, or when average segment length is short.
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

__host__ __forceinline__ void coo_conversion_buffer_init(cooConversion_buffer *buf) {
    std::memset(buf, 0, sizeof(*buf));
    buf->device = -1;
}

__host__ __forceinline__ void coo_conversion_buffer_clear(cooConversion_buffer *buf) {
    if (buf->device >= 0) cudaSetDevice(buf->device);
    if (buf->d_out_val != 0) cudaFree(buf->d_out_val);
    if (buf->d_out_uAxIdx != 0) cudaFree(buf->d_out_uAxIdx);
    if (buf->d_out_cAxIdx != 0) cudaFree(buf->d_out_cAxIdx);
    if (buf->d_val != 0) cudaFree(buf->d_val);
    if (buf->d_uAxIdx != 0) cudaFree(buf->d_uAxIdx);
    if (buf->d_cAxPtr != 0) cudaFree(buf->d_cAxPtr);
    if (buf->h_val != 0) cudaFreeHost(buf->h_val);
    if (buf->h_uAxIdx != 0) cudaFreeHost(buf->h_uAxIdx);
    if (buf->h_cAxIdx != 0) cudaFreeHost(buf->h_cAxIdx);
    if (buf->stream != 0) cudaStreamDestroy(buf->stream);
    coo_conversion_buffer_init(buf);
}

__host__ __forceinline__ int coo_conversion_buffer_setup(cooConversion_buffer *buf, int device) {
    coo_conversion_buffer_init(buf);
    buf->device = device;
    if (!cs_to_coo_cuda_check(cudaSetDevice(device), "cudaSetDevice coo convert")) return 0;
    return cs_to_coo_cuda_check(cudaStreamCreateWithFlags(&buf->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
}

__host__ __forceinline__ int coo_conversion_buffer_reserve(
    cooConversion_buffer *buf,
    unsigned int cDim,
    unsigned int nnz
) {
    if (!cs_to_coo_cuda_check(cudaSetDevice(buf->device >= 0 ? buf->device : 0), "cudaSetDevice reserve coo convert")) return 0;

    if (cDim > buf->cDim_capacity) {
        if (buf->d_cAxPtr != 0) cudaFree(buf->d_cAxPtr);
        buf->d_cAxPtr = 0;

        if (!cs_to_coo_cuda_check(cudaMalloc((void **) &buf->d_cAxPtr, (std::size_t) (cDim + 1) * sizeof(unsigned int)), "cudaMalloc d_cAxPtr")) return 0;
        buf->cDim_capacity = cDim;
    }

    if (nnz > buf->nnz_capacity) {
        if (buf->d_uAxIdx != 0) cudaFree(buf->d_uAxIdx);
        if (buf->d_val != 0) cudaFree(buf->d_val);
        if (buf->d_out_cAxIdx != 0) cudaFree(buf->d_out_cAxIdx);
        if (buf->d_out_uAxIdx != 0) cudaFree(buf->d_out_uAxIdx);
        if (buf->d_out_val != 0) cudaFree(buf->d_out_val);
        if (buf->h_cAxIdx != 0) cudaFreeHost(buf->h_cAxIdx);
        if (buf->h_uAxIdx != 0) cudaFreeHost(buf->h_uAxIdx);
        if (buf->h_val != 0) cudaFreeHost(buf->h_val);
        buf->d_uAxIdx = 0;
        buf->d_val = 0;
        buf->d_out_cAxIdx = 0;
        buf->d_out_uAxIdx = 0;
        buf->d_out_val = 0;
        buf->h_cAxIdx = 0;
        buf->h_uAxIdx = 0;
        buf->h_val = 0;

        if (nnz != 0) {
            if (!cs_to_coo_cuda_check(cudaMalloc((void **) &buf->d_uAxIdx, (std::size_t) nnz * sizeof(unsigned int)), "cudaMalloc d_uAxIdx")) return 0;
            if (!cs_to_coo_cuda_check(cudaMalloc((void **) &buf->d_val, (std::size_t) nnz * sizeof(__half)), "cudaMalloc d_val")) return 0;
            if (!cs_to_coo_cuda_check(cudaMalloc((void **) &buf->d_out_cAxIdx, (std::size_t) nnz * sizeof(unsigned int)), "cudaMalloc d_out_cAxIdx")) return 0;
            if (!cs_to_coo_cuda_check(cudaMalloc((void **) &buf->d_out_uAxIdx, (std::size_t) nnz * sizeof(unsigned int)), "cudaMalloc d_out_uAxIdx")) return 0;
            if (!cs_to_coo_cuda_check(cudaMalloc((void **) &buf->d_out_val, (std::size_t) nnz * sizeof(__half)), "cudaMalloc d_out_val")) return 0;
            if (!cs_to_coo_cuda_check(cudaMallocHost((void **) &buf->h_cAxIdx, (std::size_t) nnz * sizeof(unsigned int)), "cudaMallocHost h_cAxIdx")) return 0;
            if (!cs_to_coo_cuda_check(cudaMallocHost((void **) &buf->h_uAxIdx, (std::size_t) nnz * sizeof(unsigned int)), "cudaMallocHost h_uAxIdx")) return 0;
            if (!cs_to_coo_cuda_check(cudaMallocHost((void **) &buf->h_val, (std::size_t) nnz * sizeof(__half)), "cudaMallocHost h_val")) return 0;
        }

        buf->nnz_capacity = nnz;
    }

    return 1;
}

__host__ __forceinline__ int coo_conversion_buffer_build_from_cs(
    cooConversion_buffer *buf,
    unsigned int cDim,
    unsigned int nnz,
    const unsigned int *host_cAxPtr,
    const unsigned int *host_uAxIdx,
    const __half *host_val
) {
    if (!cs_to_coo_cuda_check(cudaSetDevice(buf->device >= 0 ? buf->device : 0), "cudaSetDevice build_to_coo")) return 0;

    if (cDim == 0 && nnz == 0) return 1;
    if (!coo_conversion_buffer_reserve(buf, cDim, nnz)) return 0;

    if (!cs_to_coo_cuda_check(cudaMemcpyAsync(buf->d_cAxPtr,
                                              host_cAxPtr,
                                              (std::size_t) (cDim + 1) * sizeof(unsigned int),
                                              cudaMemcpyHostToDevice,
                                              buf->stream),
                              "copy cAxPtr")) return 0;

    if (nnz == 0) return cs_to_coo_cuda_check(cudaStreamSynchronize(buf->stream), "cudaStreamSynchronize empty coo build");

    if (!cs_to_coo_cuda_check(cudaMemcpyAsync(buf->d_uAxIdx,
                                              host_uAxIdx,
                                              (std::size_t) nnz * sizeof(unsigned int),
                                              cudaMemcpyHostToDevice,
                                              buf->stream),
                              "copy uAxIdx")) return 0;
    if (!cs_to_coo_cuda_check(cudaMemcpyAsync(buf->d_val,
                                              host_val,
                                              (std::size_t) nnz * sizeof(__half),
                                              cudaMemcpyHostToDevice,
                                              buf->stream),
                              "copy val")) return 0;

    if (!build_coo_from_cs_raw(cDim,
                               nnz,
                               buf->d_cAxPtr,
                               buf->d_uAxIdx,
                               buf->d_val,
                               buf->d_out_cAxIdx,
                               buf->d_out_uAxIdx,
                               buf->d_out_val,
                               buf->stream)) return 0;

    if (!cs_to_coo_cuda_check(cudaMemcpyAsync(buf->h_cAxIdx,
                                              buf->d_out_cAxIdx,
                                              (std::size_t) nnz * sizeof(unsigned int),
                                              cudaMemcpyDeviceToHost,
                                              buf->stream),
                              "copy cAxIdx back")) return 0;
    if (!cs_to_coo_cuda_check(cudaMemcpyAsync(buf->h_uAxIdx,
                                              buf->d_out_uAxIdx,
                                              (std::size_t) nnz * sizeof(unsigned int),
                                              cudaMemcpyDeviceToHost,
                                              buf->stream),
                              "copy uAxIdx back")) return 0;
    if (!cs_to_coo_cuda_check(cudaMemcpyAsync(buf->h_val,
                                              buf->d_out_val,
                                              (std::size_t) nnz * sizeof(__half),
                                              cudaMemcpyDeviceToHost,
                                              buf->stream),
                              "copy val back")) return 0;

    return cs_to_coo_cuda_check(cudaStreamSynchronize(buf->stream), "cudaStreamSynchronize coo build");
}

} // namespace convert
} // namespace sparse
} // namespace matrix
