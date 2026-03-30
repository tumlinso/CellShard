#pragma once

#include "../csr.cuh"
#include "../../../convert/coo_to_csx.cuh"

#include <cstdio>
#include <cstring>

namespace cellshard {
namespace sparse {

static inline int csr_gpu_buffer_cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

struct csr_gpu_buffers {
    int device;
    cudaStream_t stream;

    unsigned int row_capacity;
    unsigned int nnz_capacity;
    std::size_t scan_capacity;

    unsigned int *d_rowIdx;
    unsigned int *d_colIdx;
    __half *d_val;
    unsigned int *d_rowPtr;
    unsigned int *d_heads;
    unsigned int *d_outCol;
    __half *d_outVal;
    void *scan_tmp;

    unsigned int *h_rowPtr;
    unsigned int *h_colIdx;
    __half *h_val;
};

__host__ __forceinline__ void csr_gpu_buffer_init(csr_gpu_buffers *buf) {
    std::memset(buf, 0, sizeof(*buf));
    buf->device = -1;
}

__host__ __forceinline__ void csr_gpu_buffer_clear(csr_gpu_buffers *buf) {
    if (buf->device >= 0) cudaSetDevice(buf->device);
    if (buf->scan_tmp != 0) cudaFree(buf->scan_tmp);
    if (buf->d_outVal != 0) cudaFree(buf->d_outVal);
    if (buf->d_outCol != 0) cudaFree(buf->d_outCol);
    if (buf->d_heads != 0) cudaFree(buf->d_heads);
    if (buf->d_rowPtr != 0) cudaFree(buf->d_rowPtr);
    if (buf->d_val != 0) cudaFree(buf->d_val);
    if (buf->d_colIdx != 0) cudaFree(buf->d_colIdx);
    if (buf->d_rowIdx != 0) cudaFree(buf->d_rowIdx);
    if (buf->h_val != 0) cudaFreeHost(buf->h_val);
    if (buf->h_colIdx != 0) cudaFreeHost(buf->h_colIdx);
    if (buf->h_rowPtr != 0) cudaFreeHost(buf->h_rowPtr);
    if (buf->stream != 0) cudaStreamDestroy(buf->stream);
    csr_gpu_buffer_init(buf);
}

__host__ __forceinline__ int csr_gpu_buffer_setup(csr_gpu_buffers *buf, int device) {
    csr_gpu_buffer_init(buf);
    buf->device = device;
    if (!csr_gpu_buffer_cuda_check(cudaSetDevice(device), "cudaSetDevice csr buffer")) return 0;
    return csr_gpu_buffer_cuda_check(cudaStreamCreateWithFlags(&buf->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
}

__host__ __forceinline__ int csr_gpu_buffer_reserve(csr_gpu_buffers *buf, unsigned int rows, unsigned int nnz) {
    std::size_t scan_bytes = 0;

    if (!csr_gpu_buffer_cuda_check(cudaSetDevice(buf->device >= 0 ? buf->device : 0), "cudaSetDevice csr reserve")) return 0;

    if (rows > buf->row_capacity) {
        if (buf->d_rowPtr != 0) cudaFree(buf->d_rowPtr);
        if (buf->d_heads != 0) cudaFree(buf->d_heads);
        if (buf->h_rowPtr != 0) cudaFreeHost(buf->h_rowPtr);
        buf->d_rowPtr = 0;
        buf->d_heads = 0;
        buf->h_rowPtr = 0;

        if (!csr_gpu_buffer_cuda_check(cudaMalloc((void **) &buf->d_rowPtr, (std::size_t) (rows + 1) * sizeof(unsigned int)), "cudaMalloc d_rowPtr")) return 0;
        if (rows != 0 && !csr_gpu_buffer_cuda_check(cudaMalloc((void **) &buf->d_heads, (std::size_t) rows * sizeof(unsigned int)), "cudaMalloc d_heads")) return 0;
        if (!csr_gpu_buffer_cuda_check(cudaMallocHost((void **) &buf->h_rowPtr, (std::size_t) (rows + 1) * sizeof(unsigned int)), "cudaMallocHost h_rowPtr")) return 0;
        buf->row_capacity = rows;
    }

    if (nnz > buf->nnz_capacity) {
        if (buf->d_rowIdx != 0) cudaFree(buf->d_rowIdx);
        if (buf->d_colIdx != 0) cudaFree(buf->d_colIdx);
        if (buf->d_val != 0) cudaFree(buf->d_val);
        if (buf->d_outCol != 0) cudaFree(buf->d_outCol);
        if (buf->d_outVal != 0) cudaFree(buf->d_outVal);
        if (buf->h_colIdx != 0) cudaFreeHost(buf->h_colIdx);
        if (buf->h_val != 0) cudaFreeHost(buf->h_val);
        buf->d_rowIdx = 0;
        buf->d_colIdx = 0;
        buf->d_val = 0;
        buf->d_outCol = 0;
        buf->d_outVal = 0;
        buf->h_colIdx = 0;
        buf->h_val = 0;

        if (nnz != 0) {
            if (!csr_gpu_buffer_cuda_check(cudaMalloc((void **) &buf->d_rowIdx, (std::size_t) nnz * sizeof(unsigned int)), "cudaMalloc d_rowIdx")) return 0;
            if (!csr_gpu_buffer_cuda_check(cudaMalloc((void **) &buf->d_colIdx, (std::size_t) nnz * sizeof(unsigned int)), "cudaMalloc d_colIdx")) return 0;
            if (!csr_gpu_buffer_cuda_check(cudaMalloc((void **) &buf->d_val, (std::size_t) nnz * sizeof(__half)), "cudaMalloc d_val")) return 0;
            if (!csr_gpu_buffer_cuda_check(cudaMalloc((void **) &buf->d_outCol, (std::size_t) nnz * sizeof(unsigned int)), "cudaMalloc d_outCol")) return 0;
            if (!csr_gpu_buffer_cuda_check(cudaMalloc((void **) &buf->d_outVal, (std::size_t) nnz * sizeof(__half)), "cudaMalloc d_outVal")) return 0;
            if (!csr_gpu_buffer_cuda_check(cudaMallocHost((void **) &buf->h_colIdx, (std::size_t) nnz * sizeof(unsigned int)), "cudaMallocHost h_colIdx")) return 0;
            if (!csr_gpu_buffer_cuda_check(cudaMallocHost((void **) &buf->h_val, (std::size_t) nnz * sizeof(__half)), "cudaMallocHost h_val")) return 0;
        }
        buf->nnz_capacity = nnz;
    }

    if (!csr_gpu_buffer_cuda_check(cub::DeviceScan::ExclusiveSum(0, scan_bytes, buf->d_rowPtr, buf->d_rowPtr, rows + 1, buf->stream), "cub csr dry scan")) return 0;
    if (scan_bytes > buf->scan_capacity) {
        if (buf->scan_tmp != 0) cudaFree(buf->scan_tmp);
        buf->scan_tmp = 0;
        if (scan_bytes != 0 && !csr_gpu_buffer_cuda_check(cudaMalloc((void **) &buf->scan_tmp, scan_bytes), "cudaMalloc scan_tmp")) return 0;
        buf->scan_capacity = scan_bytes;
    }

    return 1;
}

__host__ __forceinline__ int csr_gpu_buffer_build_from_coo(
    csr_gpu_buffers *buf,
    unsigned int rows,
    unsigned int nnz,
    const unsigned int *host_rowIdx,
    const unsigned int *host_colIdx,
    const __half *host_val
) {
    if (!csr_gpu_buffer_cuda_check(cudaSetDevice(buf->device >= 0 ? buf->device : 0), "cudaSetDevice csr build")) return 0;
    if (rows == 0 && nnz == 0) return 1;
    if (!csr_gpu_buffer_reserve(buf, rows, nnz)) return 0;

    if (nnz == 0) {
        std::memset(buf->h_rowPtr, 0, (std::size_t) (rows + 1) * sizeof(unsigned int));
        return 1;
    }

    if (!csr_gpu_buffer_cuda_check(cudaMemcpyAsync(buf->d_rowIdx,
                                                   host_rowIdx,
                                                   (std::size_t) nnz * sizeof(unsigned int),
                                                   cudaMemcpyHostToDevice,
                                                   buf->stream),
                                   "copy rowIdx")) return 0;
    if (!csr_gpu_buffer_cuda_check(cudaMemcpyAsync(buf->d_colIdx,
                                                   host_colIdx,
                                                   (std::size_t) nnz * sizeof(unsigned int),
                                                   cudaMemcpyHostToDevice,
                                                   buf->stream),
                                   "copy colIdx")) return 0;
    if (!csr_gpu_buffer_cuda_check(cudaMemcpyAsync(buf->d_val,
                                                   host_val,
                                                   (std::size_t) nnz * sizeof(__half),
                                                   cudaMemcpyHostToDevice,
                                                   buf->stream),
                                   "copy val")) return 0;

    if (!convert::build_cs_from_coo_raw(rows,
                                        nnz,
                                        buf->d_rowIdx,
                                        buf->d_colIdx,
                                        buf->d_val,
                                        buf->d_rowPtr,
                                        buf->d_heads,
                                        buf->d_outCol,
                                        buf->d_outVal,
                                        buf->scan_tmp,
                                        buf->scan_capacity,
                                        buf->stream)) return 0;

    if (!csr_gpu_buffer_cuda_check(cudaMemcpyAsync(buf->h_rowPtr,
                                                   buf->d_rowPtr,
                                                   (std::size_t) (rows + 1) * sizeof(unsigned int),
                                                   cudaMemcpyDeviceToHost,
                                                   buf->stream),
                                   "copy rowPtr back")) return 0;
    if (!csr_gpu_buffer_cuda_check(cudaMemcpyAsync(buf->h_colIdx,
                                                   buf->d_outCol,
                                                   (std::size_t) nnz * sizeof(unsigned int),
                                                   cudaMemcpyDeviceToHost,
                                                   buf->stream),
                                   "copy colIdx back")) return 0;
    if (!csr_gpu_buffer_cuda_check(cudaMemcpyAsync(buf->h_val,
                                                   buf->d_outVal,
                                                   (std::size_t) nnz * sizeof(__half),
                                                   cudaMemcpyDeviceToHost,
                                                   buf->stream),
                                   "copy val back")) return 0;

    return csr_gpu_buffer_cuda_check(cudaStreamSynchronize(buf->stream), "cudaStreamSynchronize csr build");
}

} // namespace sparse
} // namespace cellshard
