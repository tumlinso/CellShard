#pragma once

#include "kernels/csScatter.cuh"

#include <cstdio>
#include <cstring>

namespace matrix {
namespace sparse {
namespace convert {

static inline int cs_convert_cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

// Choose a 256-thread launch grid and clamp it to a sane range.
static inline void setup_block_part(
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

// Raw host-side launcher over already-allocated device buffers.
// No allocation, no copies, no ownership.
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
    int blocks_nnz = 0;
    int blocks_cdim = 0;

    if (cudaMemsetAsync(d_cAxPtr, 0, (std::size_t) (cDim + 1) * sizeof(unsigned int), stream) != cudaSuccess) return 0;
    if (nnz == 0) return 1;

    setup_block_part(cDim, nnz, &blocks_nnz, &blocks_cdim);

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

struct csConversion_buffer {
    int device;
    cudaStream_t stream;

    unsigned int cDim_capacity;
    unsigned int nnz_capacity;
    std::size_t scan_capacity;

    unsigned int *d_cAxIdx;
    unsigned int *d_uAxIdx;
    __half *d_val;

    unsigned int *d_cAxPtr;
    unsigned int *d_heads;
    unsigned int *d_out_uAx;
    __half *d_out_val;
    void *scan_tmp;

    unsigned int *h_cAxPtr;
    unsigned int *h_uAxIdx;
    __half *h_val;
};

__host__ __forceinline__ void cs_conversion_buffer_init(csConversion_buffer *buf) {
    std::memset(buf, 0, sizeof(*buf));
    buf->device = -1;
}

__host__ __forceinline__ void cs_conversion_buffer_clear(csConversion_buffer *buf) {
    if (buf->device >= 0) cudaSetDevice(buf->device);
    if (buf->scan_tmp != 0) cudaFree(buf->scan_tmp);
    if (buf->d_out_val != 0) cudaFree(buf->d_out_val);
    if (buf->d_out_uAx != 0) cudaFree(buf->d_out_uAx);
    if (buf->d_heads != 0) cudaFree(buf->d_heads);
    if (buf->d_cAxPtr != 0) cudaFree(buf->d_cAxPtr);
    if (buf->d_val != 0) cudaFree(buf->d_val);
    if (buf->d_uAxIdx != 0) cudaFree(buf->d_uAxIdx);
    if (buf->d_cAxIdx != 0) cudaFree(buf->d_cAxIdx);
    if (buf->h_val != 0) cudaFreeHost(buf->h_val);
    if (buf->h_uAxIdx != 0) cudaFreeHost(buf->h_uAxIdx);
    if (buf->h_cAxPtr != 0) cudaFreeHost(buf->h_cAxPtr);
    if (buf->stream != 0) cudaStreamDestroy(buf->stream);
    cs_conversion_buffer_init(buf);
}

__host__ __forceinline__ int cs_conversion_buffer_setup(csConversion_buffer *buf, int device) {
    cs_conversion_buffer_init(buf);
    buf->device = device;
    if (!cs_convert_cuda_check(cudaSetDevice(device), "cudaSetDevice cs convert")) return 0;
    return cs_convert_cuda_check(cudaStreamCreateWithFlags(&buf->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
}

__host__ __forceinline__ int cs_conversion_buffer_reserve(
    csConversion_buffer *buf,
    unsigned int cDim,
    unsigned int nnz
) {
    std::size_t scan_bytes = 0;

    if (!cs_convert_cuda_check(cudaSetDevice(buf->device >= 0 ? buf->device : 0), "cudaSetDevice reserve cs convert")) return 0;

    if (cDim > buf->cDim_capacity) {
        if (buf->d_cAxPtr != 0) cudaFree(buf->d_cAxPtr);
        if (buf->d_heads != 0) cudaFree(buf->d_heads);
        if (buf->h_cAxPtr != 0) cudaFreeHost(buf->h_cAxPtr);
        buf->d_cAxPtr = 0;
        buf->d_heads = 0;
        buf->h_cAxPtr = 0;

        if (!cs_convert_cuda_check(cudaMalloc((void **) &buf->d_cAxPtr, (std::size_t) (cDim + 1) * sizeof(unsigned int)), "cudaMalloc d_cAxPtr")) return 0;
        if (cDim != 0 && !cs_convert_cuda_check(cudaMalloc((void **) &buf->d_heads, (std::size_t) cDim * sizeof(unsigned int)), "cudaMalloc d_heads")) return 0;
        if (!cs_convert_cuda_check(cudaMallocHost((void **) &buf->h_cAxPtr, (std::size_t) (cDim + 1) * sizeof(unsigned int)), "cudaMallocHost h_cAxPtr")) return 0;
        buf->cDim_capacity = cDim;
    }

    if (nnz > buf->nnz_capacity) {
        if (buf->d_cAxIdx != 0) cudaFree(buf->d_cAxIdx);
        if (buf->d_uAxIdx != 0) cudaFree(buf->d_uAxIdx);
        if (buf->d_val != 0) cudaFree(buf->d_val);
        if (buf->d_out_uAx != 0) cudaFree(buf->d_out_uAx);
        if (buf->d_out_val != 0) cudaFree(buf->d_out_val);
        if (buf->h_uAxIdx != 0) cudaFreeHost(buf->h_uAxIdx);
        if (buf->h_val != 0) cudaFreeHost(buf->h_val);
        buf->d_cAxIdx = 0;
        buf->d_uAxIdx = 0;
        buf->d_val = 0;
        buf->d_out_uAx = 0;
        buf->d_out_val = 0;
        buf->h_uAxIdx = 0;
        buf->h_val = 0;

        if (nnz != 0) {
            if (!cs_convert_cuda_check(cudaMalloc((void **) &buf->d_cAxIdx, (std::size_t) nnz * sizeof(unsigned int)), "cudaMalloc d_cAxIdx")) return 0;
            if (!cs_convert_cuda_check(cudaMalloc((void **) &buf->d_uAxIdx, (std::size_t) nnz * sizeof(unsigned int)), "cudaMalloc d_uAxIdx")) return 0;
            if (!cs_convert_cuda_check(cudaMalloc((void **) &buf->d_val, (std::size_t) nnz * sizeof(__half)), "cudaMalloc d_val")) return 0;
            if (!cs_convert_cuda_check(cudaMalloc((void **) &buf->d_out_uAx, (std::size_t) nnz * sizeof(unsigned int)), "cudaMalloc d_out_uAx")) return 0;
            if (!cs_convert_cuda_check(cudaMalloc((void **) &buf->d_out_val, (std::size_t) nnz * sizeof(__half)), "cudaMalloc d_out_val")) return 0;
            if (!cs_convert_cuda_check(cudaMallocHost((void **) &buf->h_uAxIdx, (std::size_t) nnz * sizeof(unsigned int)), "cudaMallocHost h_uAxIdx")) return 0;
            if (!cs_convert_cuda_check(cudaMallocHost((void **) &buf->h_val, (std::size_t) nnz * sizeof(__half)), "cudaMallocHost h_val")) return 0;
        }
        buf->nnz_capacity = nnz;
    }

    if (!cs_convert_cuda_check(cub::DeviceScan::ExclusiveSum(0, scan_bytes, buf->d_cAxPtr, buf->d_cAxPtr, cDim + 1, buf->stream), "cub cs dry scan")) return 0;
    if (scan_bytes > buf->scan_capacity) {
        if (buf->scan_tmp != 0) cudaFree(buf->scan_tmp);
        buf->scan_tmp = 0;
        if (scan_bytes != 0 && !cs_convert_cuda_check(cudaMalloc((void **) &buf->scan_tmp, scan_bytes), "cudaMalloc scan_tmp")) return 0;
        buf->scan_capacity = scan_bytes;
    }
    return 1;
}

__host__ __forceinline__ int cs_conversion_buffer_build_from_coo(
    csConversion_buffer *buf,
    unsigned int cDim,
    unsigned int nnz,
    const unsigned int *host_cAxIdx,
    const unsigned int *host_uAxIdx,
    const __half *host_val
) {
    if (!cs_convert_cuda_check(cudaSetDevice(buf->device >= 0 ? buf->device : 0), "cudaSetDevice build_from_coo")) return 0;

    if (cDim == 0 && nnz == 0) return 1;
    if (!cs_conversion_buffer_reserve(buf, cDim, nnz)) return 0;

    if (nnz == 0) {
        std::memset(buf->h_cAxPtr, 0, (std::size_t) (cDim + 1) * sizeof(unsigned int));
        return 1;
    }

    if (!cs_convert_cuda_check(cudaMemcpyAsync(buf->d_cAxIdx,
                                               host_cAxIdx,
                                               (std::size_t) nnz * sizeof(unsigned int),
                                               cudaMemcpyHostToDevice,
                                               buf->stream),
                               "copy cAxIdx")) return 0;
    if (!cs_convert_cuda_check(cudaMemcpyAsync(buf->d_uAxIdx,
                                               host_uAxIdx,
                                               (std::size_t) nnz * sizeof(unsigned int),
                                               cudaMemcpyHostToDevice,
                                               buf->stream),
                               "copy uAxIdx")) return 0;
    if (!cs_convert_cuda_check(cudaMemcpyAsync(buf->d_val,
                                               host_val,
                                               (std::size_t) nnz * sizeof(__half),
                                               cudaMemcpyHostToDevice,
                                               buf->stream),
                               "copy val")) return 0;

    if (!build_cs_from_coo_raw(cDim,
                               nnz,
                               buf->d_cAxIdx,
                               buf->d_uAxIdx,
                               buf->d_val,
                               buf->d_cAxPtr,
                               buf->d_heads,
                               buf->d_out_uAx,
                               buf->d_out_val,
                               buf->scan_tmp,
                               buf->scan_capacity,
                               buf->stream)) return 0;

    if (!cs_convert_cuda_check(cudaMemcpyAsync(buf->h_cAxPtr,
                                               buf->d_cAxPtr,
                                               (std::size_t) (cDim + 1) * sizeof(unsigned int),
                                               cudaMemcpyDeviceToHost,
                                               buf->stream),
                               "copy cAxPtr back")) return 0;
    if (!cs_convert_cuda_check(cudaMemcpyAsync(buf->h_uAxIdx,
                                               buf->d_out_uAx,
                                               (std::size_t) nnz * sizeof(unsigned int),
                                               cudaMemcpyDeviceToHost,
                                               buf->stream),
                               "copy uAxIdx back")) return 0;
    if (!cs_convert_cuda_check(cudaMemcpyAsync(buf->h_val,
                                               buf->d_out_val,
                                               (std::size_t) nnz * sizeof(__half),
                                               cudaMemcpyDeviceToHost,
                                               buf->stream),
                               "copy val back")) return 0;

    return cs_convert_cuda_check(cudaStreamSynchronize(buf->stream), "cudaStreamSynchronize cs build");
}

} // namespace convert
} // namespace sparse
} // namespace matrix
