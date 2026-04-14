#pragma once

#include "../../formats/blocked_ell.cuh"

#include <cub/cub.cuh>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace cellshard {
namespace bucket {

struct blocked_ell_bucket_plan {
    std::vector<std::uint32_t> row_block_order;
    std::vector<std::uint32_t> row_block_width_sorted;
};

struct blocked_ell_bucket_workspace {
    int device;
    cudaStream_t stream;
    std::uint32_t row_block_capacity;
    std::uint32_t slot_capacity;
    std::size_t sort_bytes;
    types::idx_t *d_block_col_idx;
    std::uint64_t *d_keys_in;
    std::uint64_t *d_keys_out;
    std::uint32_t *d_order_in;
    std::uint32_t *d_order_out;
    void *d_sort_tmp;
};

__host__ __forceinline__ void init(blocked_ell_bucket_workspace *ws) {
    if (ws == nullptr) return;
    ws->device = -1;
    ws->stream = (cudaStream_t) 0;
    ws->row_block_capacity = 0u;
    ws->slot_capacity = 0u;
    ws->sort_bytes = 0u;
    ws->d_block_col_idx = nullptr;
    ws->d_keys_in = nullptr;
    ws->d_keys_out = nullptr;
    ws->d_order_in = nullptr;
    ws->d_order_out = nullptr;
    ws->d_sort_tmp = nullptr;
}

__host__ __forceinline__ void clear(blocked_ell_bucket_workspace *ws) {
    if (ws == nullptr) return;
    if (ws->device >= 0) cudaSetDevice(ws->device);
    if (ws->d_sort_tmp != nullptr) cudaFree(ws->d_sort_tmp);
    if (ws->d_order_out != nullptr) cudaFree(ws->d_order_out);
    if (ws->d_order_in != nullptr) cudaFree(ws->d_order_in);
    if (ws->d_keys_out != nullptr) cudaFree(ws->d_keys_out);
    if (ws->d_keys_in != nullptr) cudaFree(ws->d_keys_in);
    if (ws->d_block_col_idx != nullptr) cudaFree(ws->d_block_col_idx);
    if (ws->stream != (cudaStream_t) 0) cudaStreamDestroy(ws->stream);
    init(ws);
}

__host__ __forceinline__ int setup(blocked_ell_bucket_workspace *ws, int device) {
    if (ws == nullptr) return 0;
    init(ws);
    ws->device = device;
    if (cudaSetDevice(device) != cudaSuccess) return 0;
    return cudaStreamCreateWithFlags(&ws->stream, cudaStreamNonBlocking) == cudaSuccess;
}

__global__ static void blocked_ell_bucket_init_identity(std::uint32_t count,
                                                        std::uint32_t *order) {
    const std::uint32_t tid = (std::uint32_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const std::uint32_t stride = (std::uint32_t) (gridDim.x * blockDim.x);
    for (std::uint32_t i = tid; i < count; i += stride) order[i] = i;
}

__global__ static void blocked_ell_bucket_build_keys(std::uint32_t row_block_count,
                                                     std::uint32_t width_blocks,
                                                     const types::idx_t *block_col_idx,
                                                     std::uint64_t *keys) {
    const std::uint32_t tid = (std::uint32_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const std::uint32_t stride = (std::uint32_t) (gridDim.x * blockDim.x);
    for (std::uint32_t rb = tid; rb < row_block_count; rb += stride) {
        const std::size_t base = (std::size_t) rb * width_blocks;
        std::uint32_t width = 0u;
        std::uint32_t hash = 2166136261u;
        for (std::uint32_t slot = 0u; slot < width_blocks; ++slot) {
            const types::idx_t col = block_col_idx[base + slot];
            if (col == sparse::blocked_ell_invalid_col) continue;
            ++width;
            hash ^= (std::uint32_t) col + 0x9e3779b9u + (hash << 6) + (hash >> 2);
            hash *= 16777619u;
        }
        keys[rb] = ((std::uint64_t) width << 32) | (std::uint64_t) hash;
    }
}

__host__ __forceinline__ int reserve(blocked_ell_bucket_workspace *ws,
                                     std::uint32_t row_block_count,
                                     std::uint32_t width_blocks) {
    std::size_t sort_bytes = 0u;
    const std::size_t slot_count = (std::size_t) row_block_count * width_blocks;
    if (ws == nullptr) return 0;
    if (cudaSetDevice(ws->device >= 0 ? ws->device : 0) != cudaSuccess) return 0;
    if (slot_count > ws->slot_capacity) {
        if (ws->d_block_col_idx != nullptr) cudaFree(ws->d_block_col_idx);
        ws->d_block_col_idx = nullptr;
        if (slot_count != 0u && cudaMalloc((void **) &ws->d_block_col_idx, slot_count * sizeof(types::idx_t)) != cudaSuccess) return 0;
        ws->slot_capacity = (std::uint32_t) slot_count;
    }
    if (row_block_count > ws->row_block_capacity) {
        if (ws->d_keys_in != nullptr) cudaFree(ws->d_keys_in);
        if (ws->d_keys_out != nullptr) cudaFree(ws->d_keys_out);
        if (ws->d_order_in != nullptr) cudaFree(ws->d_order_in);
        if (ws->d_order_out != nullptr) cudaFree(ws->d_order_out);
        ws->d_keys_in = nullptr;
        ws->d_keys_out = nullptr;
        ws->d_order_in = nullptr;
        ws->d_order_out = nullptr;
        if (row_block_count != 0u) {
            if (cudaMalloc((void **) &ws->d_keys_in, (std::size_t) row_block_count * sizeof(std::uint64_t)) != cudaSuccess) return 0;
            if (cudaMalloc((void **) &ws->d_keys_out, (std::size_t) row_block_count * sizeof(std::uint64_t)) != cudaSuccess) return 0;
            if (cudaMalloc((void **) &ws->d_order_in, (std::size_t) row_block_count * sizeof(std::uint32_t)) != cudaSuccess) return 0;
            if (cudaMalloc((void **) &ws->d_order_out, (std::size_t) row_block_count * sizeof(std::uint32_t)) != cudaSuccess) return 0;
        }
        ws->row_block_capacity = row_block_count;
    }
    if (cub::DeviceRadixSort::SortPairs(nullptr,
                                        sort_bytes,
                                        (const std::uint64_t *) nullptr,
                                        (std::uint64_t *) nullptr,
                                        (const std::uint32_t *) nullptr,
                                        (std::uint32_t *) nullptr,
                                        row_block_count,
                                        0,
                                        64,
                                        ws->stream) != cudaSuccess) {
        return 0;
    }
    if (sort_bytes > ws->sort_bytes) {
        if (ws->d_sort_tmp != nullptr) cudaFree(ws->d_sort_tmp);
        ws->d_sort_tmp = nullptr;
        if (sort_bytes != 0u && cudaMalloc(&ws->d_sort_tmp, sort_bytes) != cudaSuccess) return 0;
        ws->sort_bytes = sort_bytes;
    }
    return 1;
}

__host__ __forceinline__ int build_plan(const sparse::blocked_ell *src,
                                        blocked_ell_bucket_workspace *ws,
                                        blocked_ell_bucket_plan *out) {
    const std::uint32_t row_block_count = src != nullptr ? sparse::row_block_count(src) : 0u;
    const std::uint32_t width_blocks = src != nullptr ? sparse::ell_width_blocks(src) : 0u;
    const std::size_t slot_count = (std::size_t) row_block_count * width_blocks;
    dim3 block(256u, 1u, 1u);
    dim3 grid(row_block_count == 0u ? 1u : (std::min<std::uint32_t>)(4096u, (row_block_count + 255u) / 256u), 1u, 1u);
    std::vector<std::uint64_t> host_keys;
    std::vector<std::uint32_t> host_order;
    if (src == nullptr || ws == nullptr || out == nullptr) return 0;
    out->row_block_order.clear();
    out->row_block_width_sorted.clear();
    if (row_block_count == 0u) return 1;
    if (!reserve(ws, row_block_count, width_blocks)) return 0;
    if (slot_count != 0u
        && cudaMemcpyAsync(ws->d_block_col_idx,
                           src->blockColIdx,
                           slot_count * sizeof(types::idx_t),
                           cudaMemcpyHostToDevice,
                           ws->stream) != cudaSuccess) {
        return 0;
    }
    blocked_ell_bucket_init_identity<<<grid, block, 0, ws->stream>>>(row_block_count, ws->d_order_in);
    blocked_ell_bucket_build_keys<<<grid, block, 0, ws->stream>>>(row_block_count, width_blocks, ws->d_block_col_idx, ws->d_keys_in);
    if (cudaGetLastError() != cudaSuccess) return 0;
    if (cub::DeviceRadixSort::SortPairs(ws->d_sort_tmp,
                                        ws->sort_bytes,
                                        ws->d_keys_in,
                                        ws->d_keys_out,
                                        ws->d_order_in,
                                        ws->d_order_out,
                                        row_block_count,
                                        0,
                                        64,
                                        ws->stream) != cudaSuccess) {
        return 0;
    }
    host_keys.assign(row_block_count, 0u);
    host_order.assign(row_block_count, 0u);
    if (cudaMemcpyAsync(host_keys.data(),
                        ws->d_keys_out,
                        (std::size_t) row_block_count * sizeof(std::uint64_t),
                        cudaMemcpyDeviceToHost,
                        ws->stream) != cudaSuccess) {
        return 0;
    }
    if (cudaMemcpyAsync(host_order.data(),
                        ws->d_order_out,
                        (std::size_t) row_block_count * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost,
                        ws->stream) != cudaSuccess) {
        return 0;
    }
    if (cudaStreamSynchronize(ws->stream) != cudaSuccess) return 0;
    out->row_block_order = std::move(host_order);
    out->row_block_width_sorted.resize(row_block_count, 0u);
    for (std::uint32_t i = 0u; i < row_block_count; ++i) {
        out->row_block_width_sorted[i] = (std::uint32_t) (host_keys[i] >> 32);
    }
    return 1;
}

} // namespace bucket
} // namespace cellshard
