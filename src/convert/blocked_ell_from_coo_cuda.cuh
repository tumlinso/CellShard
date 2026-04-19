#pragma once

#include "../formats/blocked_ell.cuh"
#include "../formats/triplet.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace cellshard {
namespace convert {

struct blocked_ell_from_coo_cuda_workspace {
    int device;
    cudaStream_t stream;
    int owns_stream;

    types::nnz_t nnz_capacity;
    types::nnz_t sort_nnz_capacity;
    types::dim_t row_block_capacity;
    types::dim_t feature_capacity;

    unsigned int *d_row_idx;
    unsigned int *d_col_idx;
    __half *d_val;

    unsigned int *d_feature_to_global;

    types::u64 *d_sort_keys_in;
    types::u64 *d_sort_keys_out;
    types::idx_t *d_sort_pos_in;
    types::idx_t *d_sort_pos_out;
    types::idx_t *d_head_flags;
    types::idx_t *d_group_ids;
    types::idx_t *d_row_block_counts;
    types::ptr_t *d_row_block_offsets;
    types::u32 *d_ell_width;

    void *d_sort_tmp;
    std::size_t d_sort_tmp_bytes;
    void *d_scan_tmp;
    std::size_t d_scan_tmp_bytes;
    void *d_reduce_tmp;
    std::size_t d_reduce_tmp_bytes;

    std::size_t output_capacity_bytes;
    void *d_output_block;
    types::idx_t *d_output_block_cols;
    real::storage_t *d_output_val;
    types::nnz_t *d_output_nnz;
};

namespace detail {

inline int blocked_ell_from_coo_cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

inline std::size_t blocked_ell_from_coo_align_up_bytes_(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

namespace kernels {

__global__ static void remap_feature_columns_kernel(types::nnz_t nnz,
                                                    const unsigned int * __restrict__ feature_to_global,
                                                    unsigned int * __restrict__ col_idx) {
    const types::idx_t tid = (types::idx_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::idx_t stride = (types::idx_t) (blockDim.x * gridDim.x);
    types::idx_t i = tid;
    while (i < nnz) {
        col_idx[i] = feature_to_global[col_idx[i]];
        i += stride;
    }
}

__global__ static void build_block_sort_keys_kernel(types::nnz_t nnz,
                                                    const unsigned int * __restrict__ row_idx,
                                                    const unsigned int * __restrict__ col_idx,
                                                    types::u32 block_size,
                                                    types::u64 * __restrict__ keys,
                                                    types::idx_t * __restrict__ positions) {
    const types::idx_t tid = (types::idx_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::idx_t stride = (types::idx_t) (blockDim.x * gridDim.x);
    types::idx_t i = tid;
    while (i < nnz) {
        const types::u32 row_block = block_size != 0u ? row_idx[i] / block_size : 0u;
        const types::u32 block_col = block_size != 0u ? col_idx[i] / block_size : 0u;
        keys[i] = ((types::u64) row_block << 32u) | (types::u64) block_col;
        positions[i] = i;
        i += stride;
    }
}

__global__ static void mark_block_group_heads_and_count_kernel(types::nnz_t nnz,
                                                               const types::u64 * __restrict__ sorted_keys,
                                                               types::idx_t * __restrict__ head_flags,
                                                               types::idx_t * __restrict__ row_block_counts) {
    const types::idx_t tid = (types::idx_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::idx_t stride = (types::idx_t) (blockDim.x * gridDim.x);
    types::idx_t i = tid;
    while (i < nnz) {
        const types::u64 key = sorted_keys[i];
        const int is_head = i == 0u || key != sorted_keys[i - 1u];
        head_flags[i] = is_head ? 1u : 0u;
        if (is_head) {
            const types::idx_t row_block = (types::idx_t) (key >> 32u);
            atomicAdd(row_block_counts + row_block, 1u);
        }
        i += stride;
    }
}

__global__ static void fill_invalid_block_cols_kernel(types::idx_t count,
                                                      types::idx_t * __restrict__ block_cols) {
    const types::idx_t tid = (types::idx_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::idx_t stride = (types::idx_t) (blockDim.x * gridDim.x);
    types::idx_t i = tid;
    while (i < count) {
        block_cols[i] = sparse::blocked_ell_invalid_col;
        i += stride;
    }
}

__global__ static void scatter_block_col_ids_kernel(types::nnz_t nnz,
                                                    const types::u64 * __restrict__ sorted_keys,
                                                    const types::idx_t * __restrict__ head_flags,
                                                    const types::idx_t * __restrict__ group_ids,
                                                    const types::ptr_t * __restrict__ row_block_offsets,
                                                    types::u32 ell_width,
                                                    types::idx_t * __restrict__ block_cols) {
    const types::idx_t tid = (types::idx_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::idx_t stride = (types::idx_t) (blockDim.x * gridDim.x);
    types::idx_t i = tid;
    while (i < nnz) {
        if (head_flags[i] != 0u) {
            const types::u64 key = sorted_keys[i];
            const types::idx_t row_block = (types::idx_t) (key >> 32u);
            const types::idx_t block_col = (types::idx_t) (key & 0xffffffffu);
            const types::idx_t slot = group_ids[i] - row_block_offsets[row_block];
            block_cols[(types::u64) row_block * (types::u64) ell_width + slot] = block_col;
        }
        i += stride;
    }
}

__global__ static void scatter_blocked_values_kernel(types::nnz_t nnz,
                                                     const types::u64 * __restrict__ sorted_keys,
                                                     const types::idx_t * __restrict__ sorted_pos,
                                                     const types::idx_t * __restrict__ group_ids,
                                                     const types::ptr_t * __restrict__ row_block_offsets,
                                                     const unsigned int * __restrict__ row_idx,
                                                     const unsigned int * __restrict__ col_idx,
                                                     const real::storage_t * __restrict__ val,
                                                     types::u32 block_size,
                                                     types::u32 ell_cols,
                                                     real::storage_t * __restrict__ out_val) {
    const types::idx_t tid = (types::idx_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::idx_t stride = (types::idx_t) (blockDim.x * gridDim.x);
    types::idx_t i = tid;
    while (i < nnz) {
        const types::u64 key = sorted_keys[i];
        const types::idx_t src_idx = sorted_pos[i];
        const types::idx_t row_block = (types::idx_t) (key >> 32u);
        const types::idx_t slot = group_ids[i] - row_block_offsets[row_block];
        const types::idx_t row = row_idx[src_idx];
        const types::idx_t col_in_block = block_size != 0u ? col_idx[src_idx] % block_size : 0u;
        out_val[(types::u64) row * (types::u64) ell_cols + (types::u64) slot * block_size + col_in_block] = val[src_idx];
        i += stride;
    }
}

__global__ static void count_nonzero_blocked_values_kernel(types::u64 value_count,
                                                           types::u32 rows,
                                                           types::u32 cols,
                                                           types::u32 block_size,
                                                           types::u32 ell_width,
                                                           const types::idx_t * __restrict__ block_cols,
                                                           const real::storage_t * __restrict__ values,
                                                           types::nnz_t * __restrict__ out_nnz) {
    const types::u64 tid = (types::u64) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::u64 stride = (types::u64) (blockDim.x * gridDim.x);
    types::nnz_t local = 0u;
    for (types::u64 i = tid; i < value_count; i += stride) {
        const types::u32 row = (types::u32) (i / ell_width / block_size);
        const types::u32 in_row = (types::u32) (i - (types::u64) row * (types::u64) ell_width * (types::u64) block_size);
        const types::u32 slot = in_row / block_size;
        const types::u32 col_in_block = in_row % block_size;
        const types::u32 row_block = block_size != 0u ? row / block_size : 0u;
        const types::idx_t block_col = block_cols[(types::u64) row_block * (types::u64) ell_width + slot];
        const types::u32 col = (types::u32) block_col * block_size + col_in_block;
        if (row < rows
            && block_col != sparse::blocked_ell_invalid_col
            && col < cols
            && __half2float(values[i]) != 0.0f) {
            ++local;
        }
    }
    if (local != 0u) atomicAdd(out_nnz, local);
}

} // namespace kernels

} // namespace detail

__host__ __forceinline__ void init(blocked_ell_from_coo_cuda_workspace *ws) {
    std::memset(ws, 0, sizeof(*ws));
    ws->device = -1;
}

__host__ __forceinline__ void clear(blocked_ell_from_coo_cuda_workspace *ws) {
    if (ws == nullptr) return;
    if (ws->device >= 0) cudaSetDevice(ws->device);
    if (ws->d_output_nnz != 0) cudaFree(ws->d_output_nnz);
    if (ws->d_output_block != 0) cudaFree(ws->d_output_block);
    if (ws->d_reduce_tmp != 0) cudaFree(ws->d_reduce_tmp);
    if (ws->d_scan_tmp != 0) cudaFree(ws->d_scan_tmp);
    if (ws->d_sort_tmp != 0) cudaFree(ws->d_sort_tmp);
    if (ws->d_ell_width != 0) cudaFree(ws->d_ell_width);
    if (ws->d_row_block_offsets != 0) cudaFree(ws->d_row_block_offsets);
    if (ws->d_row_block_counts != 0) cudaFree(ws->d_row_block_counts);
    if (ws->d_group_ids != 0) cudaFree(ws->d_group_ids);
    if (ws->d_head_flags != 0) cudaFree(ws->d_head_flags);
    if (ws->d_sort_pos_out != 0) cudaFree(ws->d_sort_pos_out);
    if (ws->d_sort_pos_in != 0) cudaFree(ws->d_sort_pos_in);
    if (ws->d_sort_keys_out != 0) cudaFree(ws->d_sort_keys_out);
    if (ws->d_sort_keys_in != 0) cudaFree(ws->d_sort_keys_in);
    if (ws->d_feature_to_global != 0) cudaFree(ws->d_feature_to_global);
    if (ws->d_val != 0) cudaFree(ws->d_val);
    if (ws->d_col_idx != 0) cudaFree(ws->d_col_idx);
    if (ws->d_row_idx != 0) cudaFree(ws->d_row_idx);
    if (ws->owns_stream && ws->stream != (cudaStream_t) 0) cudaStreamDestroy(ws->stream);
    init(ws);
}

__host__ __forceinline__ int setup(blocked_ell_from_coo_cuda_workspace *ws,
                                   int device,
                                   cudaStream_t stream = (cudaStream_t) 0) {
    clear(ws);
    if (!detail::blocked_ell_from_coo_cuda_check(cudaSetDevice(device), "cudaSetDevice blocked_ell_from_coo setup")) return 0;
    ws->device = device;
    if (stream == (cudaStream_t) 0) {
        if (!detail::blocked_ell_from_coo_cuda_check(cudaStreamCreateWithFlags(&ws->stream, cudaStreamNonBlocking),
                                                     "cudaStreamCreateWithFlags blocked_ell_from_coo")) return 0;
        ws->owns_stream = 1;
    } else {
        ws->stream = stream;
        ws->owns_stream = 0;
    }
    return 1;
}

__host__ __forceinline__ int reserve_upload(blocked_ell_from_coo_cuda_workspace *ws,
                                            types::nnz_t nnz) {
    if (nnz <= ws->nnz_capacity) return 1;
    if (!detail::blocked_ell_from_coo_cuda_check(cudaSetDevice(ws->device), "cudaSetDevice blocked_ell_from_coo reserve upload")) return 0;
    if (ws->d_row_idx != 0) cudaFree(ws->d_row_idx);
    if (ws->d_col_idx != 0) cudaFree(ws->d_col_idx);
    if (ws->d_val != 0) cudaFree(ws->d_val);
    ws->d_row_idx = 0;
    ws->d_col_idx = 0;
    ws->d_val = 0;
    if (nnz != 0u) {
        if (!detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_row_idx, (std::size_t) nnz * sizeof(unsigned int)),
                                                     "cudaMalloc blocked_ell_from_coo row_idx")) return 0;
        if (!detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_col_idx, (std::size_t) nnz * sizeof(unsigned int)),
                                                     "cudaMalloc blocked_ell_from_coo col_idx")) return 0;
        if (!detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_val, (std::size_t) nnz * sizeof(__half)),
                                                     "cudaMalloc blocked_ell_from_coo val")) return 0;
    }
    ws->nnz_capacity = nnz;
    return 1;
}

__host__ __forceinline__ int reserve_feature_map(blocked_ell_from_coo_cuda_workspace *ws,
                                                 types::dim_t cols) {
    if (cols <= ws->feature_capacity) return 1;
    if (!detail::blocked_ell_from_coo_cuda_check(cudaSetDevice(ws->device), "cudaSetDevice blocked_ell_from_coo reserve feature map")) return 0;
    if (ws->d_feature_to_global != 0) cudaFree(ws->d_feature_to_global);
    ws->d_feature_to_global = 0;
    if (cols != 0u &&
        !detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_feature_to_global, (std::size_t) cols * sizeof(unsigned int)),
                                                 "cudaMalloc blocked_ell_from_coo feature map")) return 0;
    ws->feature_capacity = cols;
    return 1;
}

__host__ __forceinline__ int reserve_sort(blocked_ell_from_coo_cuda_workspace *ws,
                                          types::nnz_t nnz,
                                          types::dim_t row_blocks) {
    std::size_t sort_bytes = 0u;
    std::size_t scan_nnz_bytes = 0u;
    std::size_t scan_rows_bytes = 0u;
    std::size_t reduce_bytes = 0u;
    if (!detail::blocked_ell_from_coo_cuda_check(cudaSetDevice(ws->device), "cudaSetDevice blocked_ell_from_coo reserve sort")) return 0;

    if (nnz > ws->sort_nnz_capacity || row_blocks > ws->row_block_capacity) {
        if (ws->d_sort_keys_in != 0) cudaFree(ws->d_sort_keys_in);
        if (ws->d_sort_keys_out != 0) cudaFree(ws->d_sort_keys_out);
        if (ws->d_sort_pos_in != 0) cudaFree(ws->d_sort_pos_in);
        if (ws->d_sort_pos_out != 0) cudaFree(ws->d_sort_pos_out);
        if (ws->d_head_flags != 0) cudaFree(ws->d_head_flags);
        if (ws->d_group_ids != 0) cudaFree(ws->d_group_ids);
        if (ws->d_row_block_counts != 0) cudaFree(ws->d_row_block_counts);
        if (ws->d_row_block_offsets != 0) cudaFree(ws->d_row_block_offsets);
        if (ws->d_ell_width != 0) cudaFree(ws->d_ell_width);
        ws->d_sort_keys_in = 0;
        ws->d_sort_keys_out = 0;
        ws->d_sort_pos_in = 0;
        ws->d_sort_pos_out = 0;
        ws->d_head_flags = 0;
        ws->d_group_ids = 0;
        ws->d_row_block_counts = 0;
        ws->d_row_block_offsets = 0;
        ws->d_ell_width = 0;

        if (nnz != 0u) {
            if (!detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_sort_keys_in, (std::size_t) nnz * sizeof(types::u64)),
                                                         "cudaMalloc blocked_ell_from_coo sort keys in")) return 0;
            if (!detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_sort_keys_out, (std::size_t) nnz * sizeof(types::u64)),
                                                         "cudaMalloc blocked_ell_from_coo sort keys out")) return 0;
            if (!detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_sort_pos_in, (std::size_t) nnz * sizeof(types::idx_t)),
                                                         "cudaMalloc blocked_ell_from_coo sort pos in")) return 0;
            if (!detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_sort_pos_out, (std::size_t) nnz * sizeof(types::idx_t)),
                                                         "cudaMalloc blocked_ell_from_coo sort pos out")) return 0;
            if (!detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_head_flags, (std::size_t) nnz * sizeof(types::idx_t)),
                                                         "cudaMalloc blocked_ell_from_coo head flags")) return 0;
            if (!detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_group_ids, (std::size_t) nnz * sizeof(types::idx_t)),
                                                         "cudaMalloc blocked_ell_from_coo group ids")) return 0;
        }
        if (row_blocks != 0u) {
            if (!detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_row_block_counts, (std::size_t) (row_blocks + 1u) * sizeof(types::idx_t)),
                                                         "cudaMalloc blocked_ell_from_coo row block counts")) return 0;
            if (!detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_row_block_offsets, (std::size_t) (row_blocks + 1u) * sizeof(types::ptr_t)),
                                                         "cudaMalloc blocked_ell_from_coo row block offsets")) return 0;
            if (!detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_ell_width, sizeof(types::u32)),
                                                         "cudaMalloc blocked_ell_from_coo ell width")) return 0;
        }
        ws->sort_nnz_capacity = nnz;
        ws->row_block_capacity = row_blocks;
    }

    if (nnz != 0u) {
        if (cub::DeviceRadixSort::SortPairs(nullptr,
                                            sort_bytes,
                                            (const types::u64 *) 0,
                                            (types::u64 *) 0,
                                            (const types::idx_t *) 0,
                                            (types::idx_t *) 0,
                                            nnz,
                                            0,
                                            sizeof(types::u64) * 8,
                                            ws->stream) != cudaSuccess) {
            std::fprintf(stderr, "CUB error at blocked_ell_from_coo sort sizing\n");
            return 0;
        }
        if (cub::DeviceScan::ExclusiveSum(nullptr,
                                          scan_nnz_bytes,
                                          ws->d_head_flags,
                                          ws->d_group_ids,
                                          nnz,
                                          ws->stream) != cudaSuccess) {
            std::fprintf(stderr, "CUB error at blocked_ell_from_coo group scan sizing\n");
            return 0;
        }
    }
    if (row_blocks != 0u) {
        if (cub::DeviceScan::ExclusiveSum(nullptr,
                                          scan_rows_bytes,
                                          ws->d_row_block_counts,
                                          ws->d_row_block_offsets,
                                          (std::size_t) row_blocks + 1u,
                                          ws->stream) != cudaSuccess) {
            std::fprintf(stderr, "CUB error at blocked_ell_from_coo row-block scan sizing\n");
            return 0;
        }
        if (cub::DeviceReduce::Max(nullptr,
                                   reduce_bytes,
                                   ws->d_row_block_counts,
                                   ws->d_ell_width,
                                   row_blocks,
                                   ws->stream) != cudaSuccess) {
            std::fprintf(stderr, "CUB error at blocked_ell_from_coo ell-width reduce sizing\n");
            return 0;
        }
    }

    if (sort_bytes > ws->d_sort_tmp_bytes) {
        if (ws->d_sort_tmp != 0) cudaFree(ws->d_sort_tmp);
        ws->d_sort_tmp = 0;
        if (sort_bytes != 0u &&
            !detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_sort_tmp, sort_bytes),
                                                     "cudaMalloc blocked_ell_from_coo sort tmp")) return 0;
        ws->d_sort_tmp_bytes = sort_bytes;
    }
    {
        const std::size_t scan_bytes = scan_nnz_bytes > scan_rows_bytes ? scan_nnz_bytes : scan_rows_bytes;
        if (scan_bytes > ws->d_scan_tmp_bytes) {
            if (ws->d_scan_tmp != 0) cudaFree(ws->d_scan_tmp);
            ws->d_scan_tmp = 0;
            if (scan_bytes != 0u &&
                !detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_scan_tmp, scan_bytes),
                                                         "cudaMalloc blocked_ell_from_coo scan tmp")) return 0;
            ws->d_scan_tmp_bytes = scan_bytes;
        }
    }
    if (reduce_bytes > ws->d_reduce_tmp_bytes) {
        if (ws->d_reduce_tmp != 0) cudaFree(ws->d_reduce_tmp);
        ws->d_reduce_tmp = 0;
        if (reduce_bytes != 0u &&
            !detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_reduce_tmp, reduce_bytes),
                                                     "cudaMalloc blocked_ell_from_coo reduce tmp")) return 0;
        ws->d_reduce_tmp_bytes = reduce_bytes;
    }
    return 1;
}

__host__ __forceinline__ int reserve_output(blocked_ell_from_coo_cuda_workspace *ws,
                                            types::dim_t rows,
                                            types::u32 ell_width,
                                            types::u32 block_size) {
    const types::dim_t row_blocks = block_size == 0u ? 0u : (rows + block_size - 1u) / block_size;
    const std::size_t idx_bytes = (std::size_t) row_blocks * (std::size_t) ell_width * sizeof(types::idx_t);
    const std::size_t val_offset = detail::blocked_ell_from_coo_align_up_bytes_(idx_bytes, alignof(real::storage_t));
    const std::size_t total_bytes = val_offset + (std::size_t) rows * (std::size_t) ell_width * (std::size_t) block_size * sizeof(real::storage_t);
    if (total_bytes <= ws->output_capacity_bytes) {
        ws->d_output_block_cols = idx_bytes != 0u ? (types::idx_t *) ws->d_output_block : 0;
        ws->d_output_val = total_bytes != 0u ? (real::storage_t *) ((char *) ws->d_output_block + val_offset) : 0;
        if (ws->d_output_nnz == 0u) {
            if (!detail::blocked_ell_from_coo_cuda_check(cudaSetDevice(ws->device), "cudaSetDevice blocked_ell_from_coo reserve output nnz")) return 0;
            if (!detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_output_nnz, sizeof(types::nnz_t)),
                                                         "cudaMalloc blocked_ell_from_coo output nnz")) return 0;
        }
        return 1;
    }
    if (!detail::blocked_ell_from_coo_cuda_check(cudaSetDevice(ws->device), "cudaSetDevice blocked_ell_from_coo reserve output")) return 0;
    if (ws->d_output_block != 0) cudaFree(ws->d_output_block);
    ws->d_output_block = 0;
    ws->d_output_block_cols = 0;
    ws->d_output_val = 0;
    if (total_bytes != 0u &&
        !detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_output_block, total_bytes),
                                                 "cudaMalloc blocked_ell_from_coo output")) return 0;
    if (ws->d_output_nnz == 0u
        && !detail::blocked_ell_from_coo_cuda_check(cudaMalloc(&ws->d_output_nnz, sizeof(types::nnz_t)),
                                                    "cudaMalloc blocked_ell_from_coo output nnz")) return 0;
    ws->output_capacity_bytes = total_bytes;
    ws->d_output_block_cols = idx_bytes != 0u ? (types::idx_t *) ws->d_output_block : 0;
    ws->d_output_val = total_bytes != 0u ? (real::storage_t *) ((char *) ws->d_output_block + val_offset) : 0;
    return 1;
}

__host__ __forceinline__ dim3 blocked_ell_from_coo_grid(types::nnz_t n) {
    unsigned int blocks = (unsigned int) ((n + 255u) >> 8);
    if (blocks == 0u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    return dim3(blocks, 1u, 1u);
}

__host__ __forceinline__ int upload_coo(blocked_ell_from_coo_cuda_workspace *ws,
                                        const sparse::coo *src,
                                        const types::u32 *feature_to_global) {
    const dim3 block(256u, 1u, 1u);
    const dim3 grid = blocked_ell_from_coo_grid(src != 0 ? src->nnz : 0u);
    if (ws == 0 || src == 0) return 0;
    if (!detail::blocked_ell_from_coo_cuda_check(cudaSetDevice(ws->device), "cudaSetDevice blocked_ell_from_coo upload")) return 0;
    if (!reserve_upload(ws, src->nnz)) return 0;
    if (feature_to_global != 0 && !reserve_feature_map(ws, src->cols)) return 0;
    if (src->nnz != 0u) {
        if (!detail::blocked_ell_from_coo_cuda_check(cudaMemcpyAsync(ws->d_row_idx,
                                                                     src->rowIdx,
                                                                     (std::size_t) src->nnz * sizeof(unsigned int),
                                                                     cudaMemcpyHostToDevice,
                                                                     ws->stream),
                                                     "cudaMemcpyAsync blocked_ell_from_coo row_idx")) return 0;
        if (!detail::blocked_ell_from_coo_cuda_check(cudaMemcpyAsync(ws->d_col_idx,
                                                                     src->colIdx,
                                                                     (std::size_t) src->nnz * sizeof(unsigned int),
                                                                     cudaMemcpyHostToDevice,
                                                                     ws->stream),
                                                     "cudaMemcpyAsync blocked_ell_from_coo col_idx")) return 0;
        if (!detail::blocked_ell_from_coo_cuda_check(cudaMemcpyAsync(ws->d_val,
                                                                     src->val,
                                                                     (std::size_t) src->nnz * sizeof(__half),
                                                                     cudaMemcpyHostToDevice,
                                                                     ws->stream),
                                                     "cudaMemcpyAsync blocked_ell_from_coo val")) return 0;
    }
    if (feature_to_global != 0 && src->cols != 0u) {
        if (!detail::blocked_ell_from_coo_cuda_check(cudaMemcpyAsync(ws->d_feature_to_global,
                                                                     feature_to_global,
                                                                     (std::size_t) src->cols * sizeof(unsigned int),
                                                                     cudaMemcpyHostToDevice,
                                                                     ws->stream),
                                                     "cudaMemcpyAsync blocked_ell_from_coo feature_to_global")) return 0;
        if (src->nnz != 0u) {
            detail::kernels::remap_feature_columns_kernel<<<grid, block, 0, ws->stream>>>(
                src->nnz,
                ws->d_feature_to_global,
                ws->d_col_idx
            );
            if (!detail::blocked_ell_from_coo_cuda_check(cudaGetLastError(), "remap_feature_columns_kernel")) return 0;
        }
    }
    return 1;
}

__host__ __forceinline__ int evaluate_block_size(blocked_ell_from_coo_cuda_workspace *ws,
                                                 types::dim_t rows,
                                                 types::nnz_t nnz,
                                                 types::u32 block_size,
                                                 types::u32 *ell_width_out,
                                                 types::u64 *total_blocks_out) {
    const types::dim_t row_blocks = block_size == 0u ? 0u : (rows + block_size - 1u) / block_size;
    const dim3 block(256u, 1u, 1u);
    const dim3 grid = blocked_ell_from_coo_grid(nnz);
    types::u32 ell_width = 0u;
    types::ptr_t total_blocks_device = 0u;
    types::u64 total_blocks = 0u;

    if (ell_width_out == 0 || total_blocks_out == 0 || ws == 0 || block_size == 0u) return 0;
    *ell_width_out = 0u;
    *total_blocks_out = 0u;
    if (rows == 0u || nnz == 0u) return 1;
    if (!reserve_sort(ws, nnz, row_blocks)) return 0;

    detail::kernels::build_block_sort_keys_kernel<<<grid, block, 0, ws->stream>>>(
        nnz,
        ws->d_row_idx,
        ws->d_col_idx,
        block_size,
        ws->d_sort_keys_in,
        ws->d_sort_pos_in
    );
    if (!detail::blocked_ell_from_coo_cuda_check(cudaGetLastError(), "build_block_sort_keys_kernel eval")) return 0;
    if (!detail::blocked_ell_from_coo_cuda_check(cudaMemsetAsync(ws->d_row_block_counts,
                                                                 0,
                                                                 (std::size_t) (row_blocks + 1u) * sizeof(types::idx_t),
                                                                 ws->stream),
                                                 "cudaMemsetAsync blocked_ell_from_coo row_block_counts eval")) return 0;
    if (cub::DeviceRadixSort::SortPairs(ws->d_sort_tmp,
                                        ws->d_sort_tmp_bytes,
                                        ws->d_sort_keys_in,
                                        ws->d_sort_keys_out,
                                        ws->d_sort_pos_in,
                                        ws->d_sort_pos_out,
                                        nnz,
                                        0,
                                        sizeof(types::u64) * 8,
                                        ws->stream) != cudaSuccess) {
        std::fprintf(stderr, "CUB error at blocked_ell_from_coo sort pairs eval\n");
        return 0;
    }
    detail::kernels::mark_block_group_heads_and_count_kernel<<<grid, block, 0, ws->stream>>>(
        nnz,
        ws->d_sort_keys_out,
        ws->d_head_flags,
        ws->d_row_block_counts
    );
    if (!detail::blocked_ell_from_coo_cuda_check(cudaGetLastError(), "mark_block_group_heads_and_count_kernel eval")) return 0;
    if (cub::DeviceScan::ExclusiveSum(ws->d_scan_tmp,
                                      ws->d_scan_tmp_bytes,
                                      ws->d_row_block_counts,
                                      ws->d_row_block_offsets,
                                      (std::size_t) row_blocks + 1u,
                                      ws->stream) != cudaSuccess) {
        std::fprintf(stderr, "CUB error at blocked_ell_from_coo row-block scan eval\n");
        return 0;
    }
    if (cub::DeviceReduce::Max(ws->d_reduce_tmp,
                               ws->d_reduce_tmp_bytes,
                               ws->d_row_block_counts,
                               ws->d_ell_width,
                               row_blocks,
                               ws->stream) != cudaSuccess) {
        std::fprintf(stderr, "CUB error at blocked_ell_from_coo ell-width reduce eval\n");
        return 0;
    }
    if (!detail::blocked_ell_from_coo_cuda_check(cudaMemcpyAsync(&ell_width,
                                                                 ws->d_ell_width,
                                                                 sizeof(types::u32),
                                                                 cudaMemcpyDeviceToHost,
                                                                 ws->stream),
                                                 "cudaMemcpyAsync blocked_ell_from_coo ell_width eval")) return 0;
    if (!detail::blocked_ell_from_coo_cuda_check(cudaMemcpyAsync(&total_blocks_device,
                                                                 ws->d_row_block_offsets + row_blocks,
                                                                 sizeof(types::ptr_t),
                                                                 cudaMemcpyDeviceToHost,
                                                                 ws->stream),
                                                 "cudaMemcpyAsync blocked_ell_from_coo total_blocks eval")) return 0;
    if (!detail::blocked_ell_from_coo_cuda_check(cudaStreamSynchronize(ws->stream), "cudaStreamSynchronize blocked_ell_from_coo eval")) return 0;
    *ell_width_out = ell_width;
    total_blocks = (types::u64) total_blocks_device;
    *total_blocks_out = total_blocks;
    return 1;
}

__host__ __forceinline__ int choose_blocked_ell_block_size_from_coo_cuda(
    const sparse::coo *src,
    types::dim_t cols,
    const types::u32 *feature_to_global,
    const unsigned int *candidates,
    unsigned int candidate_count,
    int device,
    blocked_ell_tune_result *out,
    blocked_ell_from_coo_cuda_workspace *workspace = 0,
    cudaStream_t stream = (cudaStream_t) 0
) {
    blocked_ell_tune_result best = { 0u, 0.0, 0u };
    blocked_ell_from_coo_cuda_workspace local_ws;
    blocked_ell_from_coo_cuda_workspace *ws = workspace;
    int owns_ws = 0;

    if (src == 0 || out == 0 || cols == 0u || candidates == 0 || candidate_count == 0u) return 0;
    if (ws == 0) {
        init(&local_ws);
        if (!setup(&local_ws, device, stream)) return 0;
        ws = &local_ws;
        owns_ws = 1;
    } else if (ws->device != device || (stream != (cudaStream_t) 0 && ws->stream != stream)) {
        if (!setup(ws, device, stream)) return 0;
    }
    if (!upload_coo(ws, src, feature_to_global)) {
        if (owns_ws) clear(ws);
        return 0;
    }

    for (unsigned int i = 0u; i < candidate_count; ++i) {
        const unsigned int block_size = candidates[i];
        const types::dim_t row_blocks = block_size == 0u ? 0u : (src->rows + block_size - 1u) / block_size;
        types::u32 ell_width = 0u;
        types::u64 total_blocks = 0u;
        std::size_t padded_bytes = 0u;
        double fill_ratio = 1.0;
        if (block_size == 0u) continue;
        if (!evaluate_block_size(ws, src->rows, src->nnz, block_size, &ell_width, &total_blocks)) {
            if (owns_ws) clear(ws);
            return 0;
        }
        padded_bytes = (std::size_t) src->rows * (std::size_t) ell_width * (std::size_t) block_size * sizeof(real::storage_t);
        fill_ratio = (row_blocks == 0u || ell_width == 0u)
            ? 1.0
            : (double) total_blocks / (double) ((types::u64) row_blocks * (types::u64) ell_width);
        if (best.block_size == 0u
            || padded_bytes + 1u < best.padded_bytes
            || (padded_bytes == best.padded_bytes && fill_ratio > best.fill_ratio + 1.0e-9)
            || (padded_bytes == best.padded_bytes && fill_ratio + 1.0e-9 >= best.fill_ratio && block_size > best.block_size)) {
            best.block_size = block_size;
            best.fill_ratio = fill_ratio;
            best.padded_bytes = padded_bytes;
        }
    }

    if (owns_ws) clear(ws);
    if (best.block_size == 0u) return 0;
    *out = best;
    return 1;
}

__host__ __forceinline__ int blocked_ell_from_coo_cuda(
    const sparse::coo *src,
    types::dim_t cols,
    const types::u32 *feature_to_global,
    unsigned int block_size,
    sparse::blocked_ell *dst,
    int device,
    blocked_ell_from_coo_cuda_workspace *workspace = 0,
    cudaStream_t stream = (cudaStream_t) 0
) {
    blocked_ell_from_coo_cuda_workspace local_ws;
    blocked_ell_from_coo_cuda_workspace *ws = workspace;
    int owns_ws = 0;
    const dim3 block(256u, 1u, 1u);
    const dim3 grid = blocked_ell_from_coo_grid(src != 0 ? src->nnz : 0u);
    const types::dim_t row_blocks = (src != 0 && block_size != 0u) ? (src->rows + block_size - 1u) / block_size : 0u;
    types::u32 ell_width = 0u;
    types::u64 total_blocks = 0u;
    types::idx_t block_col_count = 0u;
    types::nnz_t actual_nnz = 0u;

    if (src == 0 || dst == 0 || cols == 0u || block_size == 0u) return 0;
    if (ws == 0) {
        init(&local_ws);
        if (!setup(&local_ws, device, stream)) return 0;
        ws = &local_ws;
        owns_ws = 1;
    } else if (ws->device != device || (stream != (cudaStream_t) 0 && ws->stream != stream)) {
        if (!setup(ws, device, stream)) return 0;
    }
    if (!upload_coo(ws, src, feature_to_global)) {
        if (owns_ws) clear(ws);
        return 0;
    }
    if (!evaluate_block_size(ws, src->rows, src->nnz, block_size, &ell_width, &total_blocks)) {
        if (owns_ws) clear(ws);
        return 0;
    }
    block_col_count = (types::idx_t) row_blocks * (types::idx_t) ell_width;

    sparse::clear(dst);
    sparse::init(dst, src->rows, cols, src->nnz, block_size, ell_width * block_size);
    if (!sparse::allocate(dst)) {
        if (owns_ws) clear(ws);
        return 0;
    }
    if (src->rows == 0u || src->nnz == 0u || ell_width == 0u) {
        dst->nnz = 0u;
        if (owns_ws) clear(ws);
        return 1;
    }

    if (!reserve_output(ws, src->rows, ell_width, block_size)) {
        if (owns_ws) clear(ws);
        sparse::clear(dst);
        return 0;
    }

    detail::kernels::build_block_sort_keys_kernel<<<grid, block, 0, ws->stream>>>(
        src->nnz,
        ws->d_row_idx,
        ws->d_col_idx,
        block_size,
        ws->d_sort_keys_in,
        ws->d_sort_pos_in
    );
    if (!detail::blocked_ell_from_coo_cuda_check(cudaGetLastError(), "build_block_sort_keys_kernel final")) {
        if (owns_ws) clear(ws);
        sparse::clear(dst);
        return 0;
    }
    if (!detail::blocked_ell_from_coo_cuda_check(cudaMemsetAsync(ws->d_row_block_counts,
                                                                 0,
                                                                 (std::size_t) (row_blocks + 1u) * sizeof(types::idx_t),
                                                                 ws->stream),
                                                 "cudaMemsetAsync blocked_ell_from_coo row_block_counts final")) {
        if (owns_ws) clear(ws);
        sparse::clear(dst);
        return 0;
    }
    if (cub::DeviceRadixSort::SortPairs(ws->d_sort_tmp,
                                        ws->d_sort_tmp_bytes,
                                        ws->d_sort_keys_in,
                                        ws->d_sort_keys_out,
                                        ws->d_sort_pos_in,
                                        ws->d_sort_pos_out,
                                        src->nnz,
                                        0,
                                        sizeof(types::u64) * 8,
                                        ws->stream) != cudaSuccess) {
        std::fprintf(stderr, "CUB error at blocked_ell_from_coo sort pairs final\n");
        if (owns_ws) clear(ws);
        sparse::clear(dst);
        return 0;
    }
    detail::kernels::mark_block_group_heads_and_count_kernel<<<grid, block, 0, ws->stream>>>(
        src->nnz,
        ws->d_sort_keys_out,
        ws->d_head_flags,
        ws->d_row_block_counts
    );
    if (!detail::blocked_ell_from_coo_cuda_check(cudaGetLastError(), "mark_block_group_heads_and_count_kernel final")) {
        if (owns_ws) clear(ws);
        sparse::clear(dst);
        return 0;
    }
    if (cub::DeviceScan::ExclusiveSum(ws->d_scan_tmp,
                                      ws->d_scan_tmp_bytes,
                                      ws->d_head_flags,
                                      ws->d_group_ids,
                                      src->nnz,
                                      ws->stream) != cudaSuccess) {
        std::fprintf(stderr, "CUB error at blocked_ell_from_coo group scan final\n");
        if (owns_ws) clear(ws);
        sparse::clear(dst);
        return 0;
    }
    if (cub::DeviceScan::ExclusiveSum(ws->d_scan_tmp,
                                      ws->d_scan_tmp_bytes,
                                      ws->d_row_block_counts,
                                      ws->d_row_block_offsets,
                                      (std::size_t) row_blocks + 1u,
                                      ws->stream) != cudaSuccess) {
        std::fprintf(stderr, "CUB error at blocked_ell_from_coo row-block scan final\n");
        if (owns_ws) clear(ws);
        sparse::clear(dst);
        return 0;
    }
    if (!detail::blocked_ell_from_coo_cuda_check(cudaMemsetAsync(ws->d_output_val,
                                                                 0,
                                                                 (std::size_t) src->rows * (std::size_t) (ell_width * block_size) * sizeof(real::storage_t),
                                                                 ws->stream),
                                                 "cudaMemsetAsync blocked_ell_from_coo output val")) {
        if (owns_ws) clear(ws);
        sparse::clear(dst);
        return 0;
    }
    detail::kernels::fill_invalid_block_cols_kernel<<<blocked_ell_from_coo_grid(block_col_count), block, 0, ws->stream>>>(
        block_col_count,
        ws->d_output_block_cols
    );
    if (!detail::blocked_ell_from_coo_cuda_check(cudaGetLastError(), "fill_invalid_block_cols_kernel")) {
        if (owns_ws) clear(ws);
        sparse::clear(dst);
        return 0;
    }
    detail::kernels::scatter_block_col_ids_kernel<<<grid, block, 0, ws->stream>>>(
        src->nnz,
        ws->d_sort_keys_out,
        ws->d_head_flags,
        ws->d_group_ids,
        ws->d_row_block_offsets,
        ell_width,
        ws->d_output_block_cols
    );
    if (!detail::blocked_ell_from_coo_cuda_check(cudaGetLastError(), "scatter_block_col_ids_kernel")) {
        if (owns_ws) clear(ws);
        sparse::clear(dst);
        return 0;
    }
    detail::kernels::scatter_blocked_values_kernel<<<grid, block, 0, ws->stream>>>(
        src->nnz,
        ws->d_sort_keys_out,
        ws->d_sort_pos_out,
        ws->d_group_ids,
        ws->d_row_block_offsets,
        ws->d_row_idx,
        ws->d_col_idx,
        ws->d_val,
        block_size,
        ell_width * block_size,
        ws->d_output_val
    );
    if (!detail::blocked_ell_from_coo_cuda_check(cudaGetLastError(), "scatter_blocked_values_kernel")) {
        if (owns_ws) clear(ws);
        sparse::clear(dst);
        return 0;
    }
    if (!detail::blocked_ell_from_coo_cuda_check(cudaMemsetAsync(ws->d_output_nnz,
                                                                 0,
                                                                 sizeof(types::nnz_t),
                                                                 ws->stream),
                                                 "cudaMemsetAsync blocked_ell_from_coo output nnz")) {
        if (owns_ws) clear(ws);
        sparse::clear(dst);
        return 0;
    }
    {
        const types::u64 value_count = (types::u64) src->rows * (types::u64) dst->ell_cols;
        detail::kernels::count_nonzero_blocked_values_kernel<<<blocked_ell_from_coo_grid(value_count), block, 0, ws->stream>>>(
            value_count,
            src->rows,
            cols,
            block_size,
            ell_width,
            ws->d_output_block_cols,
            ws->d_output_val,
            ws->d_output_nnz
        );
        if (!detail::blocked_ell_from_coo_cuda_check(cudaGetLastError(), "count_nonzero_blocked_values_kernel")) {
            if (owns_ws) clear(ws);
            sparse::clear(dst);
            return 0;
        }
    }
    if (dst->blockColIdx != 0 && block_col_count != 0u) {
        if (!detail::blocked_ell_from_coo_cuda_check(cudaMemcpyAsync(dst->blockColIdx,
                                                                     ws->d_output_block_cols,
                                                                     (std::size_t) block_col_count * sizeof(types::idx_t),
                                                                     cudaMemcpyDeviceToHost,
                                                                     ws->stream),
                                                     "cudaMemcpyAsync blocked_ell_from_coo block cols")) {
            if (owns_ws) clear(ws);
            sparse::clear(dst);
            return 0;
        }
    }
    if (dst->val != 0 && src->rows != 0u && dst->ell_cols != 0u) {
        if (!detail::blocked_ell_from_coo_cuda_check(cudaMemcpyAsync(dst->val,
                                                                     ws->d_output_val,
                                                                     (std::size_t) src->rows * (std::size_t) dst->ell_cols * sizeof(real::storage_t),
                                                                     cudaMemcpyDeviceToHost,
                                                                     ws->stream),
                                                     "cudaMemcpyAsync blocked_ell_from_coo values")) {
            if (owns_ws) clear(ws);
            sparse::clear(dst);
            return 0;
        }
    }
    if (!detail::blocked_ell_from_coo_cuda_check(cudaMemcpyAsync(&actual_nnz,
                                                                 ws->d_output_nnz,
                                                                 sizeof(types::nnz_t),
                                                                 cudaMemcpyDeviceToHost,
                                                                 ws->stream),
                                                 "cudaMemcpyAsync blocked_ell_from_coo output nnz")) {
        if (owns_ws) clear(ws);
        sparse::clear(dst);
        return 0;
    }
    if (!detail::blocked_ell_from_coo_cuda_check(cudaStreamSynchronize(ws->stream), "cudaStreamSynchronize blocked_ell_from_coo final")) {
        if (owns_ws) clear(ws);
        sparse::clear(dst);
        return 0;
    }
    dst->nnz = actual_nnz;
    if (owns_ws) clear(ws);
    return 1;
}

__host__ __forceinline__ int blocked_ell_from_coo_cuda_auto(
    const sparse::coo *src,
    types::dim_t cols,
    const types::u32 *feature_to_global,
    const unsigned int *candidates,
    unsigned int candidate_count,
    sparse::blocked_ell *dst,
    int device,
    blocked_ell_tune_result *picked = 0,
    blocked_ell_from_coo_cuda_workspace *workspace = 0,
    cudaStream_t stream = (cudaStream_t) 0
) {
    blocked_ell_tune_result tune = { 0u, 0.0, 0u };
    blocked_ell_from_coo_cuda_workspace local_ws;
    blocked_ell_from_coo_cuda_workspace *ws = workspace;
    int owns_ws = 0;
    if (src == 0 || dst == 0 || cols == 0u || candidates == 0 || candidate_count == 0u) return 0;
    if (ws == 0) {
        init(&local_ws);
        if (!setup(&local_ws, device, stream)) return 0;
        ws = &local_ws;
        owns_ws = 1;
    } else if (ws->device != device || (stream != (cudaStream_t) 0 && ws->stream != stream)) {
        if (!setup(ws, device, stream)) return 0;
    }
    if (!choose_blocked_ell_block_size_from_coo_cuda(src,
                                                     cols,
                                                     feature_to_global,
                                                     candidates,
                                                     candidate_count,
                                                     device,
                                                     &tune,
                                                     ws,
                                                     ws->stream)) {
        if (owns_ws) clear(ws);
        return 0;
    }
    if (!blocked_ell_from_coo_cuda(src,
                                   cols,
                                   feature_to_global,
                                   tune.block_size,
                                   dst,
                                   device,
                                   ws,
                                   ws->stream)) {
        if (owns_ws) clear(ws);
        return 0;
    }
    if (picked != 0) *picked = tune;
    if (owns_ws) clear(ws);
    return 1;
}

} // namespace convert
} // namespace cellshard
