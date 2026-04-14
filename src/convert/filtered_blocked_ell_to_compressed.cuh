#pragma once

#include "../bucket/routes/compressed_major_nnz.cuh"
#include "../sharded/sharded_device.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace cellshard {
namespace convert {

namespace detail {

static inline int filtered_blocked_ell_cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

static inline std::size_t align_up_bytes_(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

namespace kernels {

__global__ static void count_filtered_blocked_ell_row_nnz(
    const device::blocked_ell_view src,
    const unsigned int output_rows,
    const unsigned int output_cols,
    const unsigned char * __restrict__ keep_rows,
    const unsigned char * __restrict__ keep_cols,
    const unsigned int * __restrict__ row_remap,
    const unsigned int * __restrict__ col_remap,
    unsigned int * __restrict__ row_ptr_shifted
) {
    const unsigned int tid = (unsigned int) ::cellshard::ptx::global_tid_1d();
    const unsigned int stride = (unsigned int) ::cellshard::ptx::global_stride_1d();
    const unsigned int block_size = src.block_size;
    const unsigned int ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
    unsigned int row = tid;

    while (row < src.rows) {
        if (keep_rows == 0 || keep_rows[row] != 0u) {
            const unsigned int out_row = row_remap != 0 ? row_remap[row] : row;
            unsigned int count = 0u;
            if (out_row < output_rows) {
                const unsigned int row_block = block_size != 0u ? row / block_size : 0u;
                for (unsigned int ell_col = 0u; ell_col < src.ell_cols; ++ell_col) {
                    const unsigned int slot = block_size != 0u ? ell_col / block_size : 0u;
                    const unsigned int lane = block_size != 0u ? ell_col % block_size : 0u;
                    const unsigned int block_col = ell_width_blocks != 0u
                        ? src.blockColIdx[(unsigned long) row_block * ell_width_blocks + slot]
                        : sparse::blocked_ell_invalid_col;
                    const unsigned int col = block_col != sparse::blocked_ell_invalid_col
                        ? block_col * block_size + lane
                        : src.cols;
                    const __half value = src.val[(unsigned long) row * src.ell_cols + ell_col];
                    if (col >= src.cols || __half2float(value) == 0.0f) continue;
                    if (keep_cols != 0 && keep_cols[col] == 0u) continue;
                    if (col_remap != 0) {
                        if (col_remap[col] >= output_cols) continue;
                    } else if (col >= output_cols) {
                        continue;
                    }
                    ++count;
                }
                row_ptr_shifted[out_row + 1u] = count;
            }
        }
        row += stride;
    }
}

__global__ static void emit_filtered_blocked_ell_compressed(
    const device::blocked_ell_view src,
    const unsigned int output_rows,
    const unsigned int output_cols,
    const unsigned char * __restrict__ keep_rows,
    const unsigned char * __restrict__ keep_cols,
    const unsigned int * __restrict__ row_remap,
    const unsigned int * __restrict__ col_remap,
    const unsigned int * __restrict__ dst_row_ptr,
    unsigned int * __restrict__ dst_minor_idx,
    __half * __restrict__ dst_val
) {
    const unsigned int tid = (unsigned int) ::cellshard::ptx::global_tid_1d();
    const unsigned int stride = (unsigned int) ::cellshard::ptx::global_stride_1d();
    const unsigned int block_size = src.block_size;
    const unsigned int ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
    unsigned int row = tid;

    while (row < src.rows) {
        if (keep_rows == 0 || keep_rows[row] != 0u) {
            const unsigned int out_row = row_remap != 0 ? row_remap[row] : row;
            if (out_row < output_rows) {
                unsigned int cursor = dst_row_ptr[out_row];
                const unsigned int row_block = block_size != 0u ? row / block_size : 0u;
                for (unsigned int ell_col = 0u; ell_col < src.ell_cols; ++ell_col) {
                    const unsigned int slot = block_size != 0u ? ell_col / block_size : 0u;
                    const unsigned int lane = block_size != 0u ? ell_col % block_size : 0u;
                    const unsigned int block_col = ell_width_blocks != 0u
                        ? src.blockColIdx[(unsigned long) row_block * ell_width_blocks + slot]
                        : sparse::blocked_ell_invalid_col;
                    const unsigned int col = block_col != sparse::blocked_ell_invalid_col
                        ? block_col * block_size + lane
                        : src.cols;
                    const __half value = src.val[(unsigned long) row * src.ell_cols + ell_col];
                    unsigned int out_col = 0u;
                    if (col >= src.cols || __half2float(value) == 0.0f) continue;
                    if (keep_cols != 0 && keep_cols[col] == 0u) continue;
                    out_col = col_remap != 0 ? col_remap[col] : col;
                    if (out_col >= output_cols) continue;
                    dst_minor_idx[cursor] = out_col;
                    dst_val[cursor] = value;
                    ++cursor;
                }
            }
        }
        row += stride;
    }
}

} // namespace kernels
} // namespace detail

struct alignas(16) filtered_blocked_ell_stats {
    types::dim_t kept_rows;
    types::dim_t output_rows;
    types::dim_t output_cols;
    types::nnz_t live_nnz;
    types::u64 kept_row_slots;
    types::u64 dead_slots;
    double live_fill_ratio;
    std::size_t dead_value_bytes;
};

struct alignas(16) filtered_blocked_ell_result {
    device::compressed_view filtered;
    bucket::compressed_major_bucket_result bucketed;
    filtered_blocked_ell_stats stats;
    int has_bucketed;
};

struct alignas(16) filtered_blocked_ell_workspace {
    int device;
    cudaStream_t stream;
    int owns_stream;

    types::dim_t rows_capacity;
    types::dim_t filtered_rows_capacity;
    types::dim_t cols_capacity;
    types::nnz_t nnz_capacity;
    types::u64 total_values_capacity;

    void *d_counts_block;
    void *d_values_block;
    void *d_scan_tmp;
    std::size_t d_scan_tmp_bytes;

    types::ptr_t *d_filtered_major_ptr;
    types::idx_t *d_filtered_minor_idx;
    real::storage_t *d_filtered_val;

    bucket::compressed_major_bucket_workspace bucket_ws;
};

__host__ __forceinline__ void init(filtered_blocked_ell_workspace *ws) {
    std::memset(ws, 0, sizeof(*ws));
    ws->device = -1;
    bucket::init(&ws->bucket_ws);
}

__host__ __forceinline__ void clear(filtered_blocked_ell_workspace *ws) {
    if (ws->device >= 0) cudaSetDevice(ws->device);
    bucket::clear(&ws->bucket_ws);
    if (ws->d_scan_tmp != 0) cudaFree(ws->d_scan_tmp);
    if (ws->d_values_block != 0) cudaFree(ws->d_values_block);
    if (ws->d_counts_block != 0) cudaFree(ws->d_counts_block);
    if (ws->owns_stream && ws->stream != (cudaStream_t) 0) cudaStreamDestroy(ws->stream);
    init(ws);
}

__host__ __forceinline__ int setup(filtered_blocked_ell_workspace *ws, int device, cudaStream_t stream = (cudaStream_t) 0) {
    clear(ws);
    if (!detail::filtered_blocked_ell_cuda_check(cudaSetDevice(device), "cudaSetDevice filtered blocked ell workspace")) return 0;
    ws->device = device;
    if (stream == (cudaStream_t) 0) {
        if (!detail::filtered_blocked_ell_cuda_check(cudaStreamCreateWithFlags(&ws->stream, cudaStreamNonBlocking),
                                                     "cudaStreamCreateWithFlags filtered blocked ell")) return 0;
        ws->owns_stream = 1;
    } else {
        ws->stream = stream;
        ws->owns_stream = 0;
    }
    return bucket::setup(&ws->bucket_ws, device);
}

__host__ __forceinline__ int reserve(filtered_blocked_ell_workspace *ws,
                                     types::dim_t src_rows,
                                     types::dim_t filtered_rows,
                                     types::dim_t filtered_cols,
                                     types::nnz_t max_nnz,
                                     types::u64 total_values) {
    if (!detail::filtered_blocked_ell_cuda_check(cudaSetDevice(ws->device >= 0 ? ws->device : 0),
                                                 "cudaSetDevice filtered blocked ell reserve")) return 0;

    if (filtered_rows + 1u > ws->filtered_rows_capacity + 1u) {
        if (ws->d_counts_block != 0) cudaFree(ws->d_counts_block);
        ws->d_counts_block = 0;
        ws->d_filtered_major_ptr = 0;
        if (filtered_rows + 1u != 0u) {
            const std::size_t bytes = (std::size_t) (filtered_rows + 1u) * sizeof(types::ptr_t);
            if (!detail::filtered_blocked_ell_cuda_check(cudaMalloc(&ws->d_counts_block, bytes), "cudaMalloc filtered blocked ell row ptr")) return 0;
            ws->d_filtered_major_ptr = (types::ptr_t *) ws->d_counts_block;
        }
        ws->filtered_rows_capacity = filtered_rows;
    }

    if (max_nnz > ws->nnz_capacity) {
        if (ws->d_values_block != 0) cudaFree(ws->d_values_block);
        ws->d_values_block = 0;
        ws->d_filtered_minor_idx = 0;
        ws->d_filtered_val = 0;
        if (max_nnz != 0u) {
            const std::size_t val_offset = detail::align_up_bytes_((std::size_t) max_nnz * sizeof(types::idx_t), alignof(real::storage_t));
            const std::size_t total_bytes = val_offset + (std::size_t) max_nnz * sizeof(real::storage_t);
            if (!detail::filtered_blocked_ell_cuda_check(cudaMalloc(&ws->d_values_block, total_bytes), "cudaMalloc filtered blocked ell values")) return 0;
            ws->d_filtered_minor_idx = (types::idx_t *) ws->d_values_block;
            ws->d_filtered_val = (real::storage_t *) ((char *) ws->d_values_block + val_offset);
        }
        ws->nnz_capacity = max_nnz;
    }

    {
        std::size_t scan_bytes = 0;
        if (cub::DeviceScan::ExclusiveSum(0,
                                          scan_bytes,
                                          ws->d_filtered_major_ptr,
                                          ws->d_filtered_major_ptr,
                                          (std::size_t) filtered_rows + 1u,
                                          ws->stream) != cudaSuccess) {
            std::fprintf(stderr, "CUB error at filtered blocked ell scan sizing\n");
            return 0;
        }
        if (scan_bytes > ws->d_scan_tmp_bytes) {
            if (ws->d_scan_tmp != 0) cudaFree(ws->d_scan_tmp);
            ws->d_scan_tmp = 0;
            if (scan_bytes != 0u && !detail::filtered_blocked_ell_cuda_check(cudaMalloc(&ws->d_scan_tmp, scan_bytes),
                                                                              "cudaMalloc filtered blocked ell scan tmp")) return 0;
            ws->d_scan_tmp_bytes = scan_bytes;
        }
    }

    ws->rows_capacity = src_rows;
    ws->cols_capacity = filtered_cols;
    ws->total_values_capacity = total_values;
    return 1;
}

__host__ __forceinline__ int filter_blocked_ell_to_compressed(
    const device::blocked_ell_view *src,
    types::dim_t output_rows,
    types::dim_t output_cols,
    const unsigned char *d_keep_rows,
    const unsigned char *d_keep_cols,
    const types::idx_t *d_row_remap,
    const types::idx_t *d_col_remap,
    filtered_blocked_ell_workspace *ws,
    filtered_blocked_ell_result *out
) {
    filtered_blocked_ell_result local{};
    unsigned int live_nnz = 0u;
    unsigned int kept_rows = output_rows;
    const types::u64 total_values = src != 0 ? (types::u64) src->rows * (types::u64) src->ell_cols : 0u;
    types::u64 kept_row_slots = 0u;
    dim3 grid;
    dim3 block;

    if (src == 0 || ws == 0) return 0;
    if (!reserve(ws, src->rows, output_rows, output_cols, src->nnz, total_values)) return 0;
    if (output_rows == 0u || output_cols == 0u) {
        local.filtered.rows = output_rows;
        local.filtered.cols = output_cols;
        local.filtered.nnz = 0u;
        local.filtered.axis = sparse::compressed_by_row;
        local.filtered.majorPtr = ws->d_filtered_major_ptr;
        local.filtered.minorIdx = ws->d_filtered_minor_idx;
        local.filtered.val = ws->d_filtered_val;
        local.stats.kept_rows = kept_rows;
        local.stats.output_rows = output_rows;
        local.stats.output_cols = output_cols;
        local.stats.live_nnz = 0u;
        local.stats.kept_row_slots = 0u;
        local.stats.dead_slots = 0u;
        local.stats.live_fill_ratio = 1.0;
        local.stats.dead_value_bytes = 0u;
        if (out != 0) *out = local;
        return 1;
    }

    if (!detail::filtered_blocked_ell_cuda_check(cudaMemsetAsync(ws->d_filtered_major_ptr,
                                                                 0,
                                                                 (std::size_t) (output_rows + 1u) * sizeof(types::ptr_t),
                                                                 ws->stream),
                                                 "cudaMemsetAsync filtered blocked ell row ptr")) return 0;

    block = dim3(128u, 1u, 1u);
    grid = dim3((src->rows + block.x - 1u) / block.x, 1u, 1u);
    if (grid.x == 0u) grid.x = 1u;
    if (grid.x > 4096u) grid.x = 4096u;

    detail::kernels::count_filtered_blocked_ell_row_nnz<<<grid, block, 0, ws->stream>>>(
        *src,
        output_rows,
        output_cols,
        d_keep_rows,
        d_keep_cols,
        d_row_remap,
        d_col_remap,
        ws->d_filtered_major_ptr
    );
    if (!detail::filtered_blocked_ell_cuda_check(cudaGetLastError(), "count_filtered_blocked_ell_row_nnz")) return 0;

    if (cub::DeviceScan::ExclusiveSum(ws->d_scan_tmp,
                                      ws->d_scan_tmp_bytes,
                                      ws->d_filtered_major_ptr,
                                      ws->d_filtered_major_ptr,
                                      (std::size_t) output_rows + 1u,
                                      ws->stream) != cudaSuccess) {
        std::fprintf(stderr, "CUB error at filtered blocked ell row scan\n");
        return 0;
    }

    if (!detail::filtered_blocked_ell_cuda_check(cudaMemcpyAsync(&live_nnz,
                                                                 ws->d_filtered_major_ptr + output_rows,
                                                                 sizeof(unsigned int),
                                                                 cudaMemcpyDeviceToHost,
                                                                 ws->stream),
                                                 "cudaMemcpyAsync filtered blocked ell nnz")) return 0;
    if (!detail::filtered_blocked_ell_cuda_check(cudaStreamSynchronize(ws->stream),
                                                 "cudaStreamSynchronize filtered blocked ell scan")) return 0;
    if (live_nnz > src->nnz) return 0;

    detail::kernels::emit_filtered_blocked_ell_compressed<<<grid, block, 0, ws->stream>>>(
        *src,
        output_rows,
        output_cols,
        d_keep_rows,
        d_keep_cols,
        d_row_remap,
        d_col_remap,
        ws->d_filtered_major_ptr,
        ws->d_filtered_minor_idx,
        ws->d_filtered_val
    );
    if (!detail::filtered_blocked_ell_cuda_check(cudaGetLastError(), "emit_filtered_blocked_ell_compressed")) return 0;

    kept_row_slots = (types::u64) kept_rows * (types::u64) src->ell_cols;
    local.filtered.rows = output_rows;
    local.filtered.cols = output_cols;
    local.filtered.nnz = live_nnz;
    local.filtered.axis = sparse::compressed_by_row;
    local.filtered.majorPtr = ws->d_filtered_major_ptr;
    local.filtered.minorIdx = ws->d_filtered_minor_idx;
    local.filtered.val = ws->d_filtered_val;
    local.stats.kept_rows = kept_rows;
    local.stats.output_rows = output_rows;
    local.stats.output_cols = output_cols;
    local.stats.live_nnz = live_nnz;
    local.stats.kept_row_slots = kept_row_slots;
    local.stats.dead_slots = kept_row_slots > (types::u64) live_nnz ? kept_row_slots - (types::u64) live_nnz : 0u;
    local.stats.live_fill_ratio = kept_row_slots == 0u ? 1.0 : (double) live_nnz / (double) kept_row_slots;
    local.stats.dead_value_bytes = (std::size_t) local.stats.dead_slots * sizeof(real::storage_t);
    local.has_bucketed = 0;

    if (out != 0) *out = local;
    return 1;
}

__host__ __forceinline__ int build_bucketed_filtered_blocked_ell_major_view(
    const device::blocked_ell_view *src,
    types::dim_t output_rows,
    types::dim_t output_cols,
    const unsigned char *d_keep_rows,
    const unsigned char *d_keep_cols,
    const types::idx_t *d_row_remap,
    const types::idx_t *d_col_remap,
    types::idx_t requested_bucket_count,
    filtered_blocked_ell_workspace *ws,
    filtered_blocked_ell_result *out
) {
    filtered_blocked_ell_result local{};

    if (ws == 0) return 0;
    if (!filter_blocked_ell_to_compressed(src,
                                          output_rows,
                                          output_cols,
                                          d_keep_rows,
                                          d_keep_cols,
                                          d_row_remap,
                                          d_col_remap,
                                          ws,
                                          &local)) return 0;
    if (!bucket::build_bucketed_major_view(&local.filtered,
                                           requested_bucket_count,
                                           &ws->bucket_ws,
                                           &local.bucketed)) return 0;
    local.has_bucketed = 1;
    if (out != 0) *out = local;
    return 1;
}

} // namespace convert
} // namespace cellshard
