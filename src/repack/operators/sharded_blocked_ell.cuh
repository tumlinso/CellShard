#pragma once

#include "../kernels/sharded_blocked_ell.cuh"
#include "../../bucket/operators/major_nnz.cuh"
#include "../../convert/coo_from_compressed_raw.cuh"
#include "../../sharded/sharded_device.cuh"

#include <cub/cub.cuh>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cellshard {
namespace repack {

struct alignas(16) sharded_blocked_ell_repack_config {
    types::dim_t output_rows;
    types::dim_t output_cols;
    types::u32 block_size;
    types::idx_t requested_bucket_count;
};

struct alignas(16) sharded_blocked_ell_repack_stats {
    types::dim_t kept_rows;
    types::dim_t output_rows;
    types::dim_t output_cols;
    types::nnz_t live_nnz;
    types::u64 kept_row_slots;
    types::u64 dead_slots;
    double live_fill_ratio;
    std::size_t dead_value_bytes;
    types::u32 ell_width;
};

struct alignas(16) sharded_blocked_ell_repack_result {
    device::compressed_view filtered;
    bucket::compressed_major_bucket_result bucketed;
    device::blocked_ell_view rebuilt;
    bucket::major_nnz_bucket_plan_view bucket_plan;
    sharded_blocked_ell_repack_stats stats;
    unsigned long shard_id;
};

struct alignas(16) sharded_blocked_ell_repack_workspace {
    int device;
    cudaStream_t stream;
    int owns_stream;

    unsigned long part_capacity;
    void *h_part_block;
    void *d_part_block;
    const device::blocked_ell_view **h_part_views;
    const device::blocked_ell_view **d_part_views;
    types::idx_t *h_part_row_offsets;
    types::idx_t *d_part_row_offsets;

    types::dim_t filtered_rows_capacity;
    types::dim_t filtered_cols_capacity;
    types::nnz_t filtered_nnz_capacity;
    void *d_filtered_block;
    types::ptr_t *d_filtered_major_ptr;
    types::idx_t *d_filtered_minor_idx;
    real::storage_t *d_filtered_val;
    void *d_filtered_scan_tmp;
    std::size_t d_filtered_scan_tmp_bytes;
    void *d_stats_block;
    unsigned long long *d_kept_rows;
    unsigned long long *d_kept_row_slots;

    bucket::compressed_major_bucket_workspace bucket_ws;

    types::nnz_t repack_nnz_capacity;
    types::dim_t row_block_capacity;
    std::size_t sort_capacity;
    std::size_t scan_capacity;
    std::size_t reduce_capacity;
    void *d_repack_block;
    types::idx_t *d_row_idx;
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
    void *d_scan_tmp;
    void *d_reduce_tmp;

    std::size_t rebuilt_storage_capacity_bytes;
    void *d_rebuilt_storage;
    types::idx_t *d_rebuilt_block_cols;
    real::storage_t *d_rebuilt_val;
};

namespace detail {

static inline int repack_cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

static inline std::size_t align_up_bytes_(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

} // namespace detail

__host__ __forceinline__ void init(sharded_blocked_ell_repack_workspace *ws) {
    std::memset(ws, 0, sizeof(*ws));
    ws->device = -1;
    bucket::init(&ws->bucket_ws);
}

__host__ __forceinline__ void clear(sharded_blocked_ell_repack_workspace *ws) {
    if (ws->device >= 0) cudaSetDevice(ws->device);
    bucket::clear(&ws->bucket_ws);
    if (ws->d_rebuilt_storage != 0) cudaFree(ws->d_rebuilt_storage);
    if (ws->d_reduce_tmp != 0) cudaFree(ws->d_reduce_tmp);
    if (ws->d_scan_tmp != 0) cudaFree(ws->d_scan_tmp);
    if (ws->d_sort_tmp != 0) cudaFree(ws->d_sort_tmp);
    if (ws->d_repack_block != 0) cudaFree(ws->d_repack_block);
    if (ws->d_stats_block != 0) cudaFree(ws->d_stats_block);
    if (ws->d_filtered_scan_tmp != 0) cudaFree(ws->d_filtered_scan_tmp);
    if (ws->d_filtered_block != 0) cudaFree(ws->d_filtered_block);
    if (ws->h_part_block != 0) cudaFreeHost(ws->h_part_block);
    if (ws->d_part_block != 0) cudaFree(ws->d_part_block);
    if (ws->owns_stream && ws->stream != (cudaStream_t) 0) cudaStreamDestroy(ws->stream);
    init(ws);
}

__host__ __forceinline__ int setup(sharded_blocked_ell_repack_workspace *ws, int device, cudaStream_t stream = (cudaStream_t) 0) {
    clear(ws);
    if (!detail::repack_cuda_check(cudaSetDevice(device), "cudaSetDevice repack workspace")) return 0;
    ws->device = device;
    if (stream == (cudaStream_t) 0) {
        if (!detail::repack_cuda_check(cudaStreamCreateWithFlags(&ws->stream, cudaStreamNonBlocking),
                                       "cudaStreamCreateWithFlags repack")) return 0;
        ws->owns_stream = 1;
    } else {
        ws->stream = stream;
        ws->owns_stream = 0;
    }
    return bucket::setup(&ws->bucket_ws, device);
}

__host__ __forceinline__ int reserve_parts(sharded_blocked_ell_repack_workspace *ws, unsigned long part_count) {
    if (!detail::repack_cuda_check(cudaSetDevice(ws->device >= 0 ? ws->device : 0), "cudaSetDevice repack reserve parts")) return 0;
    if (part_count <= ws->part_capacity) return 1;

    if (ws->h_part_block != 0) cudaFreeHost(ws->h_part_block);
    if (ws->d_part_block != 0) cudaFree(ws->d_part_block);
    ws->h_part_block = 0;
    ws->d_part_block = 0;
    ws->h_part_views = 0;
    ws->d_part_views = 0;
    ws->h_part_row_offsets = 0;
    ws->d_part_row_offsets = 0;

    if (part_count != 0ul) {
        const std::size_t h_rows_offset = detail::align_up_bytes_((std::size_t) part_count * sizeof(const device::blocked_ell_view *), alignof(types::idx_t));
        const std::size_t h_total_bytes = h_rows_offset + (std::size_t) (part_count + 1ul) * sizeof(types::idx_t);
        const std::size_t d_rows_offset = detail::align_up_bytes_((std::size_t) part_count * sizeof(const device::blocked_ell_view *), alignof(types::idx_t));
        const std::size_t d_total_bytes = d_rows_offset + (std::size_t) (part_count + 1ul) * sizeof(types::idx_t);

        if (!detail::repack_cuda_check(cudaMallocHost(&ws->h_part_block, h_total_bytes), "cudaMallocHost repack h_part_block")) return 0;
        std::memset(ws->h_part_block, 0, h_total_bytes);
        ws->h_part_views = (const device::blocked_ell_view **) ws->h_part_block;
        ws->h_part_row_offsets = (types::idx_t *) ((char *) ws->h_part_block + h_rows_offset);

        if (!detail::repack_cuda_check(cudaMalloc(&ws->d_part_block, d_total_bytes), "cudaMalloc repack d_part_block")) return 0;
        ws->d_part_views = (const device::blocked_ell_view **) ws->d_part_block;
        ws->d_part_row_offsets = (types::idx_t *) ((char *) ws->d_part_block + d_rows_offset);
    }

    ws->part_capacity = part_count;
    return 1;
}

__host__ __forceinline__ int reserve_filtered(
    sharded_blocked_ell_repack_workspace *ws,
    types::dim_t output_rows,
    types::dim_t output_cols,
    types::nnz_t max_nnz
) {
    if (!detail::repack_cuda_check(cudaSetDevice(ws->device >= 0 ? ws->device : 0), "cudaSetDevice repack reserve filtered")) return 0;

    if (output_rows > ws->filtered_rows_capacity || max_nnz > ws->filtered_nnz_capacity) {
        if (ws->d_filtered_block != 0) cudaFree(ws->d_filtered_block);
        ws->d_filtered_block = 0;
        ws->d_filtered_major_ptr = 0;
        ws->d_filtered_minor_idx = 0;
        ws->d_filtered_val = 0;

        {
            const std::size_t ptr_bytes = (std::size_t) (output_rows + 1u) * sizeof(types::ptr_t);
            const std::size_t minor_offset = detail::align_up_bytes_(ptr_bytes, alignof(types::idx_t));
            const std::size_t val_offset = detail::align_up_bytes_(minor_offset + (std::size_t) max_nnz * sizeof(types::idx_t), alignof(real::storage_t));
            const std::size_t total_bytes = val_offset + (std::size_t) max_nnz * sizeof(real::storage_t);
            if (!detail::repack_cuda_check(cudaMalloc(&ws->d_filtered_block, total_bytes == 0u ? sizeof(types::ptr_t) : total_bytes),
                                           "cudaMalloc repack filtered block")) return 0;
            ws->d_filtered_major_ptr = (types::ptr_t *) ws->d_filtered_block;
            ws->d_filtered_minor_idx = (types::idx_t *) ((char *) ws->d_filtered_block + minor_offset);
            ws->d_filtered_val = (real::storage_t *) ((char *) ws->d_filtered_block + val_offset);
        }
        ws->filtered_rows_capacity = output_rows;
        ws->filtered_nnz_capacity = max_nnz;
    }

    {
        std::size_t scan_bytes = 0;
        if (cub::DeviceScan::ExclusiveSum(0,
                                          scan_bytes,
                                          ws->d_filtered_major_ptr,
                                          ws->d_filtered_major_ptr,
                                          (std::size_t) output_rows + 1u,
                                          ws->stream) != cudaSuccess) {
            std::fprintf(stderr, "CUB error at repack filtered scan sizing\n");
            return 0;
        }
        if (scan_bytes > ws->d_filtered_scan_tmp_bytes) {
            if (ws->d_filtered_scan_tmp != 0) cudaFree(ws->d_filtered_scan_tmp);
            ws->d_filtered_scan_tmp = 0;
            if (scan_bytes != 0u &&
                !detail::repack_cuda_check(cudaMalloc(&ws->d_filtered_scan_tmp, scan_bytes), "cudaMalloc repack filtered scan tmp")) return 0;
            ws->d_filtered_scan_tmp_bytes = scan_bytes;
        }
    }

    if (ws->d_stats_block == 0) {
        if (!detail::repack_cuda_check(cudaMalloc(&ws->d_stats_block, 2u * sizeof(unsigned long long)),
                                       "cudaMalloc repack stats block")) return 0;
        ws->d_kept_rows = (unsigned long long *) ws->d_stats_block;
        ws->d_kept_row_slots = ws->d_kept_rows + 1;
    }

    ws->filtered_cols_capacity = output_cols;
    return 1;
}

__host__ __forceinline__ int reserve_repack(
    sharded_blocked_ell_repack_workspace *ws,
    types::nnz_t nnz,
    types::dim_t row_blocks
) {
    std::size_t sort_bytes = 0;
    std::size_t scan_keys_bytes = 0;
    std::size_t scan_rows_bytes = 0;
    std::size_t reduce_bytes = 0;

    if (!detail::repack_cuda_check(cudaSetDevice(ws->device >= 0 ? ws->device : 0), "cudaSetDevice repack reserve blocked ell")) return 0;

    if (nnz > ws->repack_nnz_capacity || row_blocks > ws->row_block_capacity) {
        if (ws->d_repack_block != 0) cudaFree(ws->d_repack_block);
        ws->d_repack_block = 0;
        ws->d_row_idx = 0;
        ws->d_sort_keys_in = 0;
        ws->d_sort_keys_out = 0;
        ws->d_sort_pos_in = 0;
        ws->d_sort_pos_out = 0;
        ws->d_head_flags = 0;
        ws->d_group_ids = 0;
        ws->d_row_block_counts = 0;
        ws->d_row_block_offsets = 0;
        ws->d_ell_width = 0;

        {
            const std::size_t row_idx_bytes = (std::size_t) nnz * sizeof(types::idx_t);
            const std::size_t keys_in_offset = detail::align_up_bytes_(row_idx_bytes, alignof(types::u64));
            const std::size_t keys_out_offset = detail::align_up_bytes_(keys_in_offset + (std::size_t) nnz * sizeof(types::u64), alignof(types::u64));
            const std::size_t pos_in_offset = detail::align_up_bytes_(keys_out_offset + (std::size_t) nnz * sizeof(types::u64), alignof(types::idx_t));
            const std::size_t pos_out_offset = detail::align_up_bytes_(pos_in_offset + (std::size_t) nnz * sizeof(types::idx_t), alignof(types::idx_t));
            const std::size_t head_offset = detail::align_up_bytes_(pos_out_offset + (std::size_t) nnz * sizeof(types::idx_t), alignof(types::idx_t));
            const std::size_t group_offset = detail::align_up_bytes_(head_offset + (std::size_t) nnz * sizeof(types::idx_t), alignof(types::idx_t));
            const std::size_t row_count_offset = detail::align_up_bytes_(group_offset + (std::size_t) nnz * sizeof(types::idx_t), alignof(types::idx_t));
            const std::size_t row_offset_offset = detail::align_up_bytes_(row_count_offset + (std::size_t) row_blocks * sizeof(types::idx_t), alignof(types::ptr_t));
            const std::size_t ell_width_offset = detail::align_up_bytes_(row_offset_offset + (std::size_t) (row_blocks + 1u) * sizeof(types::ptr_t), alignof(types::u32));
            const std::size_t total_bytes = ell_width_offset + sizeof(types::u32);
            if (!detail::repack_cuda_check(cudaMalloc(&ws->d_repack_block, total_bytes == 0u ? sizeof(types::u32) : total_bytes),
                                           "cudaMalloc repack blocked ell block")) return 0;
            ws->d_row_idx = (types::idx_t *) ws->d_repack_block;
            ws->d_sort_keys_in = (types::u64 *) ((char *) ws->d_repack_block + keys_in_offset);
            ws->d_sort_keys_out = (types::u64 *) ((char *) ws->d_repack_block + keys_out_offset);
            ws->d_sort_pos_in = (types::idx_t *) ((char *) ws->d_repack_block + pos_in_offset);
            ws->d_sort_pos_out = (types::idx_t *) ((char *) ws->d_repack_block + pos_out_offset);
            ws->d_head_flags = (types::idx_t *) ((char *) ws->d_repack_block + head_offset);
            ws->d_group_ids = (types::idx_t *) ((char *) ws->d_repack_block + group_offset);
            ws->d_row_block_counts = (types::idx_t *) ((char *) ws->d_repack_block + row_count_offset);
            ws->d_row_block_offsets = (types::ptr_t *) ((char *) ws->d_repack_block + row_offset_offset);
            ws->d_ell_width = (types::u32 *) ((char *) ws->d_repack_block + ell_width_offset);
        }
        ws->repack_nnz_capacity = nnz;
        ws->row_block_capacity = row_blocks;
    }

    if (cub::DeviceRadixSort::SortPairs(0,
                                        sort_bytes,
                                        (const types::u64 *) 0,
                                        (types::u64 *) 0,
                                        (const types::idx_t *) 0,
                                        (types::idx_t *) 0,
                                        nnz,
                                        0,
                                        sizeof(types::u64) * 8,
                                        ws->stream) != cudaSuccess) {
        std::fprintf(stderr, "CUB error at repack sort sizing\n");
        return 0;
    }
    if (cub::DeviceScan::ExclusiveSum(0,
                                      scan_keys_bytes,
                                      ws->d_head_flags,
                                      ws->d_group_ids,
                                      nnz,
                                      ws->stream) != cudaSuccess) {
        std::fprintf(stderr, "CUB error at repack group scan sizing\n");
        return 0;
    }
    if (cub::DeviceScan::ExclusiveSum(0,
                                      scan_rows_bytes,
                                      ws->d_row_block_counts,
                                      ws->d_row_block_offsets,
                                      (std::size_t) row_blocks + 1u,
                                      ws->stream) != cudaSuccess) {
        std::fprintf(stderr, "CUB error at repack row-block scan sizing\n");
        return 0;
    }
    if (cub::DeviceReduce::Max(0,
                               reduce_bytes,
                               ws->d_row_block_counts,
                               ws->d_ell_width,
                               row_blocks,
                               ws->stream) != cudaSuccess) {
        std::fprintf(stderr, "CUB error at repack reduce sizing\n");
        return 0;
    }

    if (sort_bytes > ws->sort_capacity) {
        if (ws->d_sort_tmp != 0) cudaFree(ws->d_sort_tmp);
        ws->d_sort_tmp = 0;
        if (sort_bytes != 0u && !detail::repack_cuda_check(cudaMalloc(&ws->d_sort_tmp, sort_bytes), "cudaMalloc repack sort tmp")) return 0;
        ws->sort_capacity = sort_bytes;
    }
    if ((scan_keys_bytes > ws->scan_capacity) || (scan_rows_bytes > ws->scan_capacity)) {
        const std::size_t scan_bytes = scan_keys_bytes > scan_rows_bytes ? scan_keys_bytes : scan_rows_bytes;
        if (ws->d_scan_tmp != 0) cudaFree(ws->d_scan_tmp);
        ws->d_scan_tmp = 0;
        if (scan_bytes != 0u && !detail::repack_cuda_check(cudaMalloc(&ws->d_scan_tmp, scan_bytes), "cudaMalloc repack scan tmp")) return 0;
        ws->scan_capacity = scan_bytes;
    }
    if (reduce_bytes > ws->reduce_capacity) {
        if (ws->d_reduce_tmp != 0) cudaFree(ws->d_reduce_tmp);
        ws->d_reduce_tmp = 0;
        if (reduce_bytes != 0u && !detail::repack_cuda_check(cudaMalloc(&ws->d_reduce_tmp, reduce_bytes), "cudaMalloc repack reduce tmp")) return 0;
        ws->reduce_capacity = reduce_bytes;
    }
    return 1;
}

__host__ __forceinline__ int reserve_rebuilt_storage(
    sharded_blocked_ell_repack_workspace *ws,
    types::dim_t rows,
    types::u32 ell_width,
    types::u32 block_size
) {
    const types::dim_t row_blocks = block_size == 0u ? 0u : (rows + block_size - 1u) / block_size;
    const std::size_t idx_bytes = (std::size_t) row_blocks * (std::size_t) ell_width * sizeof(types::idx_t);
    const std::size_t val_offset = detail::align_up_bytes_(idx_bytes, alignof(real::storage_t));
    const std::size_t total_bytes = val_offset + (std::size_t) rows * (std::size_t) ell_width * (std::size_t) block_size * sizeof(real::storage_t);

    if (!detail::repack_cuda_check(cudaSetDevice(ws->device >= 0 ? ws->device : 0), "cudaSetDevice repack reserve rebuilt")) return 0;
    if (total_bytes > ws->rebuilt_storage_capacity_bytes) {
        if (ws->d_rebuilt_storage != 0) cudaFree(ws->d_rebuilt_storage);
        ws->d_rebuilt_storage = 0;
        ws->d_rebuilt_block_cols = 0;
        ws->d_rebuilt_val = 0;
        if (total_bytes != 0u && !detail::repack_cuda_check(cudaMalloc(&ws->d_rebuilt_storage, total_bytes), "cudaMalloc repack rebuilt storage")) return 0;
        ws->rebuilt_storage_capacity_bytes = total_bytes;
    }
    if (total_bytes != 0u) {
        ws->d_rebuilt_block_cols = idx_bytes != 0u ? (types::idx_t *) ws->d_rebuilt_storage : 0;
        ws->d_rebuilt_val = (real::storage_t *) ((char *) ws->d_rebuilt_storage + val_offset);
    } else {
        ws->d_rebuilt_block_cols = 0;
        ws->d_rebuilt_val = 0;
    }
    return 1;
}

__host__ __forceinline__ int build_repacked_shard_blocked_ell(
    const device::sharded_device<sparse::blocked_ell> *state,
    const sharded<sparse::blocked_ell> *view,
    unsigned long shard_id,
    const sharded_blocked_ell_repack_config *cfg,
    const unsigned char *d_keep_rows,
    const unsigned char *d_keep_cols,
    const types::idx_t *d_row_remap,
    const types::idx_t *d_col_remap,
    sharded_blocked_ell_repack_workspace *ws,
    sharded_blocked_ell_repack_result *out
) {
    sharded_blocked_ell_repack_result local{};
    unsigned long part_begin = 0;
    unsigned long part_end = 0;
    types::dim_t shard_rows = 0;
    types::nnz_t shard_nnz = 0;
    unsigned int live_nnz = 0u;
    unsigned long long kept_rows = 0ull;
    unsigned long long kept_row_slots = 0ull;
    types::u32 ell_width = 0u;
    dim3 grid(1u, 1u, 1u);
    dim3 block(128u, 1u, 1u);
    unsigned long local_part = 0;

    if (state == 0 || view == 0 || cfg == 0 || ws == 0 || out == 0) return 0;
    if (cfg->block_size == 0u) return 0;
    if (shard_id >= view->num_shards) return 0;

    part_begin = first_partition_in_shard(view, shard_id);
    part_end = last_partition_in_shard(view, shard_id);
    shard_rows = (types::dim_t) rows_in_shard(view, shard_id);
    shard_nnz = (types::nnz_t) nnz_in_shard(view, shard_id);

    if (!reserve_parts(ws, part_end - part_begin)) return 0;
    for (local_part = 0; part_begin + local_part < part_end; ++local_part) {
        const unsigned long part_id = part_begin + local_part;
        const device::partition_record<sparse::blocked_ell> *record = state->parts + part_id;
        if (part_id >= state->capacity || record->view == 0 || record->device_id != ws->device) return 0;
        ws->h_part_views[local_part] = (const device::blocked_ell_view *) record->view;
        ws->h_part_row_offsets[local_part + 1ul] = ws->h_part_row_offsets[local_part] + (types::idx_t) view->partition_rows[part_id];
    }
    if (!detail::repack_cuda_check(cudaMemcpyAsync(ws->d_part_views,
                                                   ws->h_part_views,
                                                   (std::size_t) (part_end - part_begin) * sizeof(const device::blocked_ell_view *),
                                                   cudaMemcpyHostToDevice,
                                                   ws->stream),
                                   "copy repack part views")) return 0;
    if (!detail::repack_cuda_check(cudaMemcpyAsync(ws->d_part_row_offsets,
                                                   ws->h_part_row_offsets,
                                                   (std::size_t) (part_end - part_begin + 1ul) * sizeof(types::idx_t),
                                                   cudaMemcpyHostToDevice,
                                                   ws->stream),
                                   "copy repack part row offsets")) return 0;

    if (!reserve_filtered(ws, cfg->output_rows, cfg->output_cols, shard_nnz)) return 0;
    if (!detail::repack_cuda_check(cudaMemsetAsync(ws->d_filtered_major_ptr,
                                                   0,
                                                   (std::size_t) (cfg->output_rows + 1u) * sizeof(types::ptr_t),
                                                   ws->stream),
                                   "memset repack filtered row ptr")) return 0;
    if (!detail::repack_cuda_check(cudaMemsetAsync(ws->d_stats_block, 0, 2u * sizeof(unsigned long long), ws->stream),
                                   "memset repack stats")) return 0;

    if (shard_rows != 0u) {
        grid.x = (shard_rows + block.x - 1u) / block.x;
        if (grid.x > 4096u) grid.x = 4096u;
    }

    kernels::count_filtered_shard_blocked_ell_row_nnz<<<grid, block, 0, ws->stream>>>(
        ws->d_part_views,
        ws->d_part_row_offsets,
        (types::idx_t) (part_end - part_begin),
        shard_rows,
        cfg->output_rows,
        cfg->output_cols,
        d_keep_rows,
        d_keep_cols,
        d_row_remap,
        d_col_remap,
        ws->d_filtered_major_ptr,
        ws->d_kept_rows,
        ws->d_kept_row_slots
    );
    if (!detail::repack_cuda_check(cudaGetLastError(), "count filtered shard blocked ell row nnz")) return 0;

    if (cub::DeviceScan::ExclusiveSum(ws->d_filtered_scan_tmp,
                                      ws->d_filtered_scan_tmp_bytes,
                                      ws->d_filtered_major_ptr,
                                      ws->d_filtered_major_ptr,
                                      (std::size_t) cfg->output_rows + 1u,
                                      ws->stream) != cudaSuccess) {
        std::fprintf(stderr, "CUB error at repack filtered row scan\n");
        return 0;
    }

    if (!detail::repack_cuda_check(cudaMemcpyAsync(&live_nnz,
                                                   ws->d_filtered_major_ptr + cfg->output_rows,
                                                   sizeof(unsigned int),
                                                   cudaMemcpyDeviceToHost,
                                                   ws->stream),
                                   "copy repack live nnz")) return 0;
    if (!detail::repack_cuda_check(cudaMemcpyAsync(&kept_rows,
                                                   ws->d_kept_rows,
                                                   sizeof(unsigned long long),
                                                   cudaMemcpyDeviceToHost,
                                                   ws->stream),
                                   "copy repack kept rows")) return 0;
    if (!detail::repack_cuda_check(cudaMemcpyAsync(&kept_row_slots,
                                                   ws->d_kept_row_slots,
                                                   sizeof(unsigned long long),
                                                   cudaMemcpyDeviceToHost,
                                                   ws->stream),
                                   "copy repack kept slots")) return 0;
    if (!detail::repack_cuda_check(cudaStreamSynchronize(ws->stream), "sync repack filtered scan")) return 0;

    if (live_nnz > shard_nnz) return 0;

    local.filtered.rows = cfg->output_rows;
    local.filtered.cols = cfg->output_cols;
    local.filtered.nnz = live_nnz;
    local.filtered.axis = sparse::compressed_by_row;
    local.filtered.majorPtr = ws->d_filtered_major_ptr;
    local.filtered.minorIdx = ws->d_filtered_minor_idx;
    local.filtered.val = ws->d_filtered_val;
    local.shard_id = shard_id;
    local.stats.kept_rows = (types::dim_t) kept_rows;
    local.stats.output_rows = cfg->output_rows;
    local.stats.output_cols = cfg->output_cols;
    local.stats.live_nnz = live_nnz;
    local.stats.kept_row_slots = (types::u64) kept_row_slots;
    local.stats.dead_slots = kept_row_slots > (unsigned long long) live_nnz ? (types::u64) (kept_row_slots - (unsigned long long) live_nnz) : 0u;
    local.stats.live_fill_ratio = kept_row_slots == 0ull ? 1.0 : (double) live_nnz / (double) kept_row_slots;
    local.stats.dead_value_bytes = (std::size_t) local.stats.dead_slots * sizeof(real::storage_t);
    local.stats.ell_width = 0u;

    if (cfg->output_rows == 0u || cfg->output_cols == 0u || live_nnz == 0u) {
        local.bucketed.rebuilt = local.filtered;
        local.bucketed.plan.major_dim = cfg->output_rows;
        local.bucketed.plan.bucket_count = 1u;
        local.bucketed.plan.major_order = 0;
        local.bucketed.plan.major_nnz_sorted = 0;
        local.bucketed.plan.bucket_offsets = 0;
        local.bucketed.plan.inverse_major_order = 0;
        local.bucket_plan = local.bucketed.plan;
        local.rebuilt.rows = cfg->output_rows;
        local.rebuilt.cols = cfg->output_cols;
        local.rebuilt.nnz = live_nnz;
        local.rebuilt.block_size = cfg->block_size;
        local.rebuilt.ell_cols = 0u;
        local.rebuilt.blockColIdx = 0;
        local.rebuilt.val = 0;
        if (out != 0) *out = local;
        return 1;
    }

    kernels::emit_filtered_shard_blocked_ell_compressed<<<grid, block, 0, ws->stream>>>(
        ws->d_part_views,
        ws->d_part_row_offsets,
        (types::idx_t) (part_end - part_begin),
        shard_rows,
        cfg->output_rows,
        cfg->output_cols,
        d_keep_rows,
        d_keep_cols,
        d_row_remap,
        d_col_remap,
        ws->d_filtered_major_ptr,
        ws->d_filtered_minor_idx,
        ws->d_filtered_val
    );
    if (!detail::repack_cuda_check(cudaGetLastError(), "emit filtered shard blocked ell")) return 0;
    if (!detail::repack_cuda_check(cudaStreamSynchronize(ws->stream), "sync repack filtered emit")) return 0;

    if (!bucket::build_bucketed_major_view(&local.filtered,
                                           cfg->requested_bucket_count,
                                           &ws->bucket_ws,
                                           &local.bucketed)) return 0;
    if (!detail::repack_cuda_check(cudaStreamSynchronize(ws->bucket_ws.stream), "sync repack bucket build")) return 0;
    local.bucket_plan = local.bucketed.plan;

    {
        const types::dim_t row_blocks = (cfg->output_rows + cfg->block_size - 1u) / cfg->block_size;
        if (!reserve_repack(ws, live_nnz, row_blocks)) return 0;

        if (!convert::build_coo_from_compressed_raw(local.bucketed.rebuilt.rows,
                                                    local.bucketed.rebuilt.nnz,
                                                    local.bucketed.rebuilt.majorPtr,
                                                    local.bucketed.rebuilt.minorIdx,
                                                    local.bucketed.rebuilt.val,
                                                    ws->d_row_idx,
                                                    const_cast<types::idx_t *>(local.bucketed.rebuilt.minorIdx),
                                                    const_cast<real::storage_t *>(local.bucketed.rebuilt.val),
                                                    ws->stream)) return 0;

        kernels::build_block_sort_keys<<<grid, block, 0, ws->stream>>>(
            live_nnz,
            ws->d_row_idx,
            local.bucketed.rebuilt.minorIdx,
            cfg->block_size,
            ws->d_sort_keys_in,
            ws->d_sort_pos_in
        );
        if (!detail::repack_cuda_check(cudaGetLastError(), "build repack block sort keys")) return 0;

        if (!detail::repack_cuda_check(cudaMemsetAsync(ws->d_row_block_counts,
                                                       0,
                                                       (std::size_t) row_blocks * sizeof(types::idx_t),
                                                       ws->stream),
                                       "memset repack row block counts")) return 0;

        if (!detail::repack_cuda_check(
                cub::DeviceRadixSort::SortPairs(ws->d_sort_tmp,
                                                ws->sort_capacity,
                                                ws->d_sort_keys_in,
                                                ws->d_sort_keys_out,
                                                ws->d_sort_pos_in,
                                                ws->d_sort_pos_out,
                                                live_nnz,
                                                0,
                                                sizeof(types::u64) * 8,
                                                ws->stream),
                "cub repack sort pairs")) return 0;

        kernels::mark_block_group_heads_and_count<<<grid, block, 0, ws->stream>>>(
            live_nnz,
            ws->d_sort_keys_out,
            row_blocks,
            ws->d_head_flags,
            ws->d_row_block_counts
        );
        if (!detail::repack_cuda_check(cudaGetLastError(), "mark repack block group heads")) return 0;

        if (!detail::repack_cuda_check(
                cub::DeviceScan::ExclusiveSum(ws->d_scan_tmp,
                                              ws->scan_capacity,
                                              ws->d_head_flags,
                                              ws->d_group_ids,
                                              live_nnz,
                                              ws->stream),
                "cub repack group scan")) return 0;

        if (!detail::repack_cuda_check(
                cub::DeviceScan::ExclusiveSum(ws->d_scan_tmp,
                                              ws->scan_capacity,
                                              ws->d_row_block_counts,
                                              ws->d_row_block_offsets,
                                              (std::size_t) row_blocks + 1u,
                                              ws->stream),
                "cub repack row-block scan")) return 0;

        if (!detail::repack_cuda_check(
                cub::DeviceReduce::Max(ws->d_reduce_tmp,
                                       ws->reduce_capacity,
                                       ws->d_row_block_counts,
                                       ws->d_ell_width,
                                       row_blocks,
                                       ws->stream),
                "cub repack ell width reduce")) return 0;

        if (!detail::repack_cuda_check(cudaMemcpyAsync(&ell_width,
                                                       ws->d_ell_width,
                                                       sizeof(types::u32),
                                                       cudaMemcpyDeviceToHost,
                                                       ws->stream),
                                       "copy repack ell width")) return 0;
        if (!detail::repack_cuda_check(cudaStreamSynchronize(ws->stream), "sync repack ell width")) return 0;

        if (!reserve_rebuilt_storage(ws, cfg->output_rows, ell_width, cfg->block_size)) return 0;
        if (ell_width != 0u) {
            const types::u32 ell_cols = ell_width * cfg->block_size;
            const types::idx_t block_col_count = row_blocks * ell_width;
            if (!detail::repack_cuda_check(cudaMemsetAsync(ws->d_rebuilt_val,
                                                           0,
                                                           (std::size_t) cfg->output_rows * (std::size_t) ell_cols * sizeof(real::storage_t),
                                                           ws->stream),
                                           "memset repack rebuilt values")) return 0;
            kernels::fill_blocked_ell_invalid_cols<<<grid, block, 0, ws->stream>>>(
                block_col_count,
                ws->d_rebuilt_block_cols
            );
            if (!detail::repack_cuda_check(cudaGetLastError(), "fill repack blocked ell invalid cols")) return 0;
            kernels::scatter_blocked_ell_block_cols<<<grid, block, 0, ws->stream>>>(
                live_nnz,
                ws->d_sort_keys_out,
                ws->d_head_flags,
                ws->d_group_ids,
                ws->d_row_block_offsets,
                ell_width,
                ws->d_rebuilt_block_cols
            );
            if (!detail::repack_cuda_check(cudaGetLastError(), "scatter repack blocked ell block cols")) return 0;
            kernels::scatter_blocked_ell_values<<<grid, block, 0, ws->stream>>>(
                live_nnz,
                ws->d_sort_keys_out,
                ws->d_sort_pos_out,
                ws->d_group_ids,
                ws->d_row_block_offsets,
                ws->d_row_idx,
                local.bucketed.rebuilt.minorIdx,
                local.bucketed.rebuilt.val,
                cfg->block_size,
                ell_width,
                ell_cols,
                ws->d_rebuilt_val
            );
            if (!detail::repack_cuda_check(cudaGetLastError(), "scatter repack blocked ell values")) return 0;
            if (!detail::repack_cuda_check(cudaStreamSynchronize(ws->stream), "sync repack blocked ell scatter")) return 0;
            local.rebuilt.rows = cfg->output_rows;
            local.rebuilt.cols = cfg->output_cols;
            local.rebuilt.nnz = live_nnz;
            local.rebuilt.block_size = cfg->block_size;
            local.rebuilt.ell_cols = ell_cols;
            local.rebuilt.blockColIdx = ws->d_rebuilt_block_cols;
            local.rebuilt.val = ws->d_rebuilt_val;
            local.stats.ell_width = ell_width;
        } else {
            local.rebuilt.rows = cfg->output_rows;
            local.rebuilt.cols = cfg->output_cols;
            local.rebuilt.nnz = live_nnz;
            local.rebuilt.block_size = cfg->block_size;
            local.rebuilt.ell_cols = 0u;
            local.rebuilt.blockColIdx = 0;
            local.rebuilt.val = 0;
            local.stats.ell_width = 0u;
        }
    }

    *out = local;
    return 1;
}

} // namespace repack
} // namespace cellshard
