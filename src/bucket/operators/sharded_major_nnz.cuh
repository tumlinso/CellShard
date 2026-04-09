#pragma once

#include "major_nnz.cuh"

namespace cellshard {
namespace bucket {

namespace kernels {

// Packed compressed shard uploads place one dense compressed_view array on the
// owner GPU. Build the shard-operator pointer tables directly from that array
// so repeated bucket rebuilds do not bounce three pointer lists through host
// memory on every pass.
__global__ static void gather_compressed_part_pointers(
    const device::compressed_view * __restrict__ part_views,
    types::idx_t part_count,
    types::ptr_t ** __restrict__ out_major_ptr,
    types::idx_t ** __restrict__ out_minor_idx,
    real::storage_t ** __restrict__ out_val
) {
    const types::idx_t tid = (types::idx_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::idx_t stride = (types::idx_t) (gridDim.x * blockDim.x);
    types::idx_t i = tid;

    while (i < part_count) {
        const device::compressed_view view = part_views[i];
        out_major_ptr[i] = view.majorPtr;
        out_minor_idx[i] = view.minorIdx;
        out_val[i] = view.val;
        i += stride;
    }
}

} // namespace kernels

struct alignas(16) sharded_major_bucket_workspace {
    compressed_major_bucket_workspace base;

    unsigned long part_capacity;
    // The shard-specific pointer tables are copied to device on every rebuild,
    // so keep the host side pinned and the device side densely packed.
    void *h_part_block;
    void *d_part_block;

    types::ptr_t **h_part_major_ptr;
    types::idx_t **h_part_minor_idx;
    real::storage_t **h_part_val;
    types::idx_t *h_part_row_offsets;

    types::ptr_t **d_part_major_ptr;
    types::idx_t **d_part_minor_idx;
    real::storage_t **d_part_val;
    types::idx_t *d_part_row_offsets;
};

struct alignas(16) sharded_major_bucket_result {
    device::compressed_view rebuilt;
    major_nnz_bucket_plan_view plan;
    unsigned long shard_id;
};

__host__ __forceinline__ void init(sharded_major_bucket_workspace *ws) {
    init(&ws->base);
    ws->part_capacity = 0;
    ws->h_part_block = 0;
    ws->d_part_block = 0;
    ws->h_part_major_ptr = 0;
    ws->h_part_minor_idx = 0;
    ws->h_part_val = 0;
    ws->h_part_row_offsets = 0;
    ws->d_part_major_ptr = 0;
    ws->d_part_minor_idx = 0;
    ws->d_part_val = 0;
    ws->d_part_row_offsets = 0;
}

__host__ __forceinline__ void clear(sharded_major_bucket_workspace *ws) {
    clear(&ws->base);
    if (ws->h_part_block != 0) cudaFreeHost(ws->h_part_block);
    if (ws->base.device >= 0) cudaSetDevice(ws->base.device);
    if (ws->d_part_block != 0) cudaFree(ws->d_part_block);
    init(ws);
}

__host__ __forceinline__ int setup(sharded_major_bucket_workspace *ws, int device) {
    init(ws);
    return setup(&ws->base, device);
}

__host__ __forceinline__ int reserve_parts(sharded_major_bucket_workspace *ws, unsigned long part_count) {
    if (!bucket_cuda_check(cudaSetDevice(ws->base.device >= 0 ? ws->base.device : 0), "cudaSetDevice shard bucket reserve parts")) return 0;
    if (part_count <= ws->part_capacity) return 1;

    if (ws->h_part_block != 0) cudaFreeHost(ws->h_part_block);
    if (ws->d_part_block != 0) cudaFree(ws->d_part_block);
    ws->h_part_block = 0;
    ws->d_part_block = 0;
    ws->h_part_major_ptr = 0;
    ws->h_part_minor_idx = 0;
    ws->h_part_val = 0;
    ws->h_part_row_offsets = 0;
    ws->d_part_major_ptr = 0;
    ws->d_part_minor_idx = 0;
    ws->d_part_val = 0;
    ws->d_part_row_offsets = 0;

    if (part_count != 0) {
        const std::size_t h_major_bytes = (std::size_t) part_count * sizeof(types::ptr_t *);
        const std::size_t h_minor_offset = ::cellshard::device::align_up_bytes(h_major_bytes, alignof(types::idx_t *));
        const std::size_t h_val_offset = ::cellshard::device::align_up_bytes(h_minor_offset + (std::size_t) part_count * sizeof(types::idx_t *), alignof(real::storage_t *));
        const std::size_t h_rows_offset = ::cellshard::device::align_up_bytes(h_val_offset + (std::size_t) part_count * sizeof(real::storage_t *), alignof(types::idx_t));
        const std::size_t h_total_bytes = h_rows_offset + (std::size_t) (part_count + 1ul) * sizeof(types::idx_t);
        const std::size_t d_major_bytes = (std::size_t) part_count * sizeof(types::ptr_t *);
        const std::size_t d_minor_offset = ::cellshard::device::align_up_bytes(d_major_bytes, alignof(types::idx_t *));
        const std::size_t d_val_offset = ::cellshard::device::align_up_bytes(d_minor_offset + (std::size_t) part_count * sizeof(types::idx_t *), alignof(real::storage_t *));
        const std::size_t d_rows_offset = ::cellshard::device::align_up_bytes(d_val_offset + (std::size_t) part_count * sizeof(real::storage_t *), alignof(types::idx_t));
        const std::size_t d_total_bytes = d_rows_offset + (std::size_t) (part_count + 1ul) * sizeof(types::idx_t);

        if (!bucket_cuda_check(cudaMallocHost(&ws->h_part_block, h_total_bytes), "cudaMallocHost h_part_block")) return 0;
        std::memset(ws->h_part_block, 0, h_total_bytes);
        ws->h_part_major_ptr = (types::ptr_t **) ws->h_part_block;
        ws->h_part_minor_idx = (types::idx_t **) ((char *) ws->h_part_block + h_minor_offset);
        ws->h_part_val = (real::storage_t **) ((char *) ws->h_part_block + h_val_offset);
        ws->h_part_row_offsets = (types::idx_t *) ((char *) ws->h_part_block + h_rows_offset);

        if (!bucket_cuda_check(cudaMalloc(&ws->d_part_block, d_total_bytes), "cudaMalloc d_part_block")) return 0;
        ws->d_part_major_ptr = (types::ptr_t **) ws->d_part_block;
        ws->d_part_minor_idx = (types::idx_t **) ((char *) ws->d_part_block + d_minor_offset);
        ws->d_part_val = (real::storage_t **) ((char *) ws->d_part_block + d_val_offset);
        ws->d_part_row_offsets = (types::idx_t *) ((char *) ws->d_part_block + d_rows_offset);
    }

    ws->part_capacity = part_count;
    return 1;
}

__host__ __forceinline__ int build_bucketed_shard_major_view(
    const device::sharded_device<sparse::compressed> *state,
    const sharded<sparse::compressed> *view,
    unsigned long shard_id,
    types::idx_t requested_bucket_count,
    sharded_major_bucket_workspace *ws,
    sharded_major_bucket_result *out
) {
    unsigned long part_begin = 0;
    unsigned long part_end = 0;
    unsigned long local_part = 0;
    const device::compressed_view *packed_views = 0;
    types::dim_t shard_rows = 0;
    types::nnz_t shard_nnz = 0;
    types::idx_t bucket_count = 0;
    int can_gather_direct = 1;

    if (state == 0 || view == 0 || ws == 0 || out == 0) return 0;
    if (shard_id >= view->num_shards) return 0;

    part_begin = first_part_in_shard(view, shard_id);
    part_end = last_part_in_shard(view, shard_id);
    shard_rows = (types::dim_t) rows_in_shard(view, shard_id);
    shard_nnz = (types::nnz_t) nnz_in_shard(view, shard_id);
    bucket_count = clamp_bucket_count(shard_rows, requested_bucket_count);

    if (!reserve_parts(ws, part_end - part_begin)) return 0;
    if (!reserve(&ws->base, shard_rows, shard_nnz, bucket_count)) return 0;

    ws->h_part_row_offsets[0] = 0u;
    if (part_begin < state->capacity && state->parts[part_begin].view != 0) {
        packed_views = (const device::compressed_view *) state->parts[part_begin].view;
    }
    for (local_part = 0; part_begin + local_part < part_end; ++local_part) {
        const unsigned long part_id = part_begin + local_part;
        const device::part_record<sparse::compressed> *record = state->parts + part_id;
        if (part_id >= state->capacity || record->a0 == 0 || record->a1 == 0 || record->a2 == 0) return 0;
        if (view->part_aux[part_id] != sparse::compressed_by_row) return 0;
        if (packed_views == 0 ||
            record->group_begin != part_begin ||
            record->group_end != part_end ||
            record->device_id != ws->base.device ||
            record->view != (void *) (packed_views + local_part)) {
            can_gather_direct = 0;
        }
        ws->h_part_major_ptr[local_part] = (types::ptr_t *) record->a0;
        ws->h_part_minor_idx[local_part] = (types::idx_t *) record->a1;
        ws->h_part_val[local_part] = (real::storage_t *) record->a2;
        ws->h_part_row_offsets[local_part + 1ul] = ws->h_part_row_offsets[local_part] + (types::idx_t) view->part_rows[part_id];
    }

    if (can_gather_direct) {
        unsigned int blocks = (unsigned int) (((types::idx_t) (part_end - part_begin) + 255u) >> 8);
        if (blocks == 0u) blocks = 1u;
        if (blocks > 1024u) blocks = 1024u;
        kernels::gather_compressed_part_pointers<<<blocks, 256, 0, ws->base.stream>>>(
            packed_views,
            (types::idx_t) (part_end - part_begin),
            ws->d_part_major_ptr,
            ws->d_part_minor_idx,
            ws->d_part_val
        );
        if (!bucket_cuda_check(cudaGetLastError(), "gather packed shard part pointers")) return 0;
    } else {
        if (!bucket_cuda_check(cudaMemcpyAsync(ws->d_part_major_ptr,
                                               ws->h_part_major_ptr,
                                               (std::size_t) (part_end - part_begin) * sizeof(types::ptr_t *),
                                               cudaMemcpyHostToDevice,
                                               ws->base.stream),
                               "copy shard part major ptrs")) return 0;
        if (!bucket_cuda_check(cudaMemcpyAsync(ws->d_part_minor_idx,
                                               ws->h_part_minor_idx,
                                               (std::size_t) (part_end - part_begin) * sizeof(types::idx_t *),
                                               cudaMemcpyHostToDevice,
                                               ws->base.stream),
                               "copy shard part minor ptrs")) return 0;
        if (!bucket_cuda_check(cudaMemcpyAsync(ws->d_part_val,
                                               ws->h_part_val,
                                               (std::size_t) (part_end - part_begin) * sizeof(real::storage_t *),
                                               cudaMemcpyHostToDevice,
                                               ws->base.stream),
                               "copy shard part value ptrs")) return 0;
    }
    if (!bucket_cuda_check(cudaMemcpyAsync(ws->d_part_row_offsets,
                                           ws->h_part_row_offsets,
                                           (std::size_t) (part_end - part_begin + 1ul) * sizeof(types::idx_t),
                                           cudaMemcpyHostToDevice,
                                           ws->base.stream),
                           "copy shard part row offsets")) return 0;

    if (!build_shard_major_nnz_bucket_plan_raw(ws->d_part_major_ptr,
                                               ws->d_part_row_offsets,
                                               (types::idx_t) (part_end - part_begin),
                                               shard_rows,
                                               ws->base.d_major_nnz,
                                               ws->base.d_major_nnz_sorted,
                                               ws->base.d_major_order_in,
                                               ws->base.d_major_order_out,
                                               ws->base.d_bucket_offsets,
                                               bucket_count,
                                               ws->base.d_sort_tmp,
                                               ws->base.sort_capacity,
                                               ws->base.stream)) return 0;

    if (!rebuild_bucketed_shard_compressed_raw(ws->d_part_major_ptr,
                                               ws->d_part_minor_idx,
                                               ws->d_part_val,
                                               ws->d_part_row_offsets,
                                               (types::idx_t) (part_end - part_begin),
                                               shard_rows,
                                               ws->base.d_major_order_out,
                                               ws->base.d_rebuilt_major_ptr,
                                               ws->base.d_rebuilt_minor_idx,
                                               ws->base.d_rebuilt_val,
                                               ws->base.d_inverse_major_order,
                                               ws->base.d_scan_tmp,
                                               ws->base.scan_capacity,
                                               ws->base.stream)) return 0;

    out->rebuilt.rows = shard_rows;
    out->rebuilt.cols = (types::dim_t) view->cols;
    out->rebuilt.nnz = shard_nnz;
    out->rebuilt.axis = sparse::compressed_by_row;
    out->rebuilt.majorPtr = ws->base.d_rebuilt_major_ptr;
    out->rebuilt.minorIdx = ws->base.d_rebuilt_minor_idx;
    out->rebuilt.val = ws->base.d_rebuilt_val;
    out->plan.major_dim = shard_rows;
    out->plan.bucket_count = bucket_count;
    out->plan.major_order = ws->base.d_major_order_out;
    out->plan.major_nnz_sorted = ws->base.d_major_nnz_sorted;
    out->plan.bucket_offsets = ws->base.d_bucket_offsets;
    out->plan.inverse_major_order = ws->base.d_inverse_major_order;
    out->shard_id = shard_id;
    return 1;
}

} // namespace bucket
} // namespace cellshard
