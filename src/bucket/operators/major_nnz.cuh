#pragma once

#include "../major_nnz_raw.cuh"
#include "../../sharded/sharded_device.cuh"

#include <cstring>

namespace cellshard {
namespace bucket {

struct alignas(16) compressed_major_bucket_workspace {
    int device;
    cudaStream_t stream;

    types::dim_t major_capacity;
    types::nnz_t nnz_capacity;
    types::idx_t bucket_capacity;
    std::size_t sort_capacity;
    std::size_t scan_capacity;
    // Packed allocation blocks reduce allocator churn when the caller rebuilds
    // bucket plans for many similarly sized matrices or shards.
    void *d_major_block;
    void *d_rebuilt_block;

    types::idx_t *d_major_nnz;
    types::idx_t *d_major_nnz_sorted;
    types::idx_t *d_major_order_in;
    types::idx_t *d_major_order_out;
    types::idx_t *d_inverse_major_order;
    types::idx_t *d_bucket_offsets;

    types::ptr_t *d_rebuilt_major_ptr;
    types::idx_t *d_rebuilt_minor_idx;
    real::storage_t *d_rebuilt_val;

    void *d_sort_tmp;
    void *d_scan_tmp;
};

struct alignas(16) compressed_major_bucket_result {
    device::compressed_view rebuilt;
    major_nnz_bucket_plan_view plan;
};

__host__ __forceinline__ void init(compressed_major_bucket_workspace *ws) {
    std::memset(ws, 0, sizeof(*ws));
    ws->device = -1;
}

__host__ __forceinline__ void clear(compressed_major_bucket_workspace *ws) {
    if (ws->device >= 0) cudaSetDevice(ws->device);
    if (ws->d_scan_tmp != 0) cudaFree(ws->d_scan_tmp);
    if (ws->d_sort_tmp != 0) cudaFree(ws->d_sort_tmp);
    if (ws->d_rebuilt_block != 0) cudaFree(ws->d_rebuilt_block);
    if (ws->d_bucket_offsets != 0) cudaFree(ws->d_bucket_offsets);
    if (ws->d_major_block != 0) cudaFree(ws->d_major_block);
    if (ws->stream != 0) cudaStreamDestroy(ws->stream);
    init(ws);
}

__host__ __forceinline__ int setup(compressed_major_bucket_workspace *ws, int device) {
    init(ws);
    ws->device = device;
    if (!bucket_cuda_check(cudaSetDevice(device), "cudaSetDevice bucket workspace")) return 0;
    return bucket_cuda_check(cudaStreamCreateWithFlags(&ws->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags bucket");
}

__host__ __forceinline__ int reserve(compressed_major_bucket_workspace *ws,
                                     types::dim_t major_dim,
                                     types::nnz_t nnz,
                                     types::idx_t bucket_count) {
    std::size_t sort_bytes = 0;
    std::size_t scan_bytes = 0;

    if (!bucket_cuda_check(cudaSetDevice(ws->device >= 0 ? ws->device : 0), "cudaSetDevice bucket reserve")) return 0;
    if (!major_nnz_bucket_sort_scratch_bytes(major_dim, &sort_bytes)) return 0;
    if (!major_nnz_bucket_scan_scratch_bytes(major_dim, &scan_bytes)) return 0;

    if (major_dim > ws->major_capacity) {
        if (ws->d_major_block != 0) cudaFree(ws->d_major_block);
        ws->d_major_block = 0;
        ws->d_major_nnz = 0;
        ws->d_major_nnz_sorted = 0;
        ws->d_major_order_in = 0;
        ws->d_major_order_out = 0;
        ws->d_inverse_major_order = 0;
        ws->d_rebuilt_major_ptr = 0;

        {
            const std::size_t nnz_sorted_offset = ::cellshard::device::align_up_bytes((std::size_t) major_dim * sizeof(types::idx_t), alignof(types::idx_t));
            const std::size_t order_in_offset = ::cellshard::device::align_up_bytes(nnz_sorted_offset + (std::size_t) major_dim * sizeof(types::idx_t), alignof(types::idx_t));
            const std::size_t order_out_offset = ::cellshard::device::align_up_bytes(order_in_offset + (std::size_t) major_dim * sizeof(types::idx_t), alignof(types::idx_t));
            const std::size_t inverse_offset = ::cellshard::device::align_up_bytes(order_out_offset + (std::size_t) major_dim * sizeof(types::idx_t), alignof(types::idx_t));
            const std::size_t rebuilt_ptr_offset = ::cellshard::device::align_up_bytes(inverse_offset + (std::size_t) major_dim * sizeof(types::idx_t), alignof(types::ptr_t));
            const std::size_t total_bytes = rebuilt_ptr_offset + (std::size_t) (major_dim + 1u) * sizeof(types::ptr_t);
            if (!bucket_cuda_check(cudaMalloc(&ws->d_major_block, total_bytes == 0 ? sizeof(types::ptr_t) : total_bytes), "cudaMalloc d_major_block")) return 0;
            ws->d_major_nnz = (types::idx_t *) ws->d_major_block;
            ws->d_major_nnz_sorted = (types::idx_t *) ((char *) ws->d_major_block + nnz_sorted_offset);
            ws->d_major_order_in = (types::idx_t *) ((char *) ws->d_major_block + order_in_offset);
            ws->d_major_order_out = (types::idx_t *) ((char *) ws->d_major_block + order_out_offset);
            ws->d_inverse_major_order = (types::idx_t *) ((char *) ws->d_major_block + inverse_offset);
            ws->d_rebuilt_major_ptr = (types::ptr_t *) ((char *) ws->d_major_block + rebuilt_ptr_offset);
        }
        ws->major_capacity = major_dim;
    }

    if (nnz > ws->nnz_capacity) {
        if (ws->d_rebuilt_block != 0) cudaFree(ws->d_rebuilt_block);
        ws->d_rebuilt_block = 0;
        ws->d_rebuilt_minor_idx = 0;
        ws->d_rebuilt_val = 0;
        if (nnz != 0u) {
            const std::size_t val_offset = ::cellshard::device::align_up_bytes((std::size_t) nnz * sizeof(types::idx_t), alignof(real::storage_t));
            const std::size_t total_bytes = val_offset + (std::size_t) nnz * sizeof(real::storage_t);
            if (!bucket_cuda_check(cudaMalloc(&ws->d_rebuilt_block, total_bytes), "cudaMalloc d_rebuilt_block")) return 0;
            ws->d_rebuilt_minor_idx = (types::idx_t *) ws->d_rebuilt_block;
            ws->d_rebuilt_val = (real::storage_t *) ((char *) ws->d_rebuilt_block + val_offset);
        }
        ws->nnz_capacity = nnz;
    }

    if (bucket_count > ws->bucket_capacity) {
        if (ws->d_bucket_offsets != 0) cudaFree(ws->d_bucket_offsets);
        ws->d_bucket_offsets = 0;
        if (!bucket_cuda_check(cudaMalloc((void **) &ws->d_bucket_offsets, (std::size_t) (bucket_count + 1u) * sizeof(types::idx_t)), "cudaMalloc d_bucket_offsets")) return 0;
        ws->bucket_capacity = bucket_count;
    }

    if (sort_bytes > ws->sort_capacity) {
        if (ws->d_sort_tmp != 0) cudaFree(ws->d_sort_tmp);
        ws->d_sort_tmp = 0;
        if (sort_bytes != 0u && !bucket_cuda_check(cudaMalloc(&ws->d_sort_tmp, sort_bytes), "cudaMalloc d_sort_tmp")) return 0;
        ws->sort_capacity = sort_bytes;
    }

    if (scan_bytes > ws->scan_capacity) {
        if (ws->d_scan_tmp != 0) cudaFree(ws->d_scan_tmp);
        ws->d_scan_tmp = 0;
        if (scan_bytes != 0u && !bucket_cuda_check(cudaMalloc(&ws->d_scan_tmp, scan_bytes), "cudaMalloc d_scan_tmp")) return 0;
        ws->scan_capacity = scan_bytes;
    }

    return 1;
}

__host__ __forceinline__ int build_plan(const device::compressed_view *src,
                                        types::idx_t requested_bucket_count,
                                        compressed_major_bucket_workspace *ws,
                                        major_nnz_bucket_plan_view *out) {
    types::dim_t major_dim = 0;
    types::idx_t bucket_count = 0;

    if (src == 0 || ws == 0 || out == 0) return 0;
    major_dim = src->axis == sparse::compressed_by_col ? src->cols : src->rows;
    bucket_count = clamp_bucket_count(major_dim, requested_bucket_count);
    if (!reserve(ws, major_dim, src->nnz, bucket_count)) return 0;
    if (!build_major_nnz_bucket_plan_raw(src->majorPtr,
                                         major_dim,
                                         ws->d_major_nnz,
                                         ws->d_major_nnz_sorted,
                                         ws->d_major_order_in,
                                         ws->d_major_order_out,
                                         ws->d_bucket_offsets,
                                         bucket_count,
                                         ws->d_sort_tmp,
                                         ws->sort_capacity,
                                         ws->stream)) return 0;
    out->major_dim = major_dim;
    out->bucket_count = bucket_count;
    out->major_order = ws->d_major_order_out;
    out->major_nnz_sorted = ws->d_major_nnz_sorted;
    out->bucket_offsets = ws->d_bucket_offsets;
    out->inverse_major_order = 0;
    return 1;
}

__host__ __forceinline__ int rebuild(const device::compressed_view *src,
                                     const major_nnz_bucket_plan_view *plan,
                                     compressed_major_bucket_workspace *ws,
                                     compressed_major_bucket_result *out) {
    if (src == 0 || plan == 0 || ws == 0 || out == 0) return 0;
    if (!rebuild_compressed_major_order_raw(src->majorPtr,
                                            src->minorIdx,
                                            src->val,
                                            plan->major_dim,
                                            plan->major_order,
                                            ws->d_rebuilt_major_ptr,
                                            ws->d_rebuilt_minor_idx,
                                            ws->d_rebuilt_val,
                                            ws->d_inverse_major_order,
                                            ws->d_scan_tmp,
                                            ws->scan_capacity,
                                            ws->stream)) return 0;
    out->rebuilt.rows = src->rows;
    out->rebuilt.cols = src->cols;
    out->rebuilt.nnz = src->nnz;
    out->rebuilt.axis = src->axis;
    out->rebuilt.majorPtr = ws->d_rebuilt_major_ptr;
    out->rebuilt.minorIdx = ws->d_rebuilt_minor_idx;
    out->rebuilt.val = ws->d_rebuilt_val;
    out->plan = *plan;
    out->plan.inverse_major_order = ws->d_inverse_major_order;
    return 1;
}

__host__ __forceinline__ int build_bucketed_major_view(const device::compressed_view *src,
                                                       types::idx_t requested_bucket_count,
                                                       compressed_major_bucket_workspace *ws,
                                                       compressed_major_bucket_result *out) {
    major_nnz_bucket_plan_view plan;
    if (!build_plan(src, requested_bucket_count, ws, &plan)) return 0;
    return rebuild(src, &plan, ws, out);
}

} // namespace bucket
} // namespace cellshard
