#pragma once

#include "kernels/csBucket.cuh"

#include <cub/cub.cuh>

#include <cstdio>

namespace cellshard {
namespace bucket {

struct alignas(16) major_nnz_bucket_plan_view {
    types::dim_t major_dim;
    types::idx_t bucket_count;
    const types::idx_t *major_order;
    const types::idx_t *major_nnz_sorted;
    const types::idx_t *bucket_offsets;
    const types::idx_t *inverse_major_order;
};

static inline int bucket_cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

static inline types::idx_t clamp_bucket_count(types::dim_t major_dim, types::idx_t requested) {
    if (major_dim == 0u) return 1u;
    if (requested < 1u) return 1u;
    if (requested > major_dim) return (types::idx_t) major_dim;
    return requested;
}

static inline int major_nnz_bucket_sort_scratch_bytes(types::dim_t major_dim, std::size_t *out_bytes) {
    std::size_t bytes = 0;
    if (out_bytes == 0) return 0;
    if (cub::DeviceRadixSort::SortPairs(
            0,
            bytes,
            (const types::idx_t *) 0,
            (types::idx_t *) 0,
            (const types::idx_t *) 0,
            (types::idx_t *) 0,
            major_dim,
            0,
            sizeof(types::idx_t) * 8) != cudaSuccess) return 0;
    *out_bytes = bytes;
    return 1;
}

static inline int major_nnz_bucket_scan_scratch_bytes(types::dim_t major_dim, std::size_t *out_bytes) {
    std::size_t bytes = 0;
    if (out_bytes == 0) return 0;
    if (cub::DeviceScan::ExclusiveSum(
            0,
            bytes,
            (const types::ptr_t *) 0,
            (types::ptr_t *) 0,
            major_dim + 1u) != cudaSuccess) return 0;
    *out_bytes = bytes;
    return 1;
}

static inline int build_major_nnz_bucket_plan_raw(
    const types::ptr_t *d_major_ptr,
    types::dim_t major_dim,
    types::idx_t *d_major_nnz,
    types::idx_t *d_major_nnz_sorted,
    types::idx_t *d_major_order_in,
    types::idx_t *d_major_order_out,
    types::idx_t *d_bucket_offsets,
    types::idx_t requested_bucket_count,
    void *d_sort_tmp,
    std::size_t sort_tmp_bytes,
    cudaStream_t stream
) {
    dim3 grid;
    dim3 block;

    if (d_major_ptr == 0 || d_major_nnz == 0 || d_major_nnz_sorted == 0 || d_major_order_in == 0 || d_major_order_out == 0) return 0;
    if (major_dim == 0u) {
        if (d_bucket_offsets != 0) {
            if (!bucket_cuda_check(cudaMemsetAsync(d_bucket_offsets, 0, sizeof(types::idx_t), stream), "cudaMemsetAsync empty bucket offsets")) return 0;
        }
        return 1;
    }

    kernels::setup_cs_bucket_major_launch(major_dim, &grid, &block);

    kernels::count_major_nnz<<<grid, block, 0, stream>>>(major_dim, d_major_ptr, d_major_nnz);
    if (cudaGetLastError() != cudaSuccess) return 0;

    kernels::init_major_identity<<<grid, block, 0, stream>>>(major_dim, d_major_order_in);
    if (cudaGetLastError() != cudaSuccess) return 0;

    if (!bucket_cuda_check(
            cub::DeviceRadixSort::SortPairs(
                d_sort_tmp,
                sort_tmp_bytes,
                d_major_nnz,
                d_major_nnz_sorted,
                d_major_order_in,
                d_major_order_out,
                major_dim,
                0,
                sizeof(types::idx_t) * 8,
                stream),
            "cub bucket sort pairs")) return 0;

    if (d_bucket_offsets != 0) {
        kernels::fill_equal_count_bucket_offsets<<<1, 1, 0, stream>>>(
            major_dim,
            clamp_bucket_count(major_dim, requested_bucket_count),
            d_bucket_offsets
        );
        if (cudaGetLastError() != cudaSuccess) return 0;
    }

    return 1;
}

static inline int build_shard_major_nnz_bucket_plan_raw(
    const types::ptr_t * const *d_part_major_ptr,
    const types::idx_t *d_part_row_offsets,
    types::idx_t part_count,
    types::dim_t shard_rows,
    types::idx_t *d_major_nnz,
    types::idx_t *d_major_nnz_sorted,
    types::idx_t *d_major_order_in,
    types::idx_t *d_major_order_out,
    types::idx_t *d_bucket_offsets,
    types::idx_t requested_bucket_count,
    void *d_sort_tmp,
    std::size_t sort_tmp_bytes,
    cudaStream_t stream
) {
    dim3 grid;
    dim3 block;

    if (d_part_major_ptr == 0 || d_part_row_offsets == 0 || d_major_nnz == 0 || d_major_nnz_sorted == 0 || d_major_order_in == 0 || d_major_order_out == 0) return 0;
    if (shard_rows == 0u) {
        if (d_bucket_offsets != 0) {
            if (!bucket_cuda_check(cudaMemsetAsync(d_bucket_offsets, 0, sizeof(types::idx_t), stream), "cudaMemsetAsync empty shard bucket offsets")) return 0;
        }
        return 1;
    }

    kernels::setup_cs_bucket_major_launch(shard_rows, &grid, &block);

    kernels::count_shard_major_nnz<<<grid, block, 0, stream>>>(
        shard_rows,
        d_part_row_offsets,
        part_count,
        d_part_major_ptr,
        d_major_nnz
    );
    if (cudaGetLastError() != cudaSuccess) return 0;

    kernels::init_major_identity<<<grid, block, 0, stream>>>(shard_rows, d_major_order_in);
    if (cudaGetLastError() != cudaSuccess) return 0;

    if (!bucket_cuda_check(
            cub::DeviceRadixSort::SortPairs(
                d_sort_tmp,
                sort_tmp_bytes,
                d_major_nnz,
                d_major_nnz_sorted,
                d_major_order_in,
                d_major_order_out,
                shard_rows,
                0,
                sizeof(types::idx_t) * 8,
                stream),
            "cub shard bucket sort pairs")) return 0;

    if (d_bucket_offsets != 0) {
        kernels::fill_equal_count_bucket_offsets<<<1, 1, 0, stream>>>(
            shard_rows,
            clamp_bucket_count(shard_rows, requested_bucket_count),
            d_bucket_offsets
        );
        if (cudaGetLastError() != cudaSuccess) return 0;
    }

    return 1;
}

static inline int rebuild_compressed_major_order_raw(
    const types::ptr_t *d_src_major_ptr,
    const types::idx_t *d_src_minor_idx,
    const real::storage_t *d_src_val,
    types::dim_t major_dim,
    const types::idx_t *d_major_order,
    types::ptr_t *d_dst_major_ptr,
    types::idx_t *d_dst_minor_idx,
    real::storage_t *d_dst_val,
    types::idx_t *d_inverse_major_order,
    void *d_scan_tmp,
    std::size_t scan_tmp_bytes,
    cudaStream_t stream
) {
    dim3 grid;
    dim3 block;

    if (d_src_major_ptr == 0 || d_src_minor_idx == 0 || d_src_val == 0 || d_major_order == 0 || d_dst_major_ptr == 0 || d_dst_minor_idx == 0 || d_dst_val == 0) return 0;
    if (major_dim == 0u) return 1;

    kernels::setup_cs_bucket_major_launch(major_dim, &grid, &block);

    kernels::gather_shifted_major_counts_from_order<<<grid, block, 0, stream>>>(
        major_dim,
        d_major_order,
        d_src_major_ptr,
        d_dst_major_ptr
    );
    if (cudaGetLastError() != cudaSuccess) return 0;

    if (!bucket_cuda_check(
            cub::DeviceScan::ExclusiveSum(
                d_scan_tmp,
                scan_tmp_bytes,
                d_dst_major_ptr,
                d_dst_major_ptr,
                major_dim + 1u,
                stream),
            "cub bucket major scan")) return 0;

    kernels::reorder_compressed_major_segments<real::storage_t><<<grid, block, 0, stream>>>(
        major_dim,
        d_major_order,
        d_src_major_ptr,
        d_src_minor_idx,
        d_src_val,
        d_dst_major_ptr,
        d_dst_minor_idx,
        d_dst_val
    );
    if (cudaGetLastError() != cudaSuccess) return 0;

    if (d_inverse_major_order != 0) {
        kernels::scatter_inverse_major_order<<<grid, block, 0, stream>>>(
            major_dim,
            d_major_order,
            d_inverse_major_order
        );
        if (cudaGetLastError() != cudaSuccess) return 0;
    }

    return 1;
}

static inline int rebuild_bucketed_shard_compressed_raw(
    const types::ptr_t * const *d_part_major_ptr,
    const types::idx_t * const *d_part_minor_idx,
    const real::storage_t * const *d_part_val,
    const types::idx_t *d_part_row_offsets,
    types::idx_t part_count,
    types::dim_t shard_rows,
    const types::idx_t *d_major_order,
    types::ptr_t *d_dst_major_ptr,
    types::idx_t *d_dst_minor_idx,
    real::storage_t *d_dst_val,
    types::idx_t *d_inverse_major_order,
    void *d_scan_tmp,
    std::size_t scan_tmp_bytes,
    cudaStream_t stream
) {
    dim3 grid;
    dim3 block;

    if (d_part_major_ptr == 0 || d_part_minor_idx == 0 || d_part_val == 0 || d_part_row_offsets == 0 || d_major_order == 0 || d_dst_major_ptr == 0 || d_dst_minor_idx == 0 || d_dst_val == 0) return 0;
    if (shard_rows == 0u) return 1;

    kernels::setup_cs_bucket_major_launch(shard_rows, &grid, &block);

    kernels::gather_shifted_shard_major_counts_from_order<<<grid, block, 0, stream>>>(
        shard_rows,
        d_major_order,
        d_part_row_offsets,
        part_count,
        d_part_major_ptr,
        d_dst_major_ptr
    );
    if (cudaGetLastError() != cudaSuccess) return 0;

    if (!bucket_cuda_check(
            cub::DeviceScan::ExclusiveSum(
                d_scan_tmp,
                scan_tmp_bytes,
                d_dst_major_ptr,
                d_dst_major_ptr,
                shard_rows + 1u,
                stream),
            "cub shard bucket major scan")) return 0;

    kernels::reorder_shard_major_segments<real::storage_t><<<grid, block, 0, stream>>>(
        shard_rows,
        d_major_order,
        d_part_row_offsets,
        part_count,
        d_part_major_ptr,
        d_part_minor_idx,
        d_part_val,
        d_dst_major_ptr,
        d_dst_minor_idx,
        d_dst_val
    );
    if (cudaGetLastError() != cudaSuccess) return 0;

    if (d_inverse_major_order != 0) {
        kernels::scatter_inverse_major_order<<<grid, block, 0, stream>>>(
            shard_rows,
            d_major_order,
            d_inverse_major_order
        );
        if (cudaGetLastError() != cudaSuccess) return 0;
    }

    return 1;
}

} // namespace bucket
} // namespace cellshard
