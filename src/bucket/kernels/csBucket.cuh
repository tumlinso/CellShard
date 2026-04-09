#pragma once

#include "../../offset_span.cuh"
#include "../../types.cuh"

#include <cuda_runtime.h>

namespace cellshard {
namespace bucket {
namespace kernels {

static inline void setup_cs_bucket_major_launch(types::dim_t major_dim, dim3 *grid, dim3 *block) {
    block->x = 256;
    block->y = 1;
    block->z = 1;
    grid->x = (unsigned int) (((unsigned long) major_dim + 255ul) >> 8);
    grid->y = 1;
    grid->z = 1;
    if (grid->x < 1u) grid->x = 1u;
    if (grid->x > 4096u) grid->x = 4096u;
}

__global__ static void count_major_nnz(
    types::dim_t major_dim,
    const types::ptr_t * __restrict__ major_ptr,
    types::idx_t * __restrict__ major_nnz
) {
    const types::dim_t tid = (types::dim_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::dim_t stride = (types::dim_t) (gridDim.x * blockDim.x);
    types::dim_t i = tid;
    while (i < major_dim) {
        major_nnz[i] = (types::idx_t) (major_ptr[i + 1] - major_ptr[i]);
        i += stride;
    }
}

__global__ static void init_major_identity(
    types::dim_t major_dim,
    types::idx_t * __restrict__ major_index
) {
    const types::dim_t tid = (types::dim_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::dim_t stride = (types::dim_t) (gridDim.x * blockDim.x);
    types::dim_t i = tid;
    while (i < major_dim) {
        major_index[i] = (types::idx_t) i;
        i += stride;
    }
}

__global__ static void fill_equal_count_bucket_offsets(
    types::dim_t major_dim,
    types::idx_t bucket_count,
    types::idx_t * __restrict__ bucket_offsets
) {
    types::idx_t bucket = 0;
    types::idx_t bucket_size = 0;
    types::idx_t remainder = 0;
    types::idx_t offset = 0;

    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (bucket_offsets == 0) return;
    if (bucket_count < 1u) bucket_count = 1u;
    if (major_dim == 0u) {
        bucket_offsets[0] = 0u;
        return;
    }
    if (bucket_count > major_dim) bucket_count = major_dim;
    bucket_size = (types::idx_t) (major_dim / bucket_count);
    remainder = (types::idx_t) (major_dim % bucket_count);
    for (bucket = 0; bucket < bucket_count; ++bucket) {
        bucket_offsets[bucket] = offset;
        offset += bucket_size + (bucket < remainder ? 1u : 0u);
    }
    bucket_offsets[bucket_count] = (types::idx_t) major_dim;
}

__global__ static void gather_major_nnz_by_order(
    types::dim_t major_dim,
    const types::idx_t * __restrict__ major_nnz,
    const types::idx_t * __restrict__ major_order,
    types::idx_t * __restrict__ sorted_major_nnz
) {
    const types::dim_t tid = (types::dim_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::dim_t stride = (types::dim_t) (gridDim.x * blockDim.x);
    types::dim_t i = tid;
    while (i < major_dim) {
        sorted_major_nnz[i] = major_nnz[major_order[i]];
        i += stride;
    }
}

__global__ static void gather_shifted_major_counts_from_order(
    types::dim_t major_dim,
    const types::idx_t * __restrict__ major_order,
    const types::ptr_t * __restrict__ src_major_ptr,
    types::ptr_t * __restrict__ dst_major_counts_shifted
) {
    const types::dim_t tid = (types::dim_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::dim_t stride = (types::dim_t) (gridDim.x * blockDim.x);
    types::dim_t i = tid;

    if (tid == 0u) dst_major_counts_shifted[0] = 0u;
    while (i < major_dim) {
        const types::idx_t src_major = major_order[i];
        dst_major_counts_shifted[i + 1u] = (types::ptr_t) (src_major_ptr[src_major + 1u] - src_major_ptr[src_major]);
        i += stride;
    }
}

__global__ static void scatter_inverse_major_order(
    types::dim_t major_dim,
    const types::idx_t * __restrict__ major_order,
    types::idx_t * __restrict__ inverse_major_order
) {
    const types::dim_t tid = (types::dim_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::dim_t stride = (types::dim_t) (gridDim.x * blockDim.x);
    types::dim_t i = tid;
    while (i < major_dim) {
        inverse_major_order[major_order[i]] = (types::idx_t) i;
        i += stride;
    }
}

template<typename ValueT>
__global__ static void reorder_compressed_major_segments(
    types::dim_t major_dim,
    const types::idx_t * __restrict__ major_order,
    const types::ptr_t * __restrict__ src_major_ptr,
    const types::idx_t * __restrict__ src_minor_idx,
    const ValueT * __restrict__ src_val,
    const types::ptr_t * __restrict__ dst_major_ptr,
    types::idx_t * __restrict__ dst_minor_idx,
    ValueT * __restrict__ dst_val
) {
    const types::dim_t major_stride = (types::dim_t) gridDim.x;
    types::dim_t dst_major = (types::dim_t) blockIdx.x;

    while (dst_major < major_dim) {
        const types::idx_t src_major = major_order[dst_major];
        const types::ptr_t src_begin = src_major_ptr[src_major];
        const types::ptr_t src_end = src_major_ptr[src_major + 1u];
        const types::ptr_t dst_begin = dst_major_ptr[dst_major];
        const types::ptr_t len = src_end - src_begin;
        types::ptr_t j = (types::ptr_t) threadIdx.x;

        while (j < len) {
            dst_minor_idx[dst_begin + j] = src_minor_idx[src_begin + j];
            dst_val[dst_begin + j] = src_val[src_begin + j];
            j += (types::ptr_t) blockDim.x;
        }

        dst_major += major_stride;
    }
}

__global__ static void count_shard_major_nnz(
    types::dim_t shard_rows,
    const types::idx_t * __restrict__ part_row_offsets,
    types::idx_t part_count,
    const types::ptr_t * const * __restrict__ part_major_ptr,
    types::idx_t * __restrict__ major_nnz
) {
    const types::dim_t tid = (types::dim_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::dim_t stride = (types::dim_t) (gridDim.x * blockDim.x);
    types::dim_t row = tid;

    while (row < shard_rows) {
        const types::idx_t part = (types::idx_t) ::cellshard::find_offset_span(row, part_row_offsets, part_count);
        const types::idx_t local_row = (types::idx_t) (row - part_row_offsets[part]);
        const types::ptr_t *major_ptr = part_major_ptr[part];
        major_nnz[row] = (types::idx_t) (major_ptr[local_row + 1u] - major_ptr[local_row]);
        row += stride;
    }
}

__global__ static void gather_shifted_shard_major_counts_from_order(
    types::dim_t shard_rows,
    const types::idx_t * __restrict__ major_order,
    const types::idx_t * __restrict__ part_row_offsets,
    types::idx_t part_count,
    const types::ptr_t * const * __restrict__ part_major_ptr,
    types::ptr_t * __restrict__ dst_major_counts_shifted
) {
    const types::dim_t tid = (types::dim_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::dim_t stride = (types::dim_t) (gridDim.x * blockDim.x);
    types::dim_t row = tid;

    if (tid == 0u) dst_major_counts_shifted[0] = 0u;
    while (row < shard_rows) {
        const types::idx_t src_row = major_order[row];
        const types::idx_t part = (types::idx_t) ::cellshard::find_offset_span(src_row, part_row_offsets, part_count);
        const types::idx_t local_row = (types::idx_t) (src_row - part_row_offsets[part]);
        const types::ptr_t *major_ptr = part_major_ptr[part];
        dst_major_counts_shifted[row + 1u] = (types::ptr_t) (major_ptr[local_row + 1u] - major_ptr[local_row]);
        row += stride;
    }
}

template<typename ValueT>
__global__ static void reorder_shard_major_segments(
    types::dim_t shard_rows,
    const types::idx_t * __restrict__ major_order,
    const types::idx_t * __restrict__ part_row_offsets,
    types::idx_t part_count,
    const types::ptr_t * const * __restrict__ part_major_ptr,
    const types::idx_t * const * __restrict__ part_minor_idx,
    const ValueT * const * __restrict__ part_val,
    const types::ptr_t * __restrict__ dst_major_ptr,
    types::idx_t * __restrict__ dst_minor_idx,
    ValueT * __restrict__ dst_val
) {
    const types::dim_t major_stride = (types::dim_t) gridDim.x;
    types::dim_t dst_row = (types::dim_t) blockIdx.x;

    while (dst_row < shard_rows) {
        const types::idx_t src_row = major_order[dst_row];
        const types::idx_t part = (types::idx_t) ::cellshard::find_offset_span(src_row, part_row_offsets, part_count);
        const types::idx_t local_row = (types::idx_t) (src_row - part_row_offsets[part]);
        const types::ptr_t *src_major_ptr = part_major_ptr[part];
        const types::idx_t *src_minor = part_minor_idx[part];
        const ValueT *src_values = part_val[part];
        const types::ptr_t src_begin = src_major_ptr[local_row];
        const types::ptr_t src_end = src_major_ptr[local_row + 1u];
        const types::ptr_t dst_begin = dst_major_ptr[dst_row];
        const types::ptr_t len = src_end - src_begin;
        types::ptr_t j = (types::ptr_t) threadIdx.x;

        while (j < len) {
            dst_minor_idx[dst_begin + j] = src_minor[src_begin + j];
            dst_val[dst_begin + j] = src_values[src_begin + j];
            j += (types::ptr_t) blockDim.x;
        }

        dst_row += major_stride;
    }
}

} // namespace kernels
} // namespace bucket
} // namespace cellshard
