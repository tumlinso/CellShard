#pragma once

#include "../../extreme_ptx_primitives.cuh"
#include "../../formats/blocked_ell.cuh"
#include "../../sharded/sharded_device.cuh"

#include <cuda_runtime.h>

namespace cellshard {
namespace repack {
namespace kernels {

__device__ __forceinline__ types::idx_t shard_part_for_row(
    const types::idx_t * __restrict__ part_row_offsets,
    types::idx_t part_count,
    types::idx_t row
) {
    types::idx_t part = 0u;
    while (part + 1u < part_count && part_row_offsets[part + 1u] <= row) ++part;
    return part;
}

__global__ static void count_filtered_shard_blocked_ell_row_nnz(
    const device::blocked_ell_view * const * __restrict__ part_views,
    const types::idx_t * __restrict__ part_row_offsets,
    types::idx_t part_count,
    types::dim_t shard_rows,
    types::dim_t output_rows,
    types::dim_t output_cols,
    const unsigned char * __restrict__ keep_rows,
    const unsigned char * __restrict__ keep_cols,
    const types::idx_t * __restrict__ row_remap,
    const types::idx_t * __restrict__ col_remap,
    types::ptr_t * __restrict__ row_ptr_shifted,
    unsigned long long * __restrict__ kept_rows_out,
    unsigned long long * __restrict__ kept_slots_out
) {
    const types::idx_t tid = (types::idx_t) ::cellshard::ptx::global_tid_1d();
    const types::idx_t stride = (types::idx_t) ::cellshard::ptx::global_stride_1d();
    types::idx_t shard_row = tid;

    while (shard_row < shard_rows) {
        const int row_kept = keep_rows == 0 || keep_rows[shard_row] != 0u;
        if (row_kept) {
            const types::idx_t part = shard_part_for_row(part_row_offsets, part_count, shard_row);
            const device::blocked_ell_view src = *part_views[part];
            const types::idx_t local_row = shard_row - part_row_offsets[part];
            const types::idx_t out_row = row_remap != 0 ? row_remap[shard_row] : shard_row;
            unsigned int count = 0u;

            atomicAdd(kept_rows_out, 1ull);
            atomicAdd(kept_slots_out, (unsigned long long) src.ell_cols);

            if (out_row < output_rows) {
                const unsigned int block_size = src.block_size;
                const unsigned int row_block = block_size != 0u ? local_row / block_size : 0u;
                const unsigned int ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
                for (unsigned int ell_col = 0u; ell_col < src.ell_cols; ++ell_col) {
                    const unsigned int slot = block_size != 0u ? ell_col / block_size : 0u;
                    const unsigned int lane = block_size != 0u ? ell_col % block_size : 0u;
                    const unsigned int block_col = ell_width_blocks != 0u
                        ? src.blockColIdx[(unsigned long) row_block * ell_width_blocks + slot]
                        : sparse::blocked_ell_invalid_col;
                    const unsigned int col = block_col != sparse::blocked_ell_invalid_col
                        ? block_col * block_size + lane
                        : src.cols;
                    const __half value = src.val[(unsigned long) local_row * src.ell_cols + ell_col];
                    unsigned int out_col = 0u;
                    if (col >= src.cols || __half2float(value) == 0.0f) continue;
                    if (keep_cols != 0 && keep_cols[col] == 0u) continue;
                    out_col = col_remap != 0 ? col_remap[col] : col;
                    if (out_col >= output_cols) continue;
                    ++count;
                }
                row_ptr_shifted[out_row + 1u] = count;
            }
        }
        shard_row += stride;
    }
}

__global__ static void emit_filtered_shard_blocked_ell_compressed(
    const device::blocked_ell_view * const * __restrict__ part_views,
    const types::idx_t * __restrict__ part_row_offsets,
    types::idx_t part_count,
    types::dim_t shard_rows,
    types::dim_t output_rows,
    types::dim_t output_cols,
    const unsigned char * __restrict__ keep_rows,
    const unsigned char * __restrict__ keep_cols,
    const types::idx_t * __restrict__ row_remap,
    const types::idx_t * __restrict__ col_remap,
    const types::ptr_t * __restrict__ dst_row_ptr,
    types::idx_t * __restrict__ dst_minor_idx,
    real::storage_t * __restrict__ dst_val
) {
    const types::idx_t tid = (types::idx_t) ::cellshard::ptx::global_tid_1d();
    const types::idx_t stride = (types::idx_t) ::cellshard::ptx::global_stride_1d();
    types::idx_t shard_row = tid;

    while (shard_row < shard_rows) {
        const int row_kept = keep_rows == 0 || keep_rows[shard_row] != 0u;
        if (row_kept) {
            const types::idx_t part = shard_part_for_row(part_row_offsets, part_count, shard_row);
            const device::blocked_ell_view src = *part_views[part];
            const types::idx_t local_row = shard_row - part_row_offsets[part];
            const types::idx_t out_row = row_remap != 0 ? row_remap[shard_row] : shard_row;

            if (out_row < output_rows) {
                const unsigned int block_size = src.block_size;
                const unsigned int row_block = block_size != 0u ? local_row / block_size : 0u;
                const unsigned int ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
                unsigned int cursor = dst_row_ptr[out_row];
                for (unsigned int ell_col = 0u; ell_col < src.ell_cols; ++ell_col) {
                    const unsigned int slot = block_size != 0u ? ell_col / block_size : 0u;
                    const unsigned int lane = block_size != 0u ? ell_col % block_size : 0u;
                    const unsigned int block_col = ell_width_blocks != 0u
                        ? src.blockColIdx[(unsigned long) row_block * ell_width_blocks + slot]
                        : sparse::blocked_ell_invalid_col;
                    const unsigned int col = block_col != sparse::blocked_ell_invalid_col
                        ? block_col * block_size + lane
                        : src.cols;
                    const __half value = src.val[(unsigned long) local_row * src.ell_cols + ell_col];
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
        shard_row += stride;
    }
}

__global__ static void init_index_identity(types::nnz_t n, types::idx_t *out) {
    const types::idx_t tid = (types::idx_t) ::cellshard::ptx::global_tid_1d();
    const types::idx_t stride = (types::idx_t) ::cellshard::ptx::global_stride_1d();
    types::idx_t i = tid;
    while (i < n) {
        out[i] = i;
        i += stride;
    }
}

__global__ static void build_block_sort_keys(
    types::nnz_t nnz,
    const types::idx_t * __restrict__ row_idx,
    const types::idx_t * __restrict__ col_idx,
    types::u32 block_size,
    types::u64 * __restrict__ keys,
    types::idx_t * __restrict__ positions
) {
    const types::idx_t tid = (types::idx_t) ::cellshard::ptx::global_tid_1d();
    const types::idx_t stride = (types::idx_t) ::cellshard::ptx::global_stride_1d();
    types::idx_t i = tid;
    while (i < nnz) {
        const types::u32 row_block = block_size != 0u ? row_idx[i] / block_size : 0u;
        const types::u32 block_col = block_size != 0u ? col_idx[i] / block_size : 0u;
        keys[i] = ((types::u64) row_block << 32u) | (types::u64) block_col;
        positions[i] = i;
        i += stride;
    }
}

__global__ static void mark_block_group_heads_and_count(
    types::nnz_t nnz,
    const types::u64 * __restrict__ sorted_keys,
    types::idx_t row_blocks,
    types::idx_t * __restrict__ head_flags,
    types::idx_t * __restrict__ row_block_counts
) {
    const types::idx_t tid = (types::idx_t) ::cellshard::ptx::global_tid_1d();
    const types::idx_t stride = (types::idx_t) ::cellshard::ptx::global_stride_1d();
    types::idx_t i = tid;
    (void) row_blocks;
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

__global__ static void fill_blocked_ell_invalid_cols(
    types::idx_t count,
    types::idx_t * __restrict__ block_col_idx
) {
    const types::idx_t tid = (types::idx_t) ::cellshard::ptx::global_tid_1d();
    const types::idx_t stride = (types::idx_t) ::cellshard::ptx::global_stride_1d();
    types::idx_t i = tid;
    while (i < count) {
        block_col_idx[i] = sparse::blocked_ell_invalid_col;
        i += stride;
    }
}

__global__ static void scatter_blocked_ell_block_cols(
    types::nnz_t nnz,
    const types::u64 * __restrict__ sorted_keys,
    const types::idx_t * __restrict__ head_flags,
    const types::idx_t * __restrict__ group_ids,
    const types::ptr_t * __restrict__ row_block_offsets,
    types::u32 ell_width,
    types::idx_t * __restrict__ block_col_idx
) {
    const types::idx_t tid = (types::idx_t) ::cellshard::ptx::global_tid_1d();
    const types::idx_t stride = (types::idx_t) ::cellshard::ptx::global_stride_1d();
    types::idx_t i = tid;
    while (i < nnz) {
        if (head_flags[i] != 0u) {
            const types::u64 key = sorted_keys[i];
            const types::idx_t row_block = (types::idx_t) (key >> 32u);
            const types::idx_t block_col = (types::idx_t) (key & 0xffffffffu);
            const types::idx_t slot = group_ids[i] - row_block_offsets[row_block];
            block_col_idx[(types::u64) row_block * (types::u64) ell_width + slot] = block_col;
        }
        i += stride;
    }
}

__global__ static void scatter_blocked_ell_values(
    types::nnz_t nnz,
    const types::u64 * __restrict__ sorted_keys,
    const types::idx_t * __restrict__ sorted_pos,
    const types::idx_t * __restrict__ group_ids,
    const types::ptr_t * __restrict__ row_block_offsets,
    const types::idx_t * __restrict__ row_idx,
    const types::idx_t * __restrict__ col_idx,
    const real::storage_t * __restrict__ val,
    types::u32 block_size,
    types::u32 ell_width,
    types::u32 ell_cols,
    real::storage_t * __restrict__ out_val
) {
    const types::idx_t tid = (types::idx_t) ::cellshard::ptx::global_tid_1d();
    const types::idx_t stride = (types::idx_t) ::cellshard::ptx::global_stride_1d();
    types::idx_t i = tid;
    while (i < nnz) {
        const types::u64 key = sorted_keys[i];
        const types::idx_t src_idx = sorted_pos[i];
        const types::idx_t row_block = (types::idx_t) (key >> 32u);
        const types::idx_t slot = group_ids[i] - row_block_offsets[row_block];
        const types::idx_t row = row_idx[src_idx];
        const types::idx_t col_in_block = block_size != 0u ? col_idx[src_idx] % block_size : 0u;
        (void) ell_width;
        out_val[(types::u64) row * (types::u64) ell_cols + (types::u64) slot * block_size + col_in_block] = val[src_idx];
        i += stride;
    }
}

} // namespace kernels
} // namespace repack
} // namespace cellshard
