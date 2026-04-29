#include <CellShard/runtime/mask_groups.cuh>

#include <CellShard/formats/blocked_ell.cuh>
#include <CellShard/formats/compressed.cuh>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

#include <cuda_fp16.h>

namespace cellshard {
namespace runtime {

namespace {

struct masked_reoptimize_private {
    void *compressed_block;
    std::size_t compressed_bytes;
    void *scratch_block;
    std::size_t scratch_bytes;
};

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

std::size_t align_up_bytes(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

int cuda_ok(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CellShard runtime mask-groups CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

void bind_result(sparse_group_reduce_workspace *workspace,
                 unsigned int rows,
                 unsigned int group_count,
                 sparse_group_reduce_result *out) {
    if (out == nullptr) return;
    out->rows = rows;
    out->group_count = group_count;
    out->row_totals = workspace->row_totals;
    out->detected_features = workspace->detected_features;
    out->max_values = workspace->max_values;
    out->row_keep = workspace->row_keep;
    out->group_counts = workspace->group_counts;
    out->group_percentages = workspace->group_percentages;
}

__device__ __forceinline__ int row_participates(const row_feature_mask_view masks, unsigned int row) {
    return masks.row_keep == nullptr || masks.row_keep[row] != 0u;
}

__device__ __forceinline__ int feature_participates(const row_feature_mask_view masks, unsigned int col) {
    return masks.feature_keep == nullptr || masks.feature_keep[col] != 0u;
}

__device__ __forceinline__ unsigned int remapped_feature_or_self(const row_feature_mask_view masks, unsigned int col) {
    return masks.feature_remap != nullptr ? masks.feature_remap[col] : col;
}

__device__ __forceinline__ unsigned int remapped_row_or_self(const row_feature_mask_view masks, unsigned int row) {
    return masks.row_remap != nullptr ? masks.row_remap[row] : row;
}

__device__ __forceinline__ void emit_empty_row(unsigned int row,
                                               unsigned int group_count,
                                               const sparse_group_filter_params *filter,
                                               const row_feature_mask_view masks,
                                               float *row_totals,
                                               unsigned int *detected_features,
                                               float *max_values,
                                               unsigned char *row_keep,
                                               float *group_counts,
                                               float *group_percentages) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    if (lane == 0u) {
        row_totals[row] = 0.0f;
        detected_features[row] = 0u;
        max_values[row] = 0.0f;
        row_keep[row] = 0u;
        (void) filter;
        (void) masks;
    }
    for (unsigned int group = lane; group < group_count; group += 32u) {
        const unsigned long offset = (unsigned long) row * group_count + group;
        group_counts[offset] = 0.0f;
        group_percentages[offset] = 0.0f;
    }
}

__global__ void compute_group_reduce_blocked_ell_kernel(
    device::blocked_ell_view src,
    const std::uint32_t *__restrict__ feature_group_masks,
    row_feature_mask_view masks,
    unsigned int group_count,
    sparse_group_filter_params filter,
    float *__restrict__ row_totals,
    float *__restrict__ max_values,
    unsigned int *__restrict__ detected_features,
    unsigned char *__restrict__ row_keep,
    float *__restrict__ group_counts,
    float *__restrict__ group_percentages
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    const unsigned int block_size = src.block_size;
    const unsigned int ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
    extern __shared__ float group_scratch[];
    float *warp_groups = group_scratch + (unsigned long) warp_in_block * CELLSHARD_MAX_MASK_GROUPS;
    unsigned int row = warp_global;

    while (row < src.rows) {
        if (lane < group_count) warp_groups[lane] = 0.0f;
        __syncwarp();

        if (!row_participates(masks, row)) {
            emit_empty_row(row, group_count, &filter, masks, row_totals, detected_features, max_values, row_keep, group_counts, group_percentages);
            row += warp_stride;
            continue;
        }

        float sum = 0.0f, vmax = 0.0f, detected = 0.0f;
        const unsigned int row_block = block_size != 0u ? row / block_size : 0u;
        unsigned int ell_col = lane;

        while (ell_col < src.ell_cols) {
            const unsigned int slot = block_size != 0u ? ell_col / block_size : 0u;
            const unsigned int lane_in_block = block_size != 0u ? ell_col % block_size : 0u;
            const unsigned int block_col = ell_width_blocks != 0u
                ? src.blockColIdx[(unsigned long) row_block * ell_width_blocks + slot]
                : sparse::blocked_ell_invalid_col;
            const unsigned int gene = block_col != sparse::blocked_ell_invalid_col
                ? block_col * block_size + lane_in_block
                : src.cols;
            const float value = __half2float(src.val[(unsigned long) row * src.ell_cols + ell_col]);
            if (gene < src.cols && value != 0.0f && feature_participates(masks, gene)) {
                sum += value;
                if (feature_group_masks != nullptr && group_count != 0u) {
                    std::uint32_t mask = feature_group_masks[gene];
                    while (mask != 0u) {
                        const unsigned int group = (unsigned int) (__ffs(mask) - 1);
                        if (group < group_count) atomicAdd(warp_groups + group, value);
                        mask &= mask - 1u;
                    }
                }
                vmax = fmaxf(vmax, value);
                detected += 1.0f;
            }
            ell_col += 32u;
        }

        sum = warp_sum(sum);
        vmax = warp_max(vmax);
        detected = warp_sum(detected);
        const float row_sum = __shfl_sync(0xffffffffu, sum, 0);
        __syncwarp();

        if (lane == 0u) {
            row_totals[row] = row_sum;
            max_values[row] = vmax;
            detected_features[row] = (unsigned int) detected;
            const float group_count_for_filter = filter.fraction_group_index < group_count
                ? warp_groups[filter.fraction_group_index]
                : 0.0f;
            const float group_fraction = row_sum > 0.0f ? group_count_for_filter / row_sum : 0.0f;
            row_keep[row] = (unsigned char) (row_sum >= filter.min_total
                                             && (unsigned int) detected >= filter.min_detected_features
                                             && group_fraction <= filter.max_group_fraction);
        }
        if (group_counts != nullptr && group_percentages != nullptr && lane < group_count) {
            const float group_sum = warp_groups[lane];
            const unsigned long offset = (unsigned long) row * group_count + lane;
            group_counts[offset] = group_sum;
            group_percentages[offset] = row_sum > 0.0f ? 100.0f * group_sum / row_sum : 0.0f;
        }
        __syncwarp();

        row += warp_stride;
    }
}

__global__ void compute_group_reduce_sliced_ell_kernel(
    device::sliced_ell_view src,
    const std::uint32_t *__restrict__ feature_group_masks,
    row_feature_mask_view masks,
    unsigned int group_count,
    sparse_group_filter_params filter,
    float *__restrict__ row_totals,
    float *__restrict__ max_values,
    unsigned int *__restrict__ detected_features,
    unsigned char *__restrict__ row_keep,
    float *__restrict__ group_counts,
    float *__restrict__ group_percentages
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    extern __shared__ float group_scratch[];
    float *warp_groups = group_scratch + (unsigned long) warp_in_block * CELLSHARD_MAX_MASK_GROUPS;
    unsigned int row = warp_global;

    while (row < src.rows) {
        unsigned int slice = 0u, row_begin = 0u, width = 0u;
        unsigned long slot_base = 0ul;
        if (lane < group_count) warp_groups[lane] = 0.0f;
        __syncwarp();

        if (!row_participates(masks, row)) {
            emit_empty_row(row, group_count, &filter, masks, row_totals, detected_features, max_values, row_keep, group_counts, group_percentages);
            row += warp_stride;
            continue;
        }

        if (src.slice_count != 0u) {
            if (src.slice_rows == 32u) {
                slice = row >> 5;
                if (slice >= src.slice_count) slice = src.slice_count - 1u;
            } else if (src.slice_rows != 0u) {
                slice = row / src.slice_rows;
                if (slice >= src.slice_count) slice = src.slice_count - 1u;
            } else {
                while (slice + 1u < src.slice_count && row >= src.slice_row_offsets[slice + 1u]) ++slice;
            }
            row_begin = src.slice_row_offsets[slice];
            width = src.slice_widths[slice];
            slot_base = (unsigned long) src.slice_slot_offsets[slice]
                + (unsigned long) (row - row_begin) * (unsigned long) width;
        }

        float sum = 0.0f, vmax = 0.0f, detected = 0.0f;
        for (unsigned int slot = lane; slot < width; slot += 32u) {
            const unsigned int col = src.col_idx[slot_base + slot];
            const float value = __half2float(src.val[slot_base + slot]);
            if (col < src.cols && value != 0.0f && feature_participates(masks, col)) {
                sum += value;
                if (feature_group_masks != nullptr && group_count != 0u) {
                    std::uint32_t mask = feature_group_masks[col];
                    while (mask != 0u) {
                        const unsigned int group = (unsigned int) (__ffs(mask) - 1);
                        if (group < group_count) atomicAdd(warp_groups + group, value);
                        mask &= mask - 1u;
                    }
                }
                vmax = fmaxf(vmax, value);
                detected += 1.0f;
            }
        }

        sum = warp_sum(sum);
        vmax = warp_max(vmax);
        detected = warp_sum(detected);
        const float row_sum = __shfl_sync(0xffffffffu, sum, 0);
        __syncwarp();

        if (lane == 0u) {
            row_totals[row] = row_sum;
            max_values[row] = vmax;
            detected_features[row] = (unsigned int) detected;
            const float group_count_for_filter = filter.fraction_group_index < group_count
                ? warp_groups[filter.fraction_group_index]
                : 0.0f;
            const float group_fraction = row_sum > 0.0f ? group_count_for_filter / row_sum : 0.0f;
            row_keep[row] = (unsigned char) (row_sum >= filter.min_total
                                             && (unsigned int) detected >= filter.min_detected_features
                                             && group_fraction <= filter.max_group_fraction);
        }
        if (group_counts != nullptr && group_percentages != nullptr && lane < group_count) {
            const float group_sum = warp_groups[lane];
            const unsigned long offset = (unsigned long) row * group_count + lane;
            group_counts[offset] = group_sum;
            group_percentages[offset] = row_sum > 0.0f ? 100.0f * group_sum / row_sum : 0.0f;
        }
        __syncwarp();

        row += warp_stride;
    }
}

__global__ void compute_group_reduce_compressed_kernel(
    device::compressed_view src,
    const std::uint32_t *__restrict__ feature_group_masks,
    row_feature_mask_view masks,
    unsigned int group_count,
    sparse_group_filter_params filter,
    float *__restrict__ row_totals,
    float *__restrict__ max_values,
    unsigned int *__restrict__ detected_features,
    unsigned char *__restrict__ row_keep,
    float *__restrict__ group_counts,
    float *__restrict__ group_percentages
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    extern __shared__ float group_scratch[];
    float *warp_groups = group_scratch + (unsigned long) warp_in_block * CELLSHARD_MAX_MASK_GROUPS;
    unsigned int row = warp_global;

    while (row < src.rows) {
        if (lane < group_count) warp_groups[lane] = 0.0f;
        __syncwarp();

        if (!row_participates(masks, row)) {
            emit_empty_row(row, group_count, &filter, masks, row_totals, detected_features, max_values, row_keep, group_counts, group_percentages);
            row += warp_stride;
            continue;
        }

        const unsigned int begin = src.majorPtr[row];
        const unsigned int end = src.majorPtr[row + 1u];
        float sum = 0.0f, vmax = 0.0f, detected = 0.0f;
        for (unsigned int idx = begin + lane; idx < end; idx += 32u) {
            const unsigned int col = src.minorIdx[idx];
            const float value = __half2float(src.val[idx]);
            if (col < src.cols && value != 0.0f && feature_participates(masks, col)) {
                sum += value;
                if (feature_group_masks != nullptr && group_count != 0u) {
                    std::uint32_t mask = feature_group_masks[col];
                    while (mask != 0u) {
                        const unsigned int group = (unsigned int) (__ffs(mask) - 1);
                        if (group < group_count) atomicAdd(warp_groups + group, value);
                        mask &= mask - 1u;
                    }
                }
                vmax = fmaxf(vmax, value);
                detected += 1.0f;
            }
        }

        sum = warp_sum(sum);
        vmax = warp_max(vmax);
        detected = warp_sum(detected);
        const float row_sum = __shfl_sync(0xffffffffu, sum, 0);
        __syncwarp();

        if (lane == 0u) {
            row_totals[row] = row_sum;
            max_values[row] = vmax;
            detected_features[row] = (unsigned int) detected;
            const float group_count_for_filter = filter.fraction_group_index < group_count
                ? warp_groups[filter.fraction_group_index]
                : 0.0f;
            const float group_fraction = row_sum > 0.0f ? group_count_for_filter / row_sum : 0.0f;
            row_keep[row] = (unsigned char) (row_sum >= filter.min_total
                                             && (unsigned int) detected >= filter.min_detected_features
                                             && group_fraction <= filter.max_group_fraction);
        }
        if (group_counts != nullptr && group_percentages != nullptr && lane < group_count) {
            const float group_sum = warp_groups[lane];
            const unsigned long offset = (unsigned long) row * group_count + lane;
            group_counts[offset] = group_sum;
            group_percentages[offset] = row_sum > 0.0f ? 100.0f * group_sum / row_sum : 0.0f;
        }
        __syncwarp();

        row += warp_stride;
    }
}

__global__ void count_masked_blocked_ell_rows_kernel(device::blocked_ell_view src,
                                                     row_feature_mask_view masks,
                                                     unsigned int output_rows,
                                                     unsigned int output_cols,
                                                     unsigned int *major_ptr) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    const unsigned int block_size = src.block_size;
    const unsigned int ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
    for (unsigned int row = tid; row < src.rows; row += stride) {
        if (!row_participates(masks, row)) continue;
        const unsigned int out_row = remapped_row_or_self(masks, row);
        if (out_row >= output_rows) continue;
        unsigned int count = 0u;
        const unsigned int row_block = block_size != 0u ? row / block_size : 0u;
        for (unsigned int ell_col = 0u; ell_col < src.ell_cols; ++ell_col) {
            const unsigned int slot = block_size != 0u ? ell_col / block_size : 0u;
            const unsigned int lane_in_block = block_size != 0u ? ell_col % block_size : 0u;
            const unsigned int block_col = ell_width_blocks != 0u
                ? src.blockColIdx[(unsigned long) row_block * ell_width_blocks + slot]
                : sparse::blocked_ell_invalid_col;
            const unsigned int col = block_col != sparse::blocked_ell_invalid_col ? block_col * block_size + lane_in_block : src.cols;
            const float value = __half2float(src.val[(unsigned long) row * src.ell_cols + ell_col]);
            if (col < src.cols && value != 0.0f && feature_participates(masks, col)) {
                const unsigned int out_col = remapped_feature_or_self(masks, col);
                if (out_col < output_cols) ++count;
            }
        }
        major_ptr[out_row + 1u] = count;
    }
}

__global__ void prefix_sum_major_ptr_serial_kernel(unsigned int rows, unsigned int *major_ptr) {
    unsigned int sum = 0u;
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (unsigned int row = 0u; row < rows; ++row) {
        const unsigned int count = major_ptr[row + 1u];
        major_ptr[row] = sum;
        sum += count;
    }
    major_ptr[rows] = sum;
}

__global__ void scatter_masked_blocked_ell_to_compressed_kernel(device::blocked_ell_view src,
                                                                row_feature_mask_view masks,
                                                                unsigned int output_rows,
                                                                unsigned int output_cols,
                                                                const unsigned int *major_ptr,
                                                                unsigned int *row_write,
                                                                unsigned int *minor_idx,
                                                                __half *val) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    const unsigned int block_size = src.block_size;
    const unsigned int ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
    for (unsigned int row = tid; row < src.rows; row += stride) {
        if (!row_participates(masks, row)) continue;
        const unsigned int out_row = remapped_row_or_self(masks, row);
        if (out_row >= output_rows) continue;
        const unsigned int row_block = block_size != 0u ? row / block_size : 0u;
        for (unsigned int ell_col = 0u; ell_col < src.ell_cols; ++ell_col) {
            const unsigned int slot = block_size != 0u ? ell_col / block_size : 0u;
            const unsigned int lane_in_block = block_size != 0u ? ell_col % block_size : 0u;
            const unsigned int block_col = ell_width_blocks != 0u
                ? src.blockColIdx[(unsigned long) row_block * ell_width_blocks + slot]
                : sparse::blocked_ell_invalid_col;
            const unsigned int col = block_col != sparse::blocked_ell_invalid_col ? block_col * block_size + lane_in_block : src.cols;
            const __half hvalue = src.val[(unsigned long) row * src.ell_cols + ell_col];
            const float value = __half2float(hvalue);
            if (col < src.cols && value != 0.0f && feature_participates(masks, col)) {
                const unsigned int out_col = remapped_feature_or_self(masks, col);
                if (out_col < output_cols) {
                    const unsigned int write = major_ptr[out_row] + atomicAdd(row_write + out_row, 1u);
                    minor_idx[write] = out_col;
                    val[write] = hvalue;
                }
            }
        }
    }
}

int selected_device_id(const sparse_group_reduce_fleet_workspace *fleet, unsigned int index) {
    if (fleet == nullptr || index >= fleet->slot_count || fleet->slots == nullptr || fleet->local.device_ids == nullptr) return -1;
    return fleet->local.device_ids[fleet->slots[index]];
}

masked_reoptimize_private *private_state(masked_sparse_reoptimize_workspace *workspace) {
    return workspace != nullptr ? (masked_reoptimize_private *) workspace->private_workspace : nullptr;
}

int ensure_private_state(masked_sparse_reoptimize_workspace *workspace) {
    if (workspace == nullptr) return 0;
    if (workspace->private_workspace != nullptr) return 1;
    workspace->private_workspace = std::calloc(1u, sizeof(masked_reoptimize_private));
    return workspace->private_workspace != nullptr;
}

int reserve_reopt_compressed(masked_sparse_reoptimize_workspace *workspace,
                             unsigned int rows,
                             unsigned int nnz,
                             device::compressed_view *out) {
    if (workspace == nullptr || out == nullptr) return 0;
    if (!ensure_private_state(workspace)) return 0;
    masked_reoptimize_private *state = private_state(workspace);
    const std::size_t major_bytes = (std::size_t) (rows + 1u) * sizeof(unsigned int);
    const std::size_t minor_offset = align_up_bytes(major_bytes, alignof(unsigned int));
    const std::size_t minor_bytes = (std::size_t) nnz * sizeof(unsigned int);
    const std::size_t val_offset = align_up_bytes(minor_offset + minor_bytes, alignof(__half));
    const std::size_t val_bytes = (std::size_t) nnz * sizeof(__half);
    const std::size_t bytes = val_offset + val_bytes;
    if (bytes > state->compressed_bytes) {
        if (state->compressed_block != nullptr) {
            (void) cudaFree(state->compressed_block);
            state->compressed_block = nullptr;
            state->compressed_bytes = 0u;
        }
        if (bytes != 0u && !cuda_ok(cudaMalloc(&state->compressed_block, bytes), "cudaMalloc masked compressed output")) return 0;
        state->compressed_bytes = bytes;
    }
    char *base = (char *) state->compressed_block;
    out->majorPtr = rows + 1u != 0u ? (unsigned int *) base : nullptr;
    out->minorIdx = nnz != 0u ? (unsigned int *) (base + minor_offset) : nullptr;
    out->val = nnz != 0u ? (__half *) (base + val_offset) : nullptr;
    return 1;
}

void *reserve_reopt_scratch(masked_sparse_reoptimize_workspace *workspace, std::size_t bytes) {
    if (workspace == nullptr || !ensure_private_state(workspace)) return nullptr;
    masked_reoptimize_private *state = private_state(workspace);
    if (bytes <= state->scratch_bytes) return state->scratch_block;
    if (state->scratch_block != nullptr) {
        (void) cudaFree(state->scratch_block);
        state->scratch_block = nullptr;
        state->scratch_bytes = 0u;
    }
    if (bytes == 0u) return nullptr;
    if (!cuda_ok(cudaMalloc(&state->scratch_block, bytes), "cudaMalloc masked reoptimize scratch")) return nullptr;
    state->scratch_bytes = bytes;
    return state->scratch_block;
}

} // namespace

int predict_masked_sparse_reoptimization(const auto_reoptimize_prediction_input *input,
                                         auto_reoptimize_prediction *out) {
    if (input == nullptr || out == nullptr) return 0;
    out->should_reoptimize = 0u;
    out->estimated_net_gain_ms = 0.0f;
    return 1;
}

void init(sparse_group_reduce_workspace *workspace) {
    if (workspace == nullptr) return;
    std::memset(workspace, 0, sizeof(*workspace));
    workspace->device = -1;
}

void init(sparse_group_reduce_fleet_workspace *fleet) {
    if (fleet == nullptr) return;
    std::memset(fleet, 0, sizeof(*fleet));
    distributed::init(&fleet->local);
#if CELLSHARD_HAS_NCCL
    distributed::init(&fleet->ranked_nccl);
#endif
}

void init(masked_sparse_reoptimize_workspace *workspace) {
    if (workspace == nullptr) return;
    std::memset(workspace, 0, sizeof(*workspace));
    workspace->device = -1;
}

void clear(sparse_group_reduce_workspace *workspace) {
    if (workspace == nullptr) return;
    if (workspace->device >= 0) (void) cudaSetDevice(workspace->device);
    if (workspace->owns_stream != 0 && workspace->stream != (cudaStream_t) 0) (void) cudaStreamDestroy(workspace->stream);
    if (workspace->scratch_block != nullptr) (void) cudaFree(workspace->scratch_block);
    if (workspace->feature_block != nullptr) (void) cudaFree(workspace->feature_block);
    if (workspace->row_block != nullptr) (void) cudaFree(workspace->row_block);
    init(workspace);
}

void clear(sparse_group_reduce_fleet_workspace *fleet) {
    if (fleet == nullptr) return;
    if (fleet->devices != nullptr) {
        for (unsigned int i = 0u; i < fleet->slot_count; ++i) clear(fleet->devices + i);
    }
    if (fleet->reduce_scratch != nullptr) {
        for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
            if (fleet->reduce_scratch[i] != nullptr) {
                const int device = selected_device_id(fleet, i);
                if (device >= 0) (void) cudaSetDevice(device);
                (void) cudaFree(fleet->reduce_scratch[i]);
            }
        }
    }
#if CELLSHARD_HAS_NCCL
    distributed::clear(&fleet->ranked_nccl);
#endif
    std::free(fleet->reduce_scratch);
    std::free(fleet->reduce_scratch_bytes);
    std::free(fleet->results);
    std::free(fleet->devices);
    std::free(fleet->slots);
    distributed::clear(&fleet->local);
    init(fleet);
}

void clear(masked_sparse_reoptimize_workspace *workspace) {
    if (workspace == nullptr) return;
    if (workspace->device >= 0) (void) cudaSetDevice(workspace->device);
    if (workspace->owns_stream != 0 && workspace->stream != (cudaStream_t) 0) (void) cudaStreamDestroy(workspace->stream);
    masked_reoptimize_private *state = private_state(workspace);
    if (state != nullptr) {
        if (state->compressed_block != nullptr) (void) cudaFree(state->compressed_block);
        if (state->scratch_block != nullptr) (void) cudaFree(state->scratch_block);
        std::free(state);
    }
    init(workspace);
}

int setup(sparse_group_reduce_workspace *workspace, int device, cudaStream_t stream) {
    if (workspace == nullptr) return 0;
    clear(workspace);
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice sparse group setup")) return 0;
    workspace->device = device;
    if (stream == (cudaStream_t) 0) {
        if (!cuda_ok(cudaStreamCreateWithFlags(&workspace->stream, cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags sparse group")) return 0;
        workspace->owns_stream = 1;
    } else {
        workspace->stream = stream;
        workspace->owns_stream = 0;
    }
    return 1;
}

int setup_fleet(sparse_group_reduce_fleet_workspace *fleet,
                const sparse_group_reduce_fleet_config *config) {
    if (fleet == nullptr) return 0;
    clear(fleet);
    init(fleet);

    const unsigned int stream_flags = config != nullptr ? config->stream_flags : cudaStreamNonBlocking;
    const unsigned int enable_peer = config == nullptr || config->enable_peer_access != 0u;
    if (config != nullptr && config->device_count != 0u && config->device_ids == nullptr) return 0;
    if (!cuda_ok(distributed::discover_local(&fleet->local, 1, stream_flags), "discover sparse group fleet")) return 0;
    if (fleet->local.device_count == 0u) return 0;
    if (enable_peer != 0u && !cuda_ok(distributed::enable_peer_access(&fleet->local), "enable sparse group fleet peer access")) return 0;

    const unsigned int requested = config != nullptr ? config->device_count : 0u;
    const unsigned int selected_count = requested != 0u ? requested : fleet->local.device_count;
    fleet->slots = (unsigned int *) std::calloc((std::size_t) selected_count, sizeof(unsigned int));
    fleet->devices = (sparse_group_reduce_workspace *) std::calloc((std::size_t) selected_count, sizeof(sparse_group_reduce_workspace));
    fleet->results = (sparse_group_reduce_result *) std::calloc((std::size_t) selected_count, sizeof(sparse_group_reduce_result));
    fleet->reduce_scratch = (void **) std::calloc((std::size_t) selected_count, sizeof(void *));
    fleet->reduce_scratch_bytes = (std::size_t *) std::calloc((std::size_t) selected_count, sizeof(std::size_t));
    if (fleet->slots == nullptr || fleet->devices == nullptr || fleet->results == nullptr
        || fleet->reduce_scratch == nullptr || fleet->reduce_scratch_bytes == nullptr) {
        clear(fleet);
        return 0;
    }
    fleet->slot_count = selected_count;
    for (unsigned int i = 0u; i < selected_count; ++i) init(fleet->devices + i);

    for (unsigned int i = 0u; i < selected_count; ++i) {
        const int requested_device = requested != 0u ? config->device_ids[i] : fleet->local.device_ids[i];
        int found = -1;
        for (unsigned int slot = 0u; slot < fleet->local.device_count; ++slot) {
            if (fleet->local.device_ids[slot] == requested_device) {
                found = (int) slot;
                break;
            }
        }
        if (found < 0) {
            clear(fleet);
            return 0;
        }
        fleet->slots[i] = (unsigned int) found;
        if (!setup(fleet->devices + i, requested_device, fleet->local.streams != nullptr ? fleet->local.streams[found] : (cudaStream_t) 0)) {
            clear(fleet);
            return 0;
        }
    }

#if CELLSHARD_HAS_NCCL
    if (config != nullptr && config->ranked_nccl != nullptr && config->ranked_nccl->unique_id != nullptr) {
        if (config->ranked_nccl->local_world_ranks == nullptr
            || config->ranked_nccl->world_size <= 0
            || config->ranked_nccl->unique_id_bytes != sizeof(ncclUniqueId)) {
            clear(fleet);
            return 0;
        }
        std::unique_ptr<int[]> device_ids(new (std::nothrow) int[selected_count]);
        if (!device_ids) {
            clear(fleet);
            return 0;
        }
        for (unsigned int i = 0u; i < selected_count; ++i) device_ids[i] = selected_device_id(fleet, i);
        ncclUniqueId unique_id;
        std::memcpy(&unique_id, config->ranked_nccl->unique_id, sizeof(unique_id));
        if (distributed::init_ranked_nccl_communicator(&fleet->ranked_nccl,
                                                       device_ids.get(),
                                                       fleet->slots,
                                                       selected_count,
                                                       config->ranked_nccl->local_world_ranks,
                                                       config->ranked_nccl->world_size,
                                                       &unique_id) != ncclSuccess) {
            clear(fleet);
            return 0;
        }
    } else if (selected_count > 1u) {
        (void) distributed::init_local_nccl(&fleet->local);
    }
#else
    if (config != nullptr && config->ranked_nccl != nullptr && config->ranked_nccl->unique_id != nullptr) {
        clear(fleet);
        return 0;
    }
#endif

    return 1;
}

int setup(masked_sparse_reoptimize_workspace *workspace, int device, cudaStream_t stream) {
    if (workspace == nullptr) return 0;
    clear(workspace);
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice masked reoptimize setup")) return 0;
    workspace->device = device;
    if (stream == (cudaStream_t) 0) {
        if (!cuda_ok(cudaStreamCreateWithFlags(&workspace->stream, cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags masked reoptimize")) return 0;
        workspace->owns_stream = 1;
    } else {
        workspace->stream = stream;
        workspace->owns_stream = 0;
    }
    return ensure_private_state(workspace);
}

int reserve(sparse_group_reduce_workspace *workspace,
            unsigned int rows,
            unsigned int cols,
            unsigned int values,
            unsigned int group_count) {
    if (workspace == nullptr) return 0;
    if (group_count > CELLSHARD_MAX_MASK_GROUPS) return 0;
    if (!cuda_ok(cudaSetDevice(workspace->device >= 0 ? workspace->device : 0), "cudaSetDevice sparse group reserve")) return 0;

    if (rows > workspace->rows_capacity || group_count > workspace->group_capacity) {
        std::size_t bytes = 0u;
        char *base = nullptr;
        const unsigned int alloc_rows = rows > workspace->rows_capacity ? rows : workspace->rows_capacity;
        const unsigned int alloc_groups = group_count > workspace->group_capacity ? group_count : workspace->group_capacity;
        if (workspace->row_block != nullptr) (void) cudaFree(workspace->row_block);
        workspace->row_block = nullptr;

        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) alloc_rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) alloc_rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(unsigned int));
        bytes += (std::size_t) alloc_rows * sizeof(unsigned int);
        bytes = align_up_bytes(bytes, alignof(unsigned char));
        bytes += (std::size_t) alloc_rows * sizeof(unsigned char);
        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) alloc_rows * alloc_groups * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) alloc_rows * alloc_groups * sizeof(float);

        if (bytes != 0u && !cuda_ok(cudaMalloc(&workspace->row_block, bytes), "cudaMalloc sparse group row block")) return 0;
        base = (char *) workspace->row_block;
        bytes = 0u;
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->row_totals = (float *) (base + bytes);
        bytes += (std::size_t) alloc_rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->max_values = (float *) (base + bytes);
        bytes += (std::size_t) alloc_rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(unsigned int));
        workspace->detected_features = (unsigned int *) (base + bytes);
        bytes += (std::size_t) alloc_rows * sizeof(unsigned int);
        bytes = align_up_bytes(bytes, alignof(unsigned char));
        workspace->row_keep = (unsigned char *) (base + bytes);
        bytes += (std::size_t) alloc_rows * sizeof(unsigned char);
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->group_counts = (float *) (base + bytes);
        bytes += (std::size_t) alloc_rows * alloc_groups * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->group_percentages = (float *) (base + bytes);
        workspace->rows_capacity = alloc_rows;
        workspace->group_capacity = alloc_groups;
    }

    if (cols > workspace->cols_capacity) {
        std::size_t bytes = (std::size_t) cols * sizeof(std::uint32_t);
        if (workspace->feature_block != nullptr) (void) cudaFree(workspace->feature_block);
        workspace->feature_block = nullptr;
        if (bytes != 0u && !cuda_ok(cudaMalloc(&workspace->feature_block, bytes), "cudaMalloc sparse group feature block")) return 0;
        workspace->feature_group_masks = (std::uint32_t *) workspace->feature_block;
        workspace->cols_capacity = cols;
    }

    if (values > workspace->values_capacity) workspace->values_capacity = values;
    return 1;
}

int upload_feature_group_masks(sparse_group_reduce_workspace *workspace,
                               unsigned int cols,
                               const std::uint32_t *host_masks) {
    if (workspace == nullptr) return 0;
    if (!reserve(workspace, workspace->rows_capacity, cols, workspace->values_capacity, workspace->group_capacity)) return 0;
    if (cols == 0u) return 1;
    if (host_masks == nullptr) {
        return cuda_ok(cudaMemsetAsync(workspace->feature_group_masks,
                                       0,
                                       (std::size_t) cols * sizeof(std::uint32_t),
                                       workspace->stream),
                       "cudaMemsetAsync sparse feature group masks");
    }
    return cuda_ok(cudaMemcpyAsync(workspace->feature_group_masks,
                                   host_masks,
                                   (std::size_t) cols * sizeof(std::uint32_t),
                                   cudaMemcpyHostToDevice,
                                   workspace->stream),
                   "cudaMemcpyAsync sparse feature group masks");
}

int compute_sparse_group_reduce(const device::blocked_ell_view *src,
                                sparse_group_reduce_workspace *workspace,
                                const group_mask_config_view *groups,
                                const row_feature_mask_view *masks,
                                const sparse_group_filter_params *filter,
                                sparse_group_reduce_result *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    const unsigned int group_count = groups != nullptr ? groups->group_count : 0u;
    if (group_count > CELLSHARD_MAX_MASK_GROUPS) return 0;
    if (!reserve(workspace, src->rows, src->cols, src->rows * src->ell_cols, group_count)) return 0;
    row_feature_mask_view empty_masks{};
    const row_feature_mask_view active_masks = masks != nullptr ? *masks : empty_masks;
    const std::uint32_t *feature_masks = groups != nullptr && groups->feature_group_masks != nullptr
        ? groups->feature_group_masks
        : workspace->feature_group_masks;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    const std::size_t shared_bytes = 8u * CELLSHARD_MAX_MASK_GROUPS * sizeof(float);
    compute_group_reduce_blocked_ell_kernel<<<blocks, 256, shared_bytes, workspace->stream>>>(
        *src,
        feature_masks,
        active_masks,
        group_count,
        *filter,
        workspace->row_totals,
        workspace->max_values,
        workspace->detected_features,
        workspace->row_keep,
        group_count != 0u ? workspace->group_counts : nullptr,
        group_count != 0u ? workspace->group_percentages : nullptr);
    if (!cuda_ok(cudaGetLastError(), "compute_group_reduce_blocked_ell_kernel")) return 0;
    bind_result(workspace, src->rows, group_count, out);
    return 1;
}

int compute_sparse_group_reduce(const device::sliced_ell_view *src,
                                sparse_group_reduce_workspace *workspace,
                                const group_mask_config_view *groups,
                                const row_feature_mask_view *masks,
                                const sparse_group_filter_params *filter,
                                sparse_group_reduce_result *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    const unsigned int group_count = groups != nullptr ? groups->group_count : 0u;
    if (group_count > CELLSHARD_MAX_MASK_GROUPS) return 0;
    if (!reserve(workspace, src->rows, src->cols, src->nnz, group_count)) return 0;
    row_feature_mask_view empty_masks{};
    const row_feature_mask_view active_masks = masks != nullptr ? *masks : empty_masks;
    const std::uint32_t *feature_masks = groups != nullptr && groups->feature_group_masks != nullptr
        ? groups->feature_group_masks
        : workspace->feature_group_masks;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    const std::size_t shared_bytes = 8u * CELLSHARD_MAX_MASK_GROUPS * sizeof(float);
    compute_group_reduce_sliced_ell_kernel<<<blocks, 256, shared_bytes, workspace->stream>>>(
        *src,
        feature_masks,
        active_masks,
        group_count,
        *filter,
        workspace->row_totals,
        workspace->max_values,
        workspace->detected_features,
        workspace->row_keep,
        group_count != 0u ? workspace->group_counts : nullptr,
        group_count != 0u ? workspace->group_percentages : nullptr);
    if (!cuda_ok(cudaGetLastError(), "compute_group_reduce_sliced_ell_kernel")) return 0;
    bind_result(workspace, src->rows, group_count, out);
    return 1;
}

int compute_sparse_group_reduce_compressed_fallback(const device::compressed_view *src,
                                                   sparse_group_reduce_workspace *workspace,
                                                   const group_mask_config_view *groups,
                                                   const row_feature_mask_view *masks,
                                                   const sparse_group_filter_params *filter,
                                                   sparse_group_reduce_result *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    if (src->axis != sparse::compressed_by_row) return 0;
    const unsigned int group_count = groups != nullptr ? groups->group_count : 0u;
    if (group_count > CELLSHARD_MAX_MASK_GROUPS) return 0;
    if (!reserve(workspace, src->rows, src->cols, src->nnz, group_count)) return 0;
    row_feature_mask_view empty_masks{};
    const row_feature_mask_view active_masks = masks != nullptr ? *masks : empty_masks;
    const std::uint32_t *feature_masks = groups != nullptr && groups->feature_group_masks != nullptr
        ? groups->feature_group_masks
        : workspace->feature_group_masks;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    const std::size_t shared_bytes = 8u * CELLSHARD_MAX_MASK_GROUPS * sizeof(float);
    compute_group_reduce_compressed_kernel<<<blocks, 256, shared_bytes, workspace->stream>>>(
        *src,
        feature_masks,
        active_masks,
        group_count,
        *filter,
        workspace->row_totals,
        workspace->max_values,
        workspace->detected_features,
        workspace->row_keep,
        group_count != 0u ? workspace->group_counts : nullptr,
        group_count != 0u ? workspace->group_percentages : nullptr);
    if (!cuda_ok(cudaGetLastError(), "compute_group_reduce_compressed_kernel")) return 0;
    bind_result(workspace, src->rows, group_count, out);
    return 1;
}

int compute_sparse_group_reduce_fleet(const device::blocked_ell_view *src_by_slot,
                                      sparse_group_reduce_fleet_workspace *fleet,
                                      const group_mask_config_view *groups,
                                      const row_feature_mask_view *masks,
                                      const sparse_group_filter_params *filter,
                                      sparse_group_reduce_fleet_result *out) {
    if (src_by_slot == nullptr || fleet == nullptr || filter == nullptr || fleet->slot_count == 0u || fleet->devices == nullptr) return 0;
    const unsigned int cols = src_by_slot[0].cols;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (src_by_slot[i].cols != cols) return 0;
        if (!compute_sparse_group_reduce(src_by_slot + i, fleet->devices + i, groups, masks, filter, fleet->results + i)) return 0;
    }
    if (out != nullptr) {
        out->slot_count = fleet->slot_count;
        out->leader_index = 0u;
        out->slot_results = fleet->results;
    }
    return 1;
}

int compute_sparse_group_reduce_fleet(const device::sliced_ell_view *src_by_slot,
                                      sparse_group_reduce_fleet_workspace *fleet,
                                      const group_mask_config_view *groups,
                                      const row_feature_mask_view *masks,
                                      const sparse_group_filter_params *filter,
                                      sparse_group_reduce_fleet_result *out) {
    if (src_by_slot == nullptr || fleet == nullptr || filter == nullptr || fleet->slot_count == 0u || fleet->devices == nullptr) return 0;
    const unsigned int cols = src_by_slot[0].cols;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (src_by_slot[i].cols != cols) return 0;
        if (!compute_sparse_group_reduce(src_by_slot + i, fleet->devices + i, groups, masks, filter, fleet->results + i)) return 0;
    }
    if (out != nullptr) {
        out->slot_count = fleet->slot_count;
        out->leader_index = 0u;
        out->slot_results = fleet->results;
    }
    return 1;
}

int manual_reoptimize_masked_sparse(const device::blocked_ell_view *src,
                                    const row_feature_mask_view *masks,
                                    const masked_sparse_reoptimize_config *config,
                                    masked_sparse_reoptimize_workspace *workspace,
                                    masked_sparse_reoptimize_result *out) {
    if (src == nullptr || config == nullptr || workspace == nullptr || out == nullptr) return 0;
    if (!cuda_ok(cudaSetDevice(workspace->device >= 0 ? workspace->device : 0), "cudaSetDevice masked reoptimize")) return 0;
    row_feature_mask_view empty_masks{};
    const row_feature_mask_view active_masks = masks != nullptr ? *masks : empty_masks;
    const unsigned int output_rows = config->output_rows != 0u ? config->output_rows : src->rows;
    const unsigned int output_cols = config->output_cols != 0u ? config->output_cols : src->cols;
    const unsigned int max_nnz = src->rows * src->ell_cols;
    device::compressed_view compressed{};
    compressed.rows = output_rows;
    compressed.cols = output_cols;
    compressed.nnz = max_nnz;
    compressed.axis = sparse::compressed_by_row;
    if (!reserve_reopt_compressed(workspace, output_rows, max_nnz, &compressed)) return 0;
    if (!cuda_ok(cudaMemsetAsync(compressed.majorPtr,
                                 0,
                                 (std::size_t) (output_rows + 1u) * sizeof(unsigned int),
                                 workspace->stream),
                 "cudaMemsetAsync masked major counts")) return 0;
    unsigned int *row_write = (unsigned int *) reserve_reopt_scratch(workspace, (std::size_t) output_rows * sizeof(unsigned int));
    if (row_write == nullptr && output_rows != 0u) return 0;
    unsigned int blocks = (src->rows + 255u) >> 8;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    count_masked_blocked_ell_rows_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        active_masks,
        output_rows,
        output_cols,
        compressed.majorPtr);
    if (!cuda_ok(cudaGetLastError(), "count_masked_blocked_ell_rows_kernel")) return 0;
    prefix_sum_major_ptr_serial_kernel<<<1, 1, 0, workspace->stream>>>(output_rows, compressed.majorPtr);
    if (!cuda_ok(cudaGetLastError(), "prefix_sum_major_ptr_serial_kernel")) return 0;
    if (!cuda_ok(cudaMemsetAsync(row_write,
                                 0,
                                 (std::size_t) output_rows * sizeof(unsigned int),
                                 workspace->stream),
                 "cudaMemsetAsync masked row write")) return 0;
    scatter_masked_blocked_ell_to_compressed_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        active_masks,
        output_rows,
        output_cols,
        compressed.majorPtr,
        row_write,
        compressed.minorIdx,
        compressed.val);
    if (!cuda_ok(cudaGetLastError(), "scatter_masked_blocked_ell_to_compressed_kernel")) return 0;
    unsigned int host_nnz = 0u;
    if (!cuda_ok(cudaMemcpyAsync(&host_nnz,
                                 compressed.majorPtr + output_rows,
                                 sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost,
                                 workspace->stream),
                 "cudaMemcpyAsync masked nnz")) return 0;
    if (!cuda_ok(cudaStreamSynchronize(workspace->stream), "cudaStreamSynchronize masked reoptimize")) return 0;
    compressed.nnz = host_nnz;

    out->compressed = compressed;
    out->blocked_ell = device::blocked_ell_view{};
    out->metadata.kept_rows = output_rows;
    out->metadata.kept_features = output_cols;
    out->metadata.live_nnz = host_nnz;
    out->metadata.bytes_before = (unsigned long long) src->rows * (unsigned long long) src->ell_cols * sizeof(__half)
        + (unsigned long long) ((src->block_size != 0u ? ((src->rows + src->block_size - 1u) / src->block_size) : 0u)
                                * (src->block_size != 0u ? (src->ell_cols / src->block_size) : 0u)) * sizeof(unsigned int);
    out->metadata.bytes_after = (unsigned long long) (output_rows + 1u) * sizeof(unsigned int)
        + (unsigned long long) host_nnz * (sizeof(unsigned int) + sizeof(__half));
    out->metadata.output_layout = sparse_mask_layout_compressed;
    return 1;
}

} // namespace runtime
} // namespace cellshard
