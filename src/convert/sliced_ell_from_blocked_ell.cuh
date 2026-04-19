#pragma once

#include "../formats/blocked_ell.cuh"
#include "../formats/sliced_ell.cuh"
#include "../sharded/sharded_device.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>

namespace cellshard {
namespace convert {

struct sliced_ell_tune_result {
    unsigned int slice_rows;
    unsigned int slice_count;
    std::size_t padded_slots;
    std::size_t padded_bytes;
    double storage_ratio;
    double slice_penalty;
    double score;
};

namespace detail {

inline int sliced_ell_cuda_check_(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

inline int count_blocked_ell_row_nnz_(const sparse::blocked_ell *src,
                                      std::unique_ptr<types::u32[]> *row_nnz_out) {
    std::unique_ptr<types::u32[]> row_nnz;
    const types::u32 rows = src != nullptr ? src->rows : 0u;
    const types::u32 block_size = src != nullptr ? src->block_size : 0u;
    const types::u32 ell_width_blocks = src != nullptr ? sparse::ell_width_blocks(src) : 0u;
    if (src == nullptr || row_nnz_out == nullptr || block_size == 0u) return 0;
    row_nnz.reset(rows != 0u ? new types::u32[rows]() : nullptr);
    if (rows != 0u && !row_nnz) return 0;
    for (types::u32 row = 0u; row < rows; ++row) {
        const types::u32 row_block = row / block_size;
        types::u32 count = 0u;
        for (types::u32 slot = 0u; slot < ell_width_blocks; ++slot) {
            const types::idx_t block_col = src->blockColIdx[(std::size_t) row_block * ell_width_blocks + slot];
            if (block_col == sparse::blocked_ell_invalid_col) continue;
            for (types::u32 lane = 0u; lane < block_size; ++lane) {
                const types::u32 col = (types::u32) block_col * block_size + lane;
                if (col >= src->cols) continue;
                const float value = __half2float(src->val[(std::size_t) row * src->ell_cols + (std::size_t) slot * block_size + lane]);
                if (value != 0.0f) ++count;
            }
        }
        row_nnz[row] = count;
    }
    *row_nnz_out = std::move(row_nnz);
    return 1;
}

inline int evaluate_sliced_candidate_(const types::u32 *row_nnz,
                                      types::dim_t rows,
                                      types::nnz_t nnz,
                                      unsigned int candidate_slice_rows,
                                      sliced_ell_tune_result *out) {
    const types::u32 slice_rows = rows == 0u
        ? 0u
        : std::max<types::u32>(1u, std::min<types::u32>((types::u32) candidate_slice_rows, (types::u32) rows));
    const std::size_t slot_bytes = sizeof(types::idx_t) + sizeof(real::storage_t);
    std::size_t padded_slots = 0u;
    types::u32 slice_count = 0u;
    types::u32 row = 0u;
    sliced_ell_tune_result result{};
    if (row_nnz == nullptr || out == nullptr || (rows != 0u && candidate_slice_rows == 0u)) return 0;
    if (rows == 0u) {
        result.slice_rows = 0u;
        result.slice_count = 0u;
        result.padded_slots = 0u;
        result.padded_bytes = 0u;
        result.storage_ratio = 1.0;
        result.slice_penalty = 0.0;
        result.score = 1.0;
        *out = result;
        return 1;
    }
    while (row < rows) {
        const types::u32 row_end = std::min<types::u32>((types::u32) rows, row + slice_rows);
        types::u32 max_width = 0u;
        for (types::u32 local = row; local < row_end; ++local) {
            max_width = std::max(max_width, row_nnz[local]);
        }
        padded_slots += (std::size_t) (row_end - row) * (std::size_t) max_width;
        row = row_end;
        ++slice_count;
    }
    result.slice_rows = slice_rows;
    result.slice_count = slice_count;
    result.padded_slots = padded_slots;
    result.padded_bytes = padded_slots * slot_bytes
        + ((std::size_t) slice_count * 2u + 1u) * sizeof(types::u32);
    result.storage_ratio = nnz == 0u ? 1.0 : (double) padded_slots / (double) nnz;
    result.slice_penalty = rows == 0u ? 0.0 : 0.25 * ((double) slice_count / (double) rows);
    result.score = result.storage_ratio + result.slice_penalty;
    *out = result;
    return 1;
}

inline int build_uniform_sliced_layout_(const types::u32 *row_nnz,
                                        types::dim_t rows,
                                        unsigned int slice_rows,
                                        std::unique_ptr<types::u32[]> *slice_row_offsets_out,
                                        std::unique_ptr<types::u32[]> *slice_widths_out,
                                        types::u32 *slice_count_out) {
    const types::u32 actual_slice_rows = rows == 0u
        ? 0u
        : std::max<types::u32>(1u, std::min<types::u32>((types::u32) slice_rows, (types::u32) rows));
    const types::u32 slice_count = rows == 0u ? 0u : (types::u32) (((types::u32) rows + actual_slice_rows - 1u) / actual_slice_rows);
    std::unique_ptr<types::u32[]> slice_row_offsets;
    std::unique_ptr<types::u32[]> slice_widths;
    if (row_nnz == nullptr || slice_row_offsets_out == nullptr || slice_widths_out == nullptr || slice_count_out == nullptr) return 0;
    slice_row_offsets.reset(new types::u32[(std::size_t) slice_count + 1u]());
    slice_widths.reset(slice_count != 0u ? new types::u32[slice_count]() : nullptr);
    if (!slice_row_offsets || (slice_count != 0u && !slice_widths)) return 0;
    slice_row_offsets[0] = 0u;
    for (types::u32 slice = 0u; slice < slice_count; ++slice) {
        const types::u32 row_begin = slice * actual_slice_rows;
        const types::u32 row_end = std::min<types::u32>((types::u32) rows, row_begin + actual_slice_rows);
        types::u32 max_width = 0u;
        for (types::u32 row = row_begin; row < row_end; ++row) {
            max_width = std::max(max_width, row_nnz[row]);
        }
        slice_row_offsets[slice + 1u] = row_end;
        slice_widths[slice] = max_width;
    }
    *slice_row_offsets_out = std::move(slice_row_offsets);
    *slice_widths_out = std::move(slice_widths);
    *slice_count_out = slice_count;
    return 1;
}

namespace kernels {

__global__ static void emit_sliced_ell_from_blocked_ell_kernel(
    device::blocked_ell_view src,
    device::sliced_ell_view dst
) {
    const types::u32 tid = (types::u32) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::u32 stride = (types::u32) (blockDim.x * gridDim.x);
    const types::u32 block_size = src.block_size;
    const types::u32 ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
    types::u32 row = tid;

    while (row < src.rows) {
        types::u32 slice = 0u;
        if (dst.slice_rows != 0u) {
            slice = row / dst.slice_rows;
        } else {
            while (slice + 1u < dst.slice_count && row >= dst.slice_row_offsets[slice + 1u]) ++slice;
        }
        const types::u32 row_begin = slice < dst.slice_count
            ? (dst.slice_rows != 0u ? slice * dst.slice_rows : dst.slice_row_offsets[slice])
            : 0u;
        const types::u32 width = slice < dst.slice_count ? dst.slice_widths[slice] : 0u;
        const std::size_t row_slot_base = slice < dst.slice_count
            ? (std::size_t) dst.slice_slot_offsets[slice] + (std::size_t) (row - row_begin) * (std::size_t) width
            : 0u;
        const types::u32 row_block = block_size != 0u ? row / block_size : 0u;
        types::u32 dst_slot = 0u;

        for (types::u32 slot = 0u; slot < ell_width_blocks; ++slot) {
            const types::idx_t block_col =
                src.blockColIdx[(std::size_t) row_block * ell_width_blocks + slot];
            if (block_col == sparse::blocked_ell_invalid_col) continue;
            for (types::u32 lane = 0u; lane < block_size; ++lane) {
                const types::u32 col = (types::u32) block_col * block_size + lane;
                const real::storage_t value =
                    src.val[(std::size_t) row * src.ell_cols + (std::size_t) slot * block_size + lane];
                if (col >= src.cols || __half2float(value) == 0.0f) continue;
                if (dst_slot >= width) continue;
                dst.col_idx[row_slot_base + dst_slot] = col;
                dst.val[row_slot_base + dst_slot] = value;
                ++dst_slot;
            }
        }
        row += stride;
    }
}

} // namespace kernels

} // namespace detail

inline int choose_sliced_ell_slice_rows(const types::u32 *row_nnz,
                                        types::dim_t rows,
                                        types::nnz_t nnz,
                                        const unsigned int *candidates,
                                        unsigned int candidate_count,
                                        sliced_ell_tune_result *out) {
    sliced_ell_tune_result best{};
    if (row_nnz == nullptr || candidates == nullptr || candidate_count == 0u || out == nullptr) return 0;
    best.slice_rows = 0u;
    best.score = std::numeric_limits<double>::infinity();
    for (unsigned int i = 0u; i < candidate_count; ++i) {
        sliced_ell_tune_result candidate{};
        if (candidates[i] == 0u) continue;
        if (!detail::evaluate_sliced_candidate_(row_nnz, rows, nnz, candidates[i], &candidate)) return 0;
        if (best.slice_rows == 0u
            || candidate.score + 1.0e-12 < best.score
            || (candidate.score <= best.score + 1.0e-12 && candidate.padded_bytes < best.padded_bytes)
            || (candidate.score <= best.score + 1.0e-12 && candidate.padded_bytes == best.padded_bytes
                && candidate.slice_rows > best.slice_rows)) {
            best = candidate;
        }
    }
    if (best.slice_rows == 0u && rows != 0u) return 0;
    *out = best;
    return 1;
}

inline int choose_sliced_ell_slice_rows_from_blocked_ell(const sparse::blocked_ell *src,
                                                         const unsigned int *candidates,
                                                         unsigned int candidate_count,
                                                         sliced_ell_tune_result *out) {
    std::unique_ptr<types::u32[]> row_nnz;
    if (src == nullptr || out == nullptr) return 0;
    if (!detail::count_blocked_ell_row_nnz_(src, &row_nnz)) return 0;
    return choose_sliced_ell_slice_rows(row_nnz.get(), src->rows, src->nnz, candidates, candidate_count, out);
}

inline int sliced_ell_from_blocked_ell(const sparse::blocked_ell *src,
                                       unsigned int slice_rows,
                                       sparse::sliced_ell *dst) {
    std::unique_ptr<types::u32[]> row_nnz;
    std::unique_ptr<types::u32[]> slice_row_offsets;
    std::unique_ptr<types::u32[]> slice_widths;
    types::u32 slice_count = 0u;
    if (src == nullptr || dst == nullptr || slice_rows == 0u) return 0;
    if (!detail::count_blocked_ell_row_nnz_(src, &row_nnz)) return 0;
    if (!detail::build_uniform_sliced_layout_(row_nnz.get(),
                                              src->rows,
                                              slice_rows,
                                              &slice_row_offsets,
                                              &slice_widths,
                                              &slice_count)) return 0;

    sparse::clear(dst);
    sparse::init(dst, src->rows, src->cols, src->nnz);
    if (!sparse::allocate(dst, slice_count, slice_row_offsets.get(), slice_widths.get())) return 0;

    for (types::u32 slice = 0u; slice < dst->slice_count; ++slice) {
        const types::u32 row_begin = dst->slice_row_offsets[slice];
        const types::u32 row_end = dst->slice_row_offsets[slice + 1u];
        const types::u32 width = dst->slice_widths[slice];
        const std::size_t slice_base = sparse::slice_slot_base(dst, slice);
        for (types::u32 row = row_begin; row < row_end; ++row) {
            const types::u32 row_block = src->block_size == 0u ? 0u : row / src->block_size;
            const std::size_t dst_row_base = slice_base + (std::size_t) (row - row_begin) * (std::size_t) width;
            std::size_t dst_slot = 0u;
            for (types::u32 slot = 0u; slot < sparse::ell_width_blocks(src); ++slot) {
                const types::idx_t block_col = src->blockColIdx[(std::size_t) row_block * sparse::ell_width_blocks(src) + slot];
                if (block_col == sparse::blocked_ell_invalid_col) continue;
                for (types::u32 lane = 0u; lane < src->block_size; ++lane) {
                    const types::u32 col = (types::u32) block_col * src->block_size + lane;
                    const real::storage_t value = src->val[(std::size_t) row * src->ell_cols + (std::size_t) slot * src->block_size + lane];
                    if (col >= src->cols || __half2float(value) == 0.0f) continue;
                    dst->col_idx[dst_row_base + dst_slot] = col;
                    dst->val[dst_row_base + dst_slot] = value;
                    ++dst_slot;
                }
            }
        }
    }
    return 1;
}

inline int sliced_ell_from_blocked_ell_cuda(const sparse::blocked_ell *src,
                                            unsigned int slice_rows,
                                            sparse::sliced_ell *dst,
                                            int device = 0,
                                            cudaStream_t stream = (cudaStream_t) 0) {
    std::unique_ptr<types::u32[]> row_nnz;
    std::unique_ptr<types::u32[]> slice_row_offsets;
    std::unique_ptr<types::u32[]> slice_widths;
    types::u32 slice_count = 0u;
    device::partition_record<sparse::blocked_ell> src_record;
    device::partition_record<sparse::sliced_ell> dst_record;
    device::blocked_ell_view src_view{};
    device::sliced_ell_view dst_view{};
    std::size_t total_slots = 0u;
    int ok = 0;

    device::zero_record(&src_record);
    device::zero_record(&dst_record);
    if (src == nullptr || dst == nullptr || slice_rows == 0u) return 0;
    if (!detail::count_blocked_ell_row_nnz_(src, &row_nnz)) return 0;
    if (!detail::build_uniform_sliced_layout_(row_nnz.get(),
                                              src->rows,
                                              slice_rows,
                                              &slice_row_offsets,
                                              &slice_widths,
                                              &slice_count)) return 0;

    sparse::clear(dst);
    sparse::init(dst, src->rows, src->cols, src->nnz);
    if (!sparse::allocate(dst, slice_count, slice_row_offsets.get(), slice_widths.get())) return 0;
    total_slots = (std::size_t) sparse::total_slots(dst);
    if (!detail::sliced_ell_cuda_check_(cudaSetDevice(device), "cudaSetDevice sliced_ell_from_blocked_ell_cuda")) goto done;
    if (!detail::sliced_ell_cuda_check_(device::upload_async(src, &src_record, stream), "upload_async blocked->sliced src")) goto done;
    if (!detail::sliced_ell_cuda_check_(device::upload_async(dst, &dst_record, stream), "upload_async blocked->sliced dst")) goto done;
    src_view.rows = src->rows;
    src_view.cols = src->cols;
    src_view.nnz = src->nnz;
    src_view.block_size = src->block_size;
    src_view.ell_cols = src->ell_cols;
    src_view.blockColIdx = reinterpret_cast<types::u32 *>(src_record.a0);
    src_view.val = reinterpret_cast<real::storage_t *>(src_record.a1);
    dst_view.rows = dst->rows;
    dst_view.cols = dst->cols;
    dst_view.nnz = dst->nnz;
    dst_view.slice_count = dst->slice_count;
    dst_view.slice_rows = sparse::uniform_slice_rows(dst);
    dst_view.slice_row_offsets = reinterpret_cast<types::u32 *>(dst_record.a0);
    dst_view.slice_widths = reinterpret_cast<types::u32 *>(dst_record.a1);
    {
        const std::size_t slice_offsets_bytes = (std::size_t) (dst->slice_count + 1u) * sizeof(types::u32);
        const std::size_t widths_offset = ((slice_offsets_bytes + alignof(types::u32) - 1u) / alignof(types::u32)) * alignof(types::u32);
        const std::size_t widths_bytes = (std::size_t) dst->slice_count * sizeof(types::u32);
        const std::size_t slot_offsets_offset = ((widths_offset + widths_bytes + alignof(types::u32) - 1u) / alignof(types::u32)) * alignof(types::u32);
        dst_view.slice_slot_offsets = reinterpret_cast<types::u32 *>(reinterpret_cast<char *>(dst_record.storage) + slot_offsets_offset);
    }
    dst_view.col_idx = reinterpret_cast<types::idx_t *>(dst_record.a2);
    dst_view.val = reinterpret_cast<real::storage_t *>(dst_record.a3);
    {
        const unsigned int threads = 256u;
        const unsigned int blocks = src->rows == 0u ? 1u : (src->rows + threads - 1u) / threads;
        detail::kernels::emit_sliced_ell_from_blocked_ell_kernel<<<blocks, threads, 0, stream>>>(src_view, dst_view);
        if (!detail::sliced_ell_cuda_check_(cudaGetLastError(), "emit_sliced_ell_from_blocked_ell_kernel")) goto done;
    }
    if (!detail::sliced_ell_cuda_check_(cudaStreamSynchronize(stream), "cudaStreamSynchronize blocked->sliced emit")) goto done;
    if (total_slots != 0u) {
        if (!detail::sliced_ell_cuda_check_(cudaMemcpy(dst->col_idx,
                                                       dst_record.a2,
                                                       total_slots * sizeof(types::idx_t),
                                                       cudaMemcpyDeviceToHost),
                                            "cudaMemcpy blocked->sliced col_idx")) goto done;
        if (!detail::sliced_ell_cuda_check_(cudaMemcpy(dst->val,
                                                       dst_record.a3,
                                                       total_slots * sizeof(real::storage_t),
                                                       cudaMemcpyDeviceToHost),
                                            "cudaMemcpy blocked->sliced val")) goto done;
    }
    ok = 1;

done:
    (void) device::release(&dst_record);
    (void) device::release(&src_record);
    if (!ok) {
        sparse::clear(dst);
    }
    return ok;
}

inline int sliced_ell_from_blocked_ell_auto(const sparse::blocked_ell *src,
                                            const unsigned int *candidates,
                                            unsigned int candidate_count,
                                            sparse::sliced_ell *dst,
                                            sliced_ell_tune_result *picked = 0) {
    sliced_ell_tune_result tune{};
    if (!choose_sliced_ell_slice_rows_from_blocked_ell(src, candidates, candidate_count, &tune)) return 0;
    if (!sliced_ell_from_blocked_ell(src, tune.slice_rows == 0u ? 1u : tune.slice_rows, dst)) return 0;
    if (picked != 0) *picked = tune;
    return 1;
}

inline int sliced_ell_from_blocked_ell_cuda_auto(const sparse::blocked_ell *src,
                                                 const unsigned int *candidates,
                                                 unsigned int candidate_count,
                                                 sparse::sliced_ell *dst,
                                                 int device = 0,
                                                 cudaStream_t stream = (cudaStream_t) 0,
                                                 sliced_ell_tune_result *picked = 0) {
    sliced_ell_tune_result tune{};
    if (!choose_sliced_ell_slice_rows_from_blocked_ell(src, candidates, candidate_count, &tune)) return 0;
    if (!sliced_ell_from_blocked_ell_cuda(src,
                                          tune.slice_rows == 0u ? 1u : tune.slice_rows,
                                          dst,
                                          device,
                                          stream)) return 0;
    if (picked != 0) *picked = tune;
    return 1;
}

} // namespace convert
} // namespace cellshard
