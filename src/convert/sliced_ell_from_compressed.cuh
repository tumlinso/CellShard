#pragma once

#include "../formats/compressed.cuh"
#include "../formats/sliced_ell.cuh"
#include "../sharded/sharded_device.cuh"
#include "sliced_ell_from_blocked_ell.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>

namespace cellshard {
namespace convert {

namespace detail {

inline int count_compressed_row_nnz_(const sparse::compressed *src,
                                     std::unique_ptr<types::u32[]> *row_nnz_out) {
    std::unique_ptr<types::u32[]> row_nnz;
    if (src == nullptr || row_nnz_out == nullptr) return 0;
    row_nnz.reset(src->rows != 0u ? new types::u32[src->rows]() : nullptr);
    if (src->rows != 0u && !row_nnz) return 0;
    if (src->axis == sparse::compressed_by_row) {
        for (types::u32 row = 0u; row < src->rows; ++row) {
            const types::ptr_t begin = src->majorPtr[row];
            const types::ptr_t end = src->majorPtr[row + 1u];
            row_nnz[row] = (types::u32) (end - begin);
        }
    } else {
        for (types::u32 col = 0u; col < src->cols; ++col) {
            const types::ptr_t begin = src->majorPtr[col];
            const types::ptr_t end = src->majorPtr[col + 1u];
            for (types::ptr_t cursor = begin; cursor < end; ++cursor) {
                const types::u32 row = src->minorIdx[cursor];
                if (row >= src->rows) return 0;
                row_nnz[row] += 1u;
            }
        }
    }
    *row_nnz_out = std::move(row_nnz);
    return 1;
}

namespace kernels {

__global__ static void emit_sliced_ell_from_compressed_csr_kernel(
    device::compressed_view src,
    device::sliced_ell_view dst
) {
    const types::u32 tid = (types::u32) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::u32 stride = (types::u32) (blockDim.x * gridDim.x);
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
        const types::ptr_t begin = src.majorPtr[row];
        const types::ptr_t end = src.majorPtr[row + 1u];
        types::u32 slot = 0u;
        for (types::ptr_t cursor = begin; cursor < end && slot < width; ++cursor, ++slot) {
            dst.col_idx[row_slot_base + slot] = src.minorIdx[cursor];
            dst.val[row_slot_base + slot] = src.val[cursor];
        }
        row += stride;
    }
}

} // namespace kernels

} // namespace detail

inline int choose_sliced_ell_slice_rows_from_compressed(const sparse::compressed *src,
                                                        const unsigned int *candidates,
                                                        unsigned int candidate_count,
                                                        sliced_ell_tune_result *out) {
    std::unique_ptr<types::u32[]> row_nnz;
    if (src == nullptr || out == nullptr) return 0;
    if (!detail::count_compressed_row_nnz_(src, &row_nnz)) return 0;
    return choose_sliced_ell_slice_rows(row_nnz.get(), src->rows, src->nnz, candidates, candidate_count, out);
}

inline int sliced_ell_from_compressed(const sparse::compressed *src,
                                      unsigned int slice_rows,
                                      sparse::sliced_ell *dst) {
    std::unique_ptr<types::u32[]> row_nnz;
    std::unique_ptr<types::u32[]> slice_row_offsets;
    std::unique_ptr<types::u32[]> slice_widths;
    std::unique_ptr<types::u32[]> row_write_cursor;
    types::u32 slice_count = 0u;

    if (src == nullptr || dst == nullptr || slice_rows == 0u) return 0;
    if (!detail::count_compressed_row_nnz_(src, &row_nnz)) return 0;
    if (!detail::build_uniform_sliced_layout_(row_nnz.get(),
                                              src->rows,
                                              slice_rows,
                                              &slice_row_offsets,
                                              &slice_widths,
                                              &slice_count)) return 0;

    sparse::clear(dst);
    sparse::init(dst, src->rows, src->cols, src->nnz);
    if (!sparse::allocate(dst, slice_count, slice_row_offsets.get(), slice_widths.get())) return 0;

    row_write_cursor.reset(dst->rows != 0u ? new types::u32[dst->rows]() : nullptr);
    if (dst->rows != 0u && !row_write_cursor) {
        sparse::clear(dst);
        return 0;
    }

    if (src->axis == sparse::compressed_by_row) {
        for (types::u32 slice = 0u; slice < dst->slice_count; ++slice) {
            const types::u32 row_begin = dst->slice_row_offsets[slice];
            const types::u32 row_end = dst->slice_row_offsets[slice + 1u];
            const types::u32 width = dst->slice_widths[slice];
            const std::size_t slice_base = sparse::slice_slot_base(dst, slice);
            for (types::u32 row = row_begin; row < row_end; ++row) {
                const std::size_t dst_row_base = slice_base + (std::size_t) (row - row_begin) * (std::size_t) width;
                const types::ptr_t begin = src->majorPtr[row];
                const types::ptr_t end = src->majorPtr[row + 1u];
                types::u32 slot = 0u;
                for (types::ptr_t cursor = begin; cursor < end; ++cursor, ++slot) {
                    dst->col_idx[dst_row_base + slot] = src->minorIdx[cursor];
                    dst->val[dst_row_base + slot] = src->val[cursor];
                }
            }
        }
        return 1;
    }

    for (types::u32 row = 0u; row < dst->rows; ++row) {
        const types::u32 slice = sparse::find_slice(dst, row);
        const types::u32 row_begin = slice < dst->slice_count ? dst->slice_row_offsets[slice] : 0u;
        row_write_cursor[row] = (types::u32) (sparse::slice_slot_base(dst, slice) + (std::size_t) (row - row_begin) * (std::size_t) dst->slice_widths[slice]);
    }
    for (types::u32 col = 0u; col < src->cols; ++col) {
        const types::ptr_t begin = src->majorPtr[col];
        const types::ptr_t end = src->majorPtr[col + 1u];
        for (types::ptr_t cursor = begin; cursor < end; ++cursor) {
            const types::u32 row = src->minorIdx[cursor];
            if (row >= dst->rows) {
                sparse::clear(dst);
                return 0;
            }
            dst->col_idx[row_write_cursor[row]] = col;
            dst->val[row_write_cursor[row]] = src->val[cursor];
            row_write_cursor[row] += 1u;
        }
    }
    return 1;
}

inline int sliced_ell_from_compressed_cuda(const sparse::compressed *src,
                                           unsigned int slice_rows,
                                           sparse::sliced_ell *dst,
                                           int device = 0,
                                           cudaStream_t stream = (cudaStream_t) 0) {
    std::unique_ptr<types::u32[]> row_nnz;
    std::unique_ptr<types::u32[]> slice_row_offsets;
    std::unique_ptr<types::u32[]> slice_widths;
    types::u32 slice_count = 0u;
    device::partition_record<sparse::compressed> src_record;
    device::partition_record<sparse::sliced_ell> dst_record;
    device::compressed_view src_view{};
    device::sliced_ell_view dst_view{};
    std::size_t total_slots = 0u;
    int ok = 0;

    device::zero_record(&src_record);
    device::zero_record(&dst_record);
    if (src == nullptr || dst == nullptr || slice_rows == 0u) return 0;
    if (src->axis != sparse::compressed_by_row) {
        return sliced_ell_from_compressed(src, slice_rows, dst);
    }
    if (!detail::count_compressed_row_nnz_(src, &row_nnz)) return 0;
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

    if (!detail::sliced_ell_cuda_check_(cudaSetDevice(device), "cudaSetDevice sliced_ell_from_compressed_cuda")) goto done;
    if (!detail::sliced_ell_cuda_check_(device::upload_async(src, &src_record, stream), "upload_async compressed->sliced src")) goto done;
    if (!detail::sliced_ell_cuda_check_(device::upload_async(dst, &dst_record, stream), "upload_async compressed->sliced dst")) goto done;

    src_view.rows = src->rows;
    src_view.cols = src->cols;
    src_view.nnz = src->nnz;
    src_view.axis = src->axis;
    src_view.majorPtr = reinterpret_cast<types::u32 *>(src_record.a0);
    src_view.minorIdx = reinterpret_cast<types::idx_t *>(src_record.a1);
    src_view.val = reinterpret_cast<real::storage_t *>(src_record.a2);
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
        detail::kernels::emit_sliced_ell_from_compressed_csr_kernel<<<blocks, threads, 0, stream>>>(src_view, dst_view);
        if (!detail::sliced_ell_cuda_check_(cudaGetLastError(), "emit_sliced_ell_from_compressed_csr_kernel")) goto done;
    }
    if (!detail::sliced_ell_cuda_check_(cudaStreamSynchronize(stream), "cudaStreamSynchronize compressed->sliced emit")) goto done;
    if (total_slots != 0u) {
        if (!detail::sliced_ell_cuda_check_(cudaMemcpy(dst->col_idx,
                                                       dst_record.a2,
                                                       total_slots * sizeof(types::idx_t),
                                                       cudaMemcpyDeviceToHost),
                                            "cudaMemcpy compressed->sliced col_idx")) goto done;
        if (!detail::sliced_ell_cuda_check_(cudaMemcpy(dst->val,
                                                       dst_record.a3,
                                                       total_slots * sizeof(real::storage_t),
                                                       cudaMemcpyDeviceToHost),
                                            "cudaMemcpy compressed->sliced val")) goto done;
    }
    ok = 1;

done:
    (void) device::release(&dst_record);
    (void) device::release(&src_record);
    if (!ok) sparse::clear(dst);
    return ok;
}

inline int sliced_ell_from_compressed_auto(const sparse::compressed *src,
                                           const unsigned int *candidates,
                                           unsigned int candidate_count,
                                           sparse::sliced_ell *dst,
                                           sliced_ell_tune_result *picked = 0) {
    sliced_ell_tune_result tune{};
    if (!choose_sliced_ell_slice_rows_from_compressed(src, candidates, candidate_count, &tune)) return 0;
    if (!sliced_ell_from_compressed(src, tune.slice_rows == 0u ? 1u : tune.slice_rows, dst)) return 0;
    if (picked != 0) *picked = tune;
    return 1;
}

inline int sliced_ell_from_compressed_cuda_auto(const sparse::compressed *src,
                                                const unsigned int *candidates,
                                                unsigned int candidate_count,
                                                sparse::sliced_ell *dst,
                                                int device = 0,
                                                cudaStream_t stream = (cudaStream_t) 0,
                                                sliced_ell_tune_result *picked = 0) {
    sliced_ell_tune_result tune{};
    if (!choose_sliced_ell_slice_rows_from_compressed(src, candidates, candidate_count, &tune)) return 0;
    if (!sliced_ell_from_compressed_cuda(src,
                                         tune.slice_rows == 0u ? 1u : tune.slice_rows,
                                         dst,
                                         device,
                                         stream)) return 0;
    if (picked != 0) *picked = tune;
    return 1;
}

} // namespace convert
} // namespace cellshard
