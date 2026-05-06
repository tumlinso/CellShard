#include "sliced_execution_cuda.hh"

#include "../../sharded/sharded_device.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace cellshard {
namespace {

__device__ __forceinline__ std::uint32_t find_segment(std::uint32_t exec_row,
                                                       const std::uint32_t *segment_row_offsets,
                                                       std::uint32_t segment_count) {
    std::uint32_t lo = 0u, hi = segment_count;
    while (lo + 1u < hi) {
        const std::uint32_t mid = lo + ((hi - lo) >> 1u);
        if (segment_row_offsets[mid] <= exec_row) lo = mid;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ std::uint32_t find_source_slice(device::sliced_ell_view src,
                                                           std::uint32_t row) {
    if (src.slice_rows != 0u) {
        const std::uint32_t slice = row / src.slice_rows;
        return slice < src.slice_count ? slice : src.slice_count;
    }
    std::uint32_t lo = 0u, hi = src.slice_count;
    while (lo < hi) {
        const std::uint32_t mid = lo + ((hi - lo) >> 1u);
        if (row < src.slice_row_offsets[mid + 1u]) hi = mid;
        else lo = mid + 1u;
    }
    return lo;
}

__global__ void fill_bucketed_sliced_execution_kernel(
    device::sliced_ell_view src,
    std::uint32_t rows,
    const std::uint32_t * __restrict__ row_order,
    const std::uint32_t * __restrict__ segment_row_offsets,
    const std::uint32_t * __restrict__ segment_widths,
    const std::uint32_t * __restrict__ segment_slot_offsets,
    std::uint32_t segment_count,
    std::uint32_t * __restrict__ dst_col_idx,
    real::storage_t * __restrict__ dst_val,
    std::uint32_t * __restrict__ exec_to_canonical_rows,
    std::uint32_t * __restrict__ canonical_to_exec_rows) {
    std::uint32_t exec_row = (std::uint32_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const std::uint32_t stride = (std::uint32_t) (gridDim.x * blockDim.x);

    while (exec_row < rows) {
        const std::uint32_t canonical_row = row_order[exec_row];
        const std::uint32_t segment = find_segment(exec_row, segment_row_offsets, segment_count);
        const std::uint32_t segment_row = exec_row - segment_row_offsets[segment];
        const std::uint32_t dst_width = segment_widths[segment];
        const std::uint32_t dst_base = segment_slot_offsets[segment] + segment_row * dst_width;
        const std::uint32_t src_slice = find_source_slice(src, canonical_row);
        std::uint32_t dst_slot = 0u;

        if (src_slice < src.slice_count) {
            const std::uint32_t src_row_begin = src.slice_row_offsets[src_slice];
            const std::uint32_t src_width = src.slice_widths[src_slice];
            const std::uint32_t src_base = src.slice_slot_offsets[src_slice]
                + (canonical_row - src_row_begin) * src_width;
            for (std::uint32_t src_slot = 0u; src_slot < src_width && dst_slot < dst_width; ++src_slot) {
                const std::uint32_t col = src.col_idx[src_base + src_slot];
                if (col == sparse::sliced_ell_invalid_col) continue;
                dst_col_idx[dst_base + dst_slot] = col;
                dst_val[dst_base + dst_slot] = src.val[src_base + src_slot];
                ++dst_slot;
            }
        }

        exec_to_canonical_rows[exec_row] = canonical_row;
        canonical_to_exec_rows[canonical_row] = exec_row;
        exec_row += stride;
    }
}

} // namespace

int fill_bucketed_sliced_execution_partition_cuda(bucketed_sliced_ell_partition *out,
                                                  const sparse::sliced_ell *part,
                                                  const sliced_execution_cuda_layout *layout) {
    device::partition_record<sparse::sliced_ell> src_record;
    cudaStream_t stream = 0;
    std::uint32_t *d_row_order = 0, *d_segment_offsets = 0, *d_segment_widths = 0, *d_segment_slot_offsets = 0;
    std::uint32_t *d_dst_col_idx = 0, *d_exec_to_canonical = 0, *d_canonical_to_exec = 0;
    real::storage_t *d_dst_val = 0;
    std::vector<std::uint32_t> segment_slot_offsets;
    std::uint32_t total_slots = 0u;
    cudaError_t err = cudaSuccess;
    int ok = 0;

    device::zero_record(&src_record);
    if (out == 0 || part == 0 || layout == 0
        || layout->row_order == 0
        || layout->segment_row_offsets == 0
        || (layout->segment_count != 0u && layout->segment_widths == 0)
        || out->rows != part->rows
        || out->segment_count != layout->segment_count) return 0;
    if (out->rows == 0u) return 1;

    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) goto done;
    err = device::upload_async(part, &src_record, stream);
    if (err != cudaSuccess) goto done;

    segment_slot_offsets.assign((std::size_t) layout->segment_count, 0u);
    for (std::uint32_t segment = 0u; segment < layout->segment_count; ++segment) {
        const std::uint32_t rows = layout->segment_row_offsets[segment + 1u] - layout->segment_row_offsets[segment];
        segment_slot_offsets[segment] = total_slots;
        total_slots += rows * layout->segment_widths[segment];
    }

    err = cudaMalloc((void **) &d_row_order, (std::size_t) out->rows * sizeof(std::uint32_t));
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **) &d_segment_offsets, (std::size_t) (layout->segment_count + 1u) * sizeof(std::uint32_t));
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **) &d_segment_widths, (std::size_t) layout->segment_count * sizeof(std::uint32_t));
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **) &d_segment_slot_offsets, (std::size_t) layout->segment_count * sizeof(std::uint32_t));
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **) &d_exec_to_canonical, (std::size_t) out->rows * sizeof(std::uint32_t));
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **) &d_canonical_to_exec, (std::size_t) out->rows * sizeof(std::uint32_t));
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **) &d_dst_col_idx, std::max<std::size_t>(1u, (std::size_t) total_slots) * sizeof(std::uint32_t));
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **) &d_dst_val, std::max<std::size_t>(1u, (std::size_t) total_slots) * sizeof(real::storage_t));
    if (err != cudaSuccess) goto done;

    err = cudaMemcpyAsync(d_row_order,
                          layout->row_order,
                          (std::size_t) out->rows * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) goto done;
    err = cudaMemcpyAsync(d_segment_offsets,
                          layout->segment_row_offsets,
                          (std::size_t) (layout->segment_count + 1u) * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) goto done;
    err = cudaMemcpyAsync(d_segment_widths,
                          layout->segment_widths,
                          (std::size_t) layout->segment_count * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) goto done;
    err = cudaMemcpyAsync(d_segment_slot_offsets,
                          segment_slot_offsets.data(),
                          (std::size_t) layout->segment_count * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) goto done;
    err = cudaMemsetAsync(d_dst_col_idx, 0xff, (std::size_t) total_slots * sizeof(std::uint32_t), stream);
    if (err != cudaSuccess) goto done;
    err = cudaMemsetAsync(d_dst_val, 0, (std::size_t) total_slots * sizeof(real::storage_t), stream);
    if (err != cudaSuccess) goto done;

    {
        const unsigned int threads = 256u;
        unsigned int blocks = (out->rows + threads - 1u) / threads;
        if (blocks == 0u) blocks = 1u;
        if (blocks > 4096u) blocks = 4096u;
        device::sliced_ell_view src_view{};
        src_view.rows = part->rows;
        src_view.cols = part->cols;
        src_view.nnz = part->nnz;
        src_view.slice_count = part->slice_count;
        src_view.slice_rows = sparse::uniform_slice_rows(part);
        src_view.slice_row_offsets = reinterpret_cast<std::uint32_t *>(src_record.a0);
        src_view.slice_widths = reinterpret_cast<std::uint32_t *>(src_record.a1);
        src_view.col_idx = reinterpret_cast<std::uint32_t *>(src_record.a2);
        src_view.val = reinterpret_cast<real::storage_t *>(src_record.a3);
        {
            const std::size_t slice_offsets_bytes = (std::size_t) (part->slice_count + 1u) * sizeof(std::uint32_t);
            const std::size_t widths_offset = device::align_up_bytes(slice_offsets_bytes, alignof(std::uint32_t));
            const std::size_t widths_bytes = (std::size_t) part->slice_count * sizeof(std::uint32_t);
            const std::size_t slot_offsets_offset = device::align_up_bytes(widths_offset + widths_bytes, alignof(std::uint32_t));
            src_view.slice_slot_offsets = reinterpret_cast<std::uint32_t *>(reinterpret_cast<char *>(src_record.storage) + slot_offsets_offset);
        }
        fill_bucketed_sliced_execution_kernel<<<blocks, threads, 0, stream>>>(
            src_view,
            out->rows,
            d_row_order,
            d_segment_offsets,
            d_segment_widths,
            d_segment_slot_offsets,
            layout->segment_count,
            d_dst_col_idx,
            d_dst_val,
            d_exec_to_canonical,
            d_canonical_to_exec);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto done;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) goto done;
    err = cudaMemcpy(out->exec_to_canonical_rows,
                     d_exec_to_canonical,
                     (std::size_t) out->rows * sizeof(std::uint32_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto done;
    err = cudaMemcpy(out->canonical_to_exec_rows,
                     d_canonical_to_exec,
                     (std::size_t) out->rows * sizeof(std::uint32_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto done;
    for (std::uint32_t segment = 0u; segment < out->segment_count; ++segment) {
        sparse::sliced_ell *dst = out->segments + segment;
        const std::size_t slots = (std::size_t) sparse::total_slots(dst);
        if (slots == 0u) continue;
        err = cudaMemcpy(dst->col_idx,
                         d_dst_col_idx + segment_slot_offsets[segment],
                         slots * sizeof(std::uint32_t),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto done;
        err = cudaMemcpy(dst->val,
                         d_dst_val + segment_slot_offsets[segment],
                         slots * sizeof(real::storage_t),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto done;
    }

    ok = 1;

done:
    if (d_dst_val != 0) cudaFree(d_dst_val);
    if (d_dst_col_idx != 0) cudaFree(d_dst_col_idx);
    if (d_canonical_to_exec != 0) cudaFree(d_canonical_to_exec);
    if (d_exec_to_canonical != 0) cudaFree(d_exec_to_canonical);
    if (d_segment_slot_offsets != 0) cudaFree(d_segment_slot_offsets);
    if (d_segment_widths != 0) cudaFree(d_segment_widths);
    if (d_segment_offsets != 0) cudaFree(d_segment_offsets);
    if (d_row_order != 0) cudaFree(d_row_order);
    (void) device::release(&src_record);
    if (stream != 0) cudaStreamDestroy(stream);
    return ok;
}

} // namespace cellshard
