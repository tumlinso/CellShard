#pragma once

#include "../types.cuh"

#include <cstdlib>
#include <cstring>

namespace cellshard {
namespace sparse {

enum {
    sliced_ell_host_registered = 1u << 0,
    sliced_ell_invalid_col = 0xffffffffu
};

struct alignas(16) sliced_ell {
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;
    types::u32 slice_count;
    types::u32 flags;

    void *storage;
    types::u32 *slice_row_offsets;
    types::u32 *slice_widths;
    types::idx_t *col_idx;
    real::storage_t *val;
};

__host__ __device__ __forceinline__ unsigned long pack_sliced_ell_aux(types::u32 slice_count, types::u32 total_slots) {
    return ((unsigned long) total_slots << 32u) | (unsigned long) slice_count;
}

__host__ __device__ __forceinline__ types::u32 unpack_sliced_ell_slice_count(unsigned long aux) {
    return (types::u32) (aux & 0xfffffffful);
}

__host__ __device__ __forceinline__ types::u32 unpack_sliced_ell_total_slots(unsigned long aux) {
    return (types::u32) (aux >> 32u);
}

__host__ __device__ __forceinline__ void init(
    sliced_ell * __restrict__ m,
    types::dim_t rows = 0,
    types::dim_t cols = 0,
    types::nnz_t nnz = 0
) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->slice_count = 0u;
    m->flags = 0u;
    m->storage = 0;
    m->slice_row_offsets = 0;
    m->slice_widths = 0;
    m->col_idx = 0;
    m->val = 0;
}

__host__ __device__ __forceinline__ types::u32 total_slots(const sliced_ell * __restrict__ m) {
    types::u32 total = 0u;
    types::u32 slice = 0u;
    if (m == 0 || m->slice_row_offsets == 0 || m->slice_widths == 0) return 0u;
    for (slice = 0u; slice < m->slice_count; ++slice) {
        total += (m->slice_row_offsets[slice + 1u] - m->slice_row_offsets[slice]) * m->slice_widths[slice];
    }
    return total;
}

__host__ __device__ __forceinline__ std::size_t slice_slot_base(const sliced_ell * __restrict__ m, types::u32 slice_id) {
    std::size_t base = 0u;
    types::u32 slice = 0u;
    if (m == 0 || m->slice_row_offsets == 0 || m->slice_widths == 0 || slice_id > m->slice_count) return 0u;
    for (slice = 0u; slice < slice_id; ++slice) {
        base += (std::size_t) (m->slice_row_offsets[slice + 1u] - m->slice_row_offsets[slice]) * (std::size_t) m->slice_widths[slice];
    }
    return base;
}

__host__ __device__ __forceinline__ types::u32 find_slice(const sliced_ell * __restrict__ m, types::dim_t row) {
    types::u32 slice = 0u;
    if (m == 0 || m->slice_row_offsets == 0 || row >= m->rows) return m != 0 ? m->slice_count : 0u;
    for (slice = 0u; slice < m->slice_count; ++slice) {
        if (row < m->slice_row_offsets[slice + 1u]) return slice;
    }
    return m->slice_count;
}

__host__ __device__ __forceinline__ std::size_t bytes(const sliced_ell * __restrict__ m) {
    const std::size_t offsets_bytes = (std::size_t) (m->slice_count + 1u) * sizeof(types::u32);
    const std::size_t widths_offset = ((offsets_bytes + alignof(types::u32) - 1u) / alignof(types::u32)) * alignof(types::u32);
    const std::size_t widths_bytes = (std::size_t) m->slice_count * sizeof(types::u32);
    const std::size_t col_offset = ((widths_offset + widths_bytes + alignof(types::idx_t) - 1u) / alignof(types::idx_t)) * alignof(types::idx_t);
    const std::size_t slot_count = (std::size_t) total_slots(m);
    const std::size_t val_offset = ((col_offset + slot_count * sizeof(types::idx_t) + alignof(real::storage_t) - 1u) / alignof(real::storage_t)) * alignof(real::storage_t);
    return sizeof(*m)
        + val_offset
        + slot_count * sizeof(real::storage_t);
}

__host__ __device__ __forceinline__ types::u32 row_nnz(const sliced_ell * __restrict__ m, types::dim_t row) {
    const types::u32 slice = find_slice(m, row);
    const types::u32 row_begin = slice < m->slice_count ? m->slice_row_offsets[slice] : 0u;
    const types::u32 width = slice < m->slice_count ? m->slice_widths[slice] : 0u;
    const std::size_t slot_base = slice_slot_base(m, slice) + (std::size_t) (row - row_begin) * (std::size_t) width;
    types::u32 count = 0u;
    types::u32 slot = 0u;
    if (m == 0 || slice >= m->slice_count) return 0u;
    for (slot = 0u; slot < width; ++slot) {
        if (m->col_idx[slot_base + slot] != sliced_ell_invalid_col) ++count;
    }
    return count;
}

__host__ __device__ __forceinline__ const real::storage_t *at(const sliced_ell * __restrict__ m, types::dim_t r, types::idx_t c) {
    const types::u32 slice = find_slice(m, r);
    const types::u32 row_begin = slice < m->slice_count ? m->slice_row_offsets[slice] : 0u;
    const types::u32 width = slice < m->slice_count ? m->slice_widths[slice] : 0u;
    const std::size_t slot_base = slice_slot_base(m, slice) + (std::size_t) (r - row_begin) * (std::size_t) width;
    types::u32 slot = 0u;
    if (m == 0 || r >= m->rows || c >= m->cols || slice >= m->slice_count) return 0;
    for (slot = 0u; slot < width; ++slot) {
        if (m->col_idx[slot_base + slot] == c) return m->val + slot_base + slot;
    }
    return 0;
}

__host__ __device__ __forceinline__ real::storage_t *at(sliced_ell * __restrict__ m, types::dim_t r, types::idx_t c) {
    const types::u32 slice = find_slice(m, r);
    const types::u32 row_begin = slice < m->slice_count ? m->slice_row_offsets[slice] : 0u;
    const types::u32 width = slice < m->slice_count ? m->slice_widths[slice] : 0u;
    const std::size_t slot_base = slice_slot_base(m, slice) + (std::size_t) (r - row_begin) * (std::size_t) width;
    types::u32 slot = 0u;
    if (m == 0 || r >= m->rows || c >= m->cols || slice >= m->slice_count) return 0;
    for (slot = 0u; slot < width; ++slot) {
        if (m->col_idx[slot_base + slot] == c) return m->val + slot_base + slot;
    }
    return 0;
}

__host__ __forceinline__ void clear(sliced_ell * __restrict__ m) {
    if ((m->flags & sliced_ell_host_registered) != 0u && m->storage != 0) cudaHostUnregister(m->storage);
    if (m->storage != 0) std::free(m->storage);
    else {
        std::free(m->slice_row_offsets);
        std::free(m->slice_widths);
        std::free(m->col_idx);
        std::free(m->val);
    }
    init(m);
}

__host__ __forceinline__ int allocate(sliced_ell * __restrict__ m,
                                      types::u32 slice_count,
                                      const types::u32 *slice_row_offsets,
                                      const types::u32 *slice_widths) {
    types::u32 total = 0u;
    types::u32 slice = 0u;
    std::size_t offsets_bytes = 0u;
    std::size_t widths_offset = 0u;
    std::size_t widths_bytes = 0u;
    std::size_t col_offset = 0u;
    std::size_t col_bytes = 0u;
    std::size_t val_offset = 0u;
    std::size_t val_bytes = 0u;
    std::size_t total_bytes = 0u;
    void *storage = 0;
    if (m == 0 || slice_row_offsets == 0 || slice_widths == 0) return 0;
    if (slice_count != 0u && slice_row_offsets[slice_count] != m->rows) return 0;
    for (slice = 0u; slice < slice_count; ++slice) {
        total += (slice_row_offsets[slice + 1u] - slice_row_offsets[slice]) * slice_widths[slice];
    }
    offsets_bytes = (std::size_t) (slice_count + 1u) * sizeof(types::u32);
    widths_offset = ((offsets_bytes + alignof(types::u32) - 1u) / alignof(types::u32)) * alignof(types::u32);
    widths_bytes = (std::size_t) slice_count * sizeof(types::u32);
    col_offset = ((widths_offset + widths_bytes + alignof(types::idx_t) - 1u) / alignof(types::idx_t)) * alignof(types::idx_t);
    col_bytes = (std::size_t) total * sizeof(types::idx_t);
    val_offset = ((col_offset + col_bytes + alignof(real::storage_t) - 1u) / alignof(real::storage_t)) * alignof(real::storage_t);
    val_bytes = (std::size_t) total * sizeof(real::storage_t);
    total_bytes = val_offset + val_bytes;
    if ((m->flags & sliced_ell_host_registered) != 0u && m->storage != 0) cudaHostUnregister(m->storage);
    if (m->storage != 0) std::free(m->storage);
    else {
        std::free(m->slice_row_offsets);
        std::free(m->slice_widths);
        std::free(m->col_idx);
        std::free(m->val);
    }
    m->flags = 0u;
    m->storage = 0;
    m->slice_row_offsets = 0;
    m->slice_widths = 0;
    m->col_idx = 0;
    m->val = 0;
    m->slice_count = slice_count;
    if (total_bytes == 0u) return 1;
    storage = std::malloc(total_bytes);
    if (storage == 0) {
        m->slice_count = 0u;
        return 0;
    }
    m->storage = storage;
    m->slice_row_offsets = (types::u32 *) storage;
    m->slice_widths = (types::u32 *) ((char *) storage + widths_offset);
    m->col_idx = (types::idx_t *) ((char *) storage + col_offset);
    m->val = (real::storage_t *) ((char *) storage + val_offset);
    std::memcpy(m->slice_row_offsets, slice_row_offsets, offsets_bytes);
    if (slice_count != 0u) std::memcpy(m->slice_widths, slice_widths, widths_bytes);
    for (types::u32 slot = 0u; slot < total; ++slot) m->col_idx[slot] = sliced_ell_invalid_col;
    if (val_bytes != 0u) std::memset(m->val, 0, val_bytes);
    return 1;
}

__host__ __forceinline__ int pin(sliced_ell * __restrict__ m) {
    const std::size_t total_bytes = bytes(m) - sizeof(*m);
    cudaError_t err = cudaSuccess;
    if (m->storage == 0 || total_bytes == 0u) return 1;
    if ((m->flags & sliced_ell_host_registered) != 0u) return 1;
    err = cudaHostRegister(m->storage, total_bytes, cudaHostRegisterPortable);
    if (err == cudaSuccess) {
        m->flags |= sliced_ell_host_registered;
        return 1;
    }
    if (err == cudaErrorHostMemoryAlreadyRegistered) {
        cudaGetLastError();
        m->flags |= sliced_ell_host_registered;
        return 1;
    }
    cudaGetLastError();
    return 0;
}

__host__ __forceinline__ void unpin(sliced_ell * __restrict__ m) {
    if ((m->flags & sliced_ell_host_registered) == 0u || m->storage == 0) return;
    cudaHostUnregister(m->storage);
    m->flags &= ~sliced_ell_host_registered;
}

__host__ __device__ __forceinline__ int host_registered(const sliced_ell * __restrict__ m) {
    return (m->flags & sliced_ell_host_registered) != 0u;
}

} // namespace sparse
} // namespace cellshard
