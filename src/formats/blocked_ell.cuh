#pragma once

#include "../types.cuh"

#include <cstdlib>
#include <cuda_runtime.h>

namespace cellshard {
namespace sparse {

enum {
    blocked_ell_host_registered = 1u << 0,
    blocked_ell_invalid_col = 0xffffffffu
};

struct alignas(16) blocked_ell {
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;
    types::u32 block_size;
    types::u32 ell_cols;
    types::u32 flags;

    void *storage;
    types::idx_t *blockColIdx;
    real::storage_t *val;
};

__host__ __device__ __forceinline__ types::dim_t row_block_count(const blocked_ell * __restrict__ m) {
    if (m->block_size == 0u) return 0u;
    return (m->rows + m->block_size - 1u) / m->block_size;
}

__host__ __device__ __forceinline__ types::dim_t col_block_count(const blocked_ell * __restrict__ m) {
    if (m->block_size == 0u) return 0u;
    return (m->cols + m->block_size - 1u) / m->block_size;
}

__host__ __device__ __forceinline__ types::u32 ell_width_blocks(const blocked_ell * __restrict__ m) {
    if (m->block_size == 0u || m->ell_cols == 0u) return 0u;
    return m->ell_cols / m->block_size;
}

__host__ __device__ __forceinline__ unsigned long pack_blocked_ell_aux(types::u32 block_size, unsigned long ell_width) {
    return ((unsigned long) ell_width << 16u) | (unsigned long) block_size;
}

__host__ __device__ __forceinline__ types::u32 unpack_blocked_ell_block_size(unsigned long aux) {
    return (types::u32) (aux & 0xfffful);
}

__host__ __device__ __forceinline__ unsigned long unpack_blocked_ell_ell_width(unsigned long aux) {
    return aux >> 16u;
}

__host__ __device__ __forceinline__ types::u32 unpack_blocked_ell_cols(unsigned long aux) {
    return (types::u32) (unpack_blocked_ell_ell_width(aux) * (unsigned long) unpack_blocked_ell_block_size(aux));
}

__host__ __device__ __forceinline__ void init(
    blocked_ell * __restrict__ m,
    types::dim_t rows = 0,
    types::dim_t cols = 0,
    types::nnz_t nnz = 0,
    types::u32 block_size = 0,
    types::u32 ell_cols = 0
) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->block_size = block_size;
    m->ell_cols = ell_cols;
    m->flags = 0u;
    m->storage = 0;
    m->blockColIdx = 0;
    m->val = 0;
}

__host__ __device__ __forceinline__ std::size_t bytes(const blocked_ell * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) row_block_count(m) * (std::size_t) ell_width_blocks(m) * sizeof(types::idx_t)
        + (std::size_t) m->rows * (std::size_t) m->ell_cols * sizeof(real::storage_t);
}

__host__ __device__ __forceinline__ const real::storage_t *at(const blocked_ell * __restrict__ m, types::dim_t r, types::idx_t c) {
    const types::u32 block = m->block_size;
    const types::u32 width = ell_width_blocks(m);
    const types::u32 row_block = block == 0u ? 0u : r / block;
    const types::u32 block_col = block == 0u ? 0u : c / block;
    const types::u32 col_in_block = block == 0u ? 0u : c % block;
    if (block == 0u || r >= m->rows || c >= m->cols) return 0;
    for (types::u32 slot = 0; slot < width; ++slot) {
        const types::idx_t stored = m->blockColIdx[(std::size_t) row_block * width + slot];
        if (stored == block_col) {
            return m->val + (std::size_t) r * m->ell_cols + (std::size_t) slot * block + col_in_block;
        }
    }
    return 0;
}

__host__ __device__ __forceinline__ real::storage_t *at(blocked_ell * __restrict__ m, types::dim_t r, types::idx_t c) {
    const types::u32 block = m->block_size;
    const types::u32 width = ell_width_blocks(m);
    const types::u32 row_block = block == 0u ? 0u : r / block;
    const types::u32 block_col = block == 0u ? 0u : c / block;
    const types::u32 col_in_block = block == 0u ? 0u : c % block;
    if (block == 0u || r >= m->rows || c >= m->cols) return 0;
    for (types::u32 slot = 0; slot < width; ++slot) {
        const types::idx_t stored = m->blockColIdx[(std::size_t) row_block * width + slot];
        if (stored == block_col) {
            return m->val + (std::size_t) r * m->ell_cols + (std::size_t) slot * block + col_in_block;
        }
    }
    return 0;
}

__host__ __forceinline__ void clear(blocked_ell * __restrict__ m) {
    if ((m->flags & blocked_ell_host_registered) != 0u && m->storage != 0) cudaHostUnregister(m->storage);
    if (m->storage != 0) std::free(m->storage);
    else {
        std::free(m->blockColIdx);
        std::free(m->val);
    }
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->block_size = 0;
    m->ell_cols = 0;
    m->flags = 0u;
    m->storage = 0;
    m->blockColIdx = 0;
    m->val = 0;
}

__host__ __forceinline__ int allocate(blocked_ell * __restrict__ m) {
    const std::size_t idx_bytes = (std::size_t) row_block_count(m) * (std::size_t) ell_width_blocks(m) * sizeof(types::idx_t);
    const std::size_t val_offset = ((idx_bytes + alignof(real::storage_t) - 1u) / alignof(real::storage_t)) * alignof(real::storage_t);
    const std::size_t total_bytes = val_offset + (std::size_t) m->rows * (std::size_t) m->ell_cols * sizeof(real::storage_t);
    void *storage = 0;

    if ((m->flags & blocked_ell_host_registered) != 0u && m->storage != 0) cudaHostUnregister(m->storage);
    if (m->storage != 0) std::free(m->storage);
    else {
        std::free(m->blockColIdx);
        std::free(m->val);
    }
    m->flags = 0u;
    m->storage = 0;
    m->blockColIdx = 0;
    m->val = 0;

    if (total_bytes == 0u) return 1;
    storage = std::malloc(total_bytes);
    if (storage == 0) return 0;
    m->storage = storage;
    m->blockColIdx = idx_bytes != 0u ? (types::idx_t *) storage : 0;
    m->val = m->rows != 0u && m->ell_cols != 0u ? (real::storage_t *) ((char *) storage + val_offset) : 0;
    return 1;
}

__host__ __forceinline__ int pin(blocked_ell * __restrict__ m) {
    const std::size_t idx_bytes = (std::size_t) row_block_count(m) * (std::size_t) ell_width_blocks(m) * sizeof(types::idx_t);
    const std::size_t val_offset = ((idx_bytes + alignof(real::storage_t) - 1u) / alignof(real::storage_t)) * alignof(real::storage_t);
    const std::size_t total_bytes = val_offset + (std::size_t) m->rows * (std::size_t) m->ell_cols * sizeof(real::storage_t);
    cudaError_t err = cudaSuccess;

    if (m->storage == 0 || total_bytes == 0u) return 1;
    if ((m->flags & blocked_ell_host_registered) != 0u) return 1;
    err = cudaHostRegister(m->storage, total_bytes, cudaHostRegisterPortable);
    if (err == cudaSuccess) {
        m->flags |= blocked_ell_host_registered;
        return 1;
    }
    if (err == cudaErrorHostMemoryAlreadyRegistered) {
        cudaGetLastError();
        m->flags |= blocked_ell_host_registered;
        return 1;
    }
    cudaGetLastError();
    return 0;
}

__host__ __forceinline__ void unpin(blocked_ell * __restrict__ m) {
    if ((m->flags & blocked_ell_host_registered) == 0u || m->storage == 0) return;
    cudaHostUnregister(m->storage);
    m->flags &= ~blocked_ell_host_registered;
}

__host__ __device__ __forceinline__ int host_registered(const blocked_ell * __restrict__ m) {
    return (m->flags & blocked_ell_host_registered) != 0u;
}

} // namespace sparse
} // namespace cellshard
