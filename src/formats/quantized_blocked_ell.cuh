#pragma once

#include "../types.cuh"

#include <cstdlib>

namespace cellshard {
namespace sparse {

enum {
    quantized_blocked_ell_host_registered = 1u << 0,
    quantized_blocked_ell_invalid_col = 0xffffffffu
};

__host__ __device__ __forceinline__ unsigned long pack_quantized_blocked_ell_aux(
    types::u32 bits,
    types::u32 block_size,
    unsigned long ell_width) {
    return (ell_width << 24u) | ((unsigned long) (bits & 0xffu) << 16u) | (unsigned long) block_size;
}

__host__ __device__ __forceinline__ types::u32 unpack_quantized_blocked_ell_block_size(unsigned long aux) {
    return (types::u32) (aux & 0xfffful);
}

__host__ __device__ __forceinline__ types::u32 unpack_quantized_blocked_ell_bits(unsigned long aux) {
    return (types::u32) ((aux >> 16u) & 0xfful);
}

__host__ __device__ __forceinline__ unsigned long unpack_quantized_blocked_ell_ell_width(unsigned long aux) {
    return aux >> 24u;
}

__host__ __device__ __forceinline__ types::u32 unpack_quantized_blocked_ell_cols(unsigned long aux) {
    return (types::u32) (unpack_quantized_blocked_ell_ell_width(aux) * (unsigned long) unpack_quantized_blocked_ell_block_size(aux));
}

__host__ __device__ __forceinline__ types::u32 quantized_blocked_ell_codes_per_byte(types::u32 bits) {
    return bits == 0u ? 0u : 8u / bits;
}

__host__ __device__ __forceinline__ types::u32 quantized_blocked_ell_row_bytes(types::u32 bits, types::u32 ell_cols) {
    const types::u32 codes_per_byte = quantized_blocked_ell_codes_per_byte(bits);
    return codes_per_byte == 0u ? 0u : (ell_cols + codes_per_byte - 1u) / codes_per_byte;
}

__host__ __device__ __forceinline__ types::u32 quantized_blocked_ell_aligned_row_bytes(
    types::u32 bits,
    types::u32 ell_cols,
    types::u32 alignment = 16u) {
    const types::u32 bytes = quantized_blocked_ell_row_bytes(bits, ell_cols);
    return alignment <= 1u ? bytes : ((bytes + alignment - 1u) / alignment) * alignment;
}

struct alignas(16) quantized_blocked_ell {
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;
    types::u32 block_size;
    types::u32 ell_cols;
    types::u32 bits;
    types::u32 row_stride_bytes;
    types::u32 decode_policy;
    types::u32 flags;

    void *storage;
    types::idx_t *blockColIdx;
    std::uint8_t *packed_values;
    float *column_scales;
    float *column_offsets;
    float *row_offsets;
};

__host__ __device__ __forceinline__ types::dim_t row_block_count(const quantized_blocked_ell * __restrict__ m) {
    if (m->block_size == 0u) return 0u;
    return (m->rows + m->block_size - 1u) / m->block_size;
}

__host__ __device__ __forceinline__ types::u32 ell_width_blocks(const quantized_blocked_ell * __restrict__ m) {
    if (m->block_size == 0u || m->ell_cols == 0u) return 0u;
    return m->ell_cols / m->block_size;
}

__host__ __device__ __forceinline__ std::size_t block_col_idx_count(const quantized_blocked_ell * __restrict__ m) {
    return (std::size_t) row_block_count(m) * (std::size_t) ell_width_blocks(m);
}

__host__ __device__ __forceinline__ std::size_t packed_value_bytes(const quantized_blocked_ell * __restrict__ m) {
    return (std::size_t) m->rows * (std::size_t) m->row_stride_bytes;
}

__host__ __device__ __forceinline__ void init(
    quantized_blocked_ell * __restrict__ m,
    types::dim_t rows = 0,
    types::dim_t cols = 0,
    types::nnz_t nnz = 0,
    types::u32 block_size = 0,
    types::u32 ell_cols = 0,
    types::u32 bits = 0,
    types::u32 decode_policy = 0,
    types::u32 row_stride_bytes = 0) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->block_size = block_size;
    m->ell_cols = ell_cols;
    m->bits = bits;
    m->row_stride_bytes = row_stride_bytes != 0u ? row_stride_bytes : quantized_blocked_ell_aligned_row_bytes(bits, ell_cols);
    m->decode_policy = decode_policy;
    m->flags = 0u;
    m->storage = 0;
    m->blockColIdx = 0;
    m->packed_values = 0;
    m->column_scales = 0;
    m->column_offsets = 0;
    m->row_offsets = 0;
}

__host__ __device__ __forceinline__ std::size_t bytes(const quantized_blocked_ell * __restrict__ m) {
    return sizeof(*m)
        + block_col_idx_count(m) * sizeof(types::idx_t)
        + packed_value_bytes(m)
        + (std::size_t) m->cols * sizeof(float)
        + (std::size_t) m->cols * sizeof(float)
        + (std::size_t) m->rows * sizeof(float);
}

__host__ __forceinline__ void clear(quantized_blocked_ell * __restrict__ m) {
    if ((m->flags & quantized_blocked_ell_host_registered) != 0u && m->storage != 0) cudaHostUnregister(m->storage);
    if (m->storage != 0) std::free(m->storage);
    else {
        std::free(m->blockColIdx);
        std::free(m->packed_values);
        std::free(m->column_scales);
        std::free(m->column_offsets);
        std::free(m->row_offsets);
    }
    init(m);
}

__host__ __forceinline__ int allocate(quantized_blocked_ell * __restrict__ m) {
    const std::size_t idx_bytes = block_col_idx_count(m) * sizeof(types::idx_t);
    const std::size_t packed_offset = ((idx_bytes + alignof(std::uint8_t) - 1u) / alignof(std::uint8_t)) * alignof(std::uint8_t);
    const std::size_t packed_bytes = packed_value_bytes(m);
    const std::size_t column_scales_offset = ((packed_offset + packed_bytes + alignof(float) - 1u) / alignof(float)) * alignof(float);
    const std::size_t column_offsets_offset = column_scales_offset + (std::size_t) m->cols * sizeof(float);
    const std::size_t row_offsets_offset = column_offsets_offset + (std::size_t) m->cols * sizeof(float);
    const std::size_t total_bytes = row_offsets_offset + (std::size_t) m->rows * sizeof(float);
    void *storage = 0;

    if ((m->flags & quantized_blocked_ell_host_registered) != 0u && m->storage != 0) cudaHostUnregister(m->storage);
    if (m->storage != 0) std::free(m->storage);
    else {
        std::free(m->blockColIdx);
        std::free(m->packed_values);
        std::free(m->column_scales);
        std::free(m->column_offsets);
        std::free(m->row_offsets);
    }
    m->flags = 0u;
    m->storage = 0;
    m->blockColIdx = 0;
    m->packed_values = 0;
    m->column_scales = 0;
    m->column_offsets = 0;
    m->row_offsets = 0;

    if (total_bytes == 0u) return 1;
    storage = std::malloc(total_bytes);
    if (storage == 0) return 0;
    m->storage = storage;
    m->blockColIdx = idx_bytes != 0u ? (types::idx_t *) storage : 0;
    m->packed_values = packed_bytes != 0u ? (std::uint8_t *) ((char *) storage + packed_offset) : 0;
    m->column_scales = m->cols != 0u ? (float *) ((char *) storage + column_scales_offset) : 0;
    m->column_offsets = m->cols != 0u ? (float *) ((char *) storage + column_offsets_offset) : 0;
    m->row_offsets = m->rows != 0u ? (float *) ((char *) storage + row_offsets_offset) : 0;
    return 1;
}

__host__ __forceinline__ int pin(quantized_blocked_ell * __restrict__ m) {
    const std::size_t total_bytes = bytes(m) - sizeof(*m);
    cudaError_t err = cudaSuccess;

    if (m->storage == 0 || total_bytes == 0u) return 1;
    if ((m->flags & quantized_blocked_ell_host_registered) != 0u) return 1;
    err = cudaHostRegister(m->storage, total_bytes, cudaHostRegisterPortable);
    if (err == cudaSuccess) {
        m->flags |= quantized_blocked_ell_host_registered;
        return 1;
    }
    if (err == cudaErrorHostMemoryAlreadyRegistered) {
        cudaGetLastError();
        m->flags |= quantized_blocked_ell_host_registered;
        return 1;
    }
    cudaGetLastError();
    return 0;
}

__host__ __forceinline__ void unpin(quantized_blocked_ell * __restrict__ m) {
    if ((m->flags & quantized_blocked_ell_host_registered) == 0u || m->storage == 0) return;
    cudaHostUnregister(m->storage);
    m->flags &= ~quantized_blocked_ell_host_registered;
}

__host__ __device__ __forceinline__ int host_registered(const quantized_blocked_ell * __restrict__ m) {
    return (m->flags & quantized_blocked_ell_host_registered) != 0u;
}

} // namespace sparse
} // namespace cellshard
