#pragma once

#include "../types.cuh"

#include <cstdlib>
#include <cuda_runtime.h>

namespace cellshard {
namespace sparse {

// CSR/CSC axis selector.
enum {
    compressed_by_row = 0,
    compressed_by_col = 1
};

// Host-side storage flags. These are not part of the on-device compressed_view;
// they only describe how the packed host allocation is managed.
enum {
    compressed_host_registered = 1u << 0
};

// CSR/CSC-style sparse layout.
// axis decides whether majorPtr walks rows or columns.
struct alignas(16) compressed {
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;
    types::u32 axis;
    types::u32 flags;

    // When non-null, this owns the packed host allocation containing all three
    // arrays. majorPtr/minorIdx/val point inside it.
    void *storage;
    types::ptr_t *majorPtr;
    types::idx_t *minorIdx;
    real::storage_t *val;
};

// Axis-dependent dimensions are kept explicit so callers can reason about exact
// memory traffic and indexing.
__host__ __device__ __forceinline__ types::dim_t major_dim(const compressed * __restrict__ m) {
    return m->axis == compressed_by_col ? m->cols : m->rows;
}

__host__ __device__ __forceinline__ types::dim_t minor_dim(const compressed * __restrict__ m) {
    return m->axis == compressed_by_col ? m->rows : m->cols;
}

// Metadata-only init.
__host__ __device__ __forceinline__ void init(
    compressed * __restrict__ m,
    types::dim_t rows = 0,
    types::dim_t cols = 0,
    types::nnz_t nnz = 0,
    types::u32 axis = compressed_by_row
) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->axis = axis;
    m->flags = 0;
    m->storage = 0;
    m->majorPtr = 0;
    m->minorIdx = 0;
    m->val = 0;
}

// Total resident host bytes for one materialized compressed part.
__host__ __device__ __forceinline__ std::size_t bytes(const compressed * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) (major_dim(m) + 1) * sizeof(types::ptr_t)
        + (std::size_t) m->nnz * sizeof(types::idx_t)
        + (std::size_t) m->nnz * sizeof(real::storage_t);
}

// Convenience random access by scanning one major segment. This is not meant to
// be a hot-path sparse lookup primitive.
__host__ __device__ __forceinline__ const real::storage_t *at(const compressed * __restrict__ m, types::dim_t r, types::idx_t c) {
    const types::dim_t major = m->axis == compressed_by_col ? c : r;
    const types::idx_t minor = m->axis == compressed_by_col ? r : c;
    const types::ptr_t begin = m->majorPtr[major];
    const types::ptr_t end = m->majorPtr[major + 1];
    for (types::ptr_t i = begin; i < end; ++i) {
        if (m->minorIdx[i] == minor) return m->val + i;
    }
    return 0;
}

__host__ __device__ __forceinline__ real::storage_t *at(compressed * __restrict__ m, types::dim_t r, types::idx_t c) {
    const types::dim_t major = m->axis == compressed_by_col ? c : r;
    const types::idx_t minor = m->axis == compressed_by_col ? r : c;
    const types::ptr_t begin = m->majorPtr[major];
    const types::ptr_t end = m->majorPtr[major + 1];
    for (types::ptr_t i = begin; i < end; ++i) {
        if (m->minorIdx[i] == minor) return m->val + i;
    }
    return 0;
}

// Release all three host arrays and reset metadata.
__host__ __forceinline__ void clear(compressed * __restrict__ m) {
    if ((m->flags & compressed_host_registered) != 0u && m->storage != 0) cudaHostUnregister(m->storage);
    if (m->storage != 0) std::free(m->storage);
    else {
        std::free(m->majorPtr);
        std::free(m->minorIdx);
        std::free(m->val);
    }
    m->storage = 0;
    m->majorPtr = 0;
    m->minorIdx = 0;
    m->val = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->axis = compressed_by_row;
    m->flags = 0;
}

// allocate() tears down prior storage and rebuilds all arrays from current
// rows/cols/nnz/axis metadata.
__host__ __forceinline__ int allocate(compressed * __restrict__ m) {
    const std::size_t ptr_count = (std::size_t) major_dim(m) + 1;
    const std::size_t major_bytes = ptr_count * sizeof(types::ptr_t);
    const std::size_t minor_offset = ((major_bytes + alignof(types::idx_t) - 1u) / alignof(types::idx_t)) * alignof(types::idx_t);
    const std::size_t val_offset = ((minor_offset + (std::size_t) m->nnz * sizeof(types::idx_t) + alignof(real::storage_t) - 1u) / alignof(real::storage_t)) * alignof(real::storage_t);
    const std::size_t total_bytes = val_offset + (std::size_t) m->nnz * sizeof(real::storage_t);
    void *storage = 0;

    if ((m->flags & compressed_host_registered) != 0u && m->storage != 0) cudaHostUnregister(m->storage);
    if (m->storage != 0) std::free(m->storage);
    else {
        std::free(m->majorPtr);
        std::free(m->minorIdx);
        std::free(m->val);
    }
    m->flags = 0;
    m->storage = 0;
    m->majorPtr = 0;
    m->minorIdx = 0;
    m->val = 0;

    if (total_bytes == 0) return 1;
    storage = std::malloc(total_bytes);
    if (storage == 0) return 0;
    m->storage = storage;
    m->majorPtr = ptr_count != 0 ? (types::ptr_t *) storage : 0;
    m->minorIdx = m->nnz != 0 ? (types::idx_t *) ((char *) storage + minor_offset) : 0;
    m->val = m->nnz != 0 ? (real::storage_t *) ((char *) storage + val_offset) : 0;
    return 1;
}

// Register the packed host payload with CUDA so repeated H2D uploads use the
// driver's pinned fast path instead of implicit pageable staging buffers.
//
// This is most valuable for long-lived resident parts that are uploaded many
// times, such as synthetic benchmark inputs or hot shards kept in host memory.
// The registration is tied to m->storage, so callers must not free or rebuild
// the allocation without going through clear()/allocate()/unpin().
__host__ __forceinline__ int pin(compressed * __restrict__ m) {
    const std::size_t ptr_count = (std::size_t) major_dim(m) + 1u;
    const std::size_t major_bytes = ptr_count * sizeof(types::ptr_t);
    const std::size_t minor_offset = ((major_bytes + alignof(types::idx_t) - 1u) / alignof(types::idx_t)) * alignof(types::idx_t);
    const std::size_t val_offset = ((minor_offset + (std::size_t) m->nnz * sizeof(types::idx_t) + alignof(real::storage_t) - 1u) / alignof(real::storage_t)) * alignof(real::storage_t);
    const std::size_t total_bytes = val_offset + (std::size_t) m->nnz * sizeof(real::storage_t);
    cudaError_t err = cudaSuccess;

    if (m->storage == 0 || total_bytes == 0) return 1;
    if ((m->flags & compressed_host_registered) != 0u) return 1;
    err = cudaHostRegister(m->storage, total_bytes, cudaHostRegisterPortable);
    if (err == cudaSuccess) {
        m->flags |= compressed_host_registered;
        return 1;
    }
    if (err == cudaErrorHostMemoryAlreadyRegistered) {
        cudaGetLastError();
        m->flags |= compressed_host_registered;
        return 1;
    }
    cudaGetLastError();
    return 0;
}

__host__ __forceinline__ void unpin(compressed * __restrict__ m) {
    if ((m->flags & compressed_host_registered) == 0u || m->storage == 0) return;
    cudaHostUnregister(m->storage);
    m->flags &= ~compressed_host_registered;
}

__host__ __device__ __forceinline__ int host_registered(const compressed * __restrict__ m) {
    return (m->flags & compressed_host_registered) != 0u;
}

} // namespace sparse
} // namespace cellshard
