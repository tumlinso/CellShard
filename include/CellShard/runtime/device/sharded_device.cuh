#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../host/sharded_host.cuh"

namespace cellshard {
namespace device {

// Device residency state is deliberately blunt:
// - one record per logical host partition
// - one device-side descriptor pointer
// - up to four device payload pointers
//
// These records do not alias host memory. Any upload/stage path below allocates
// fresh device memory and copies payload into it.
template<typename MatrixT>
struct alignas(16) partition_record {
    void *allocation;
    std::size_t allocation_bytes;
    void *storage;
    void *view;
    void *a0;
    void *a1;
    void *a2;
    void *a3;
    unsigned long group_begin;
    unsigned long group_end;
    int device_id;
};

struct cached_allocation {
    void *ptr;
    std::size_t bytes;
    int device_id;
};

template<typename MatrixT>
struct sharded_device {
    unsigned long capacity;
    partition_record<MatrixT> *parts;
    unsigned long cache_count;
    unsigned long cache_capacity;
    cached_allocation *cache;
};

template<typename MatrixT>
inline cudaError_t release(partition_record<MatrixT> *record);

template<typename MatrixT>
inline cudaError_t release_partition(sharded_device<MatrixT> *state, unsigned long partId);

template<typename MatrixT>
inline cudaError_t upload_partition_async(sharded_device<MatrixT> *state, const ::cellshard::sharded<MatrixT> *view, unsigned long partId, int deviceId, cudaStream_t stream);

struct alignas(16) dense_view {
    unsigned int rows;
    unsigned int cols;
    __half *val;
};

struct alignas(16) compressed_view {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned int axis;
    unsigned int *majorPtr;
    unsigned int *minorIdx;
    __half *val;
};

struct alignas(16) blocked_ell_view {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned int block_size;
    unsigned int ell_cols;
    unsigned int *blockColIdx;
    __half *val;
};

struct alignas(16) sliced_ell_view {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned int slice_count;
    unsigned int slice_rows;
    unsigned int *slice_row_offsets;
    unsigned int *slice_widths;
    unsigned int *slice_slot_offsets;
    unsigned int *col_idx;
    __half *val;
};

struct alignas(16) coo_view {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned int *rowIdx;
    unsigned int *colIdx;
    __half *val;
};

struct alignas(16) dia_view {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned int num_diagonals;
    int *offsets;
    __half *val;
};

template<typename MatrixT>
__host__ __forceinline__ void init(sharded_device<MatrixT> *state) {
    state->capacity = 0;
    state->parts = 0;
    state->cache_count = 0;
    state->cache_capacity = 0;
    state->cache = 0;
}

template<typename MatrixT>
__host__ __forceinline__ void clear(sharded_device<MatrixT> *state) {
    unsigned long i = 0;
    if (state->parts != 0) {
        for (i = 0; i < state->capacity; ++i) {
            if (state->parts[i].view != 0) release_partition(state, i);
        }
    }
    if (state->cache != 0) {
        for (i = 0; i < state->cache_count; ++i) {
            if (state->cache[i].ptr != 0) {
                cudaSetDevice(state->cache[i].device_id >= 0 ? state->cache[i].device_id : 0);
                cudaFree(state->cache[i].ptr);
            }
        }
    }
    std::free(state->parts);
    std::free(state->cache);
    state->capacity = 0;
    state->parts = 0;
    state->cache_count = 0;
    state->cache_capacity = 0;
    state->cache = 0;
}

template<typename MatrixT>
__host__ __forceinline__ int reserve(sharded_device<MatrixT> *state, unsigned long capacity) {
    partition_record<MatrixT> *records = 0;
    unsigned long i = 0;

    if (capacity <= state->capacity) return 1;
    // Grow the per-part residency table on host. This copies metadata only.
    // Device allocations referenced by the old records remain owned by those
    // records and are transferred bitwise into the new table below.
    records = (partition_record<MatrixT> *) std::malloc((std::size_t) capacity * sizeof(partition_record<MatrixT>));
    if (records == 0) return 0;
    std::memset(records, 0, (std::size_t) capacity * sizeof(partition_record<MatrixT>));
    for (i = 0; i < capacity; ++i) records[i].device_id = -1;
    for (i = 0; i < state->capacity; ++i) records[i] = state->parts[i];
    std::free(state->parts);
    state->parts = records;
    state->capacity = capacity;
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ void zero_record(partition_record<MatrixT> *record) {
    record->allocation = 0;
    record->allocation_bytes = 0;
    record->storage = 0;
    record->view = 0;
    record->a0 = 0;
    record->a1 = 0;
    record->a2 = 0;
    record->a3 = 0;
    record->group_begin = 0;
    record->group_end = 0;
    record->device_id = -1;
}

// Keep all device-side allocations aligned and packed inside one storage block
// per uploaded partition. This reduces cudaMalloc/cudaFree pressure dramatically on
// V100-era drivers where allocator churn can dominate small-part staging.
__host__ __device__ __forceinline__ std::size_t align_up_bytes(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

// Keep a small per-device free list for packed shard allocations. Repeated
// stage/release loops often revisit the same shard sizes, and on V100-era
// drivers the raw cudaMalloc/cudaFree path is expensive enough to show up
// clearly in Nsight Systems.
template<typename MatrixT>
__host__ __forceinline__ void *take_cached_allocation(sharded_device<MatrixT> *state, int deviceId, std::size_t min_bytes, std::size_t *out_bytes) {
    unsigned long best = (unsigned long) -1;
    unsigned long i = 0;

    if (out_bytes != 0) *out_bytes = 0;
    if (state == 0 || state->cache == 0) return 0;
    for (i = 0; i < state->cache_count; ++i) {
        if (state->cache[i].ptr == 0) continue;
        if (state->cache[i].device_id != deviceId) continue;
        if (state->cache[i].bytes < min_bytes) continue;
        if (best == (unsigned long) -1 || state->cache[i].bytes < state->cache[best].bytes) best = i;
    }
    if (best == (unsigned long) -1) return 0;
    {
        void *ptr = state->cache[best].ptr;
        const std::size_t bytes = state->cache[best].bytes;
        state->cache[best] = state->cache[state->cache_count - 1ul];
        --state->cache_count;
        if (out_bytes != 0) *out_bytes = bytes;
        return ptr;
    }
}

template<typename MatrixT>
__host__ __forceinline__ cudaError_t cache_allocation(sharded_device<MatrixT> *state, int deviceId, void *ptr, std::size_t bytes) {
    cached_allocation *grown = 0;
    unsigned long capacity = 0;

    if (ptr == 0) return cudaSuccess;
    if (state == 0 || bytes == 0) return cudaFree(ptr);
    if (state->cache_count >= 16ul) return cudaFree(ptr);
    if (state->cache_count == state->cache_capacity) {
        capacity = state->cache_capacity != 0 ? state->cache_capacity * 2ul : 8ul;
        if (capacity < 16ul) capacity = 16ul;
        grown = (cached_allocation *) std::realloc(state->cache, (std::size_t) capacity * sizeof(cached_allocation));
        if (grown == 0) return cudaFree(ptr);
        state->cache = grown;
        state->cache_capacity = capacity;
    }
    state->cache[state->cache_count].ptr = ptr;
    state->cache[state->cache_count].bytes = bytes;
    state->cache[state->cache_count].device_id = deviceId;
    ++state->cache_count;
    return cudaSuccess;
}

// Upload is not a view conversion. It performs:
// 1. cudaMalloc for the dense payload
// 2. one host->device copy of all values
// 3. cudaMalloc for the device-side descriptor
// 4. one host->device copy of that descriptor
//
// Callers should treat this as a full materialization step.
__host__ __forceinline__ cudaError_t upload(const ::cellshard::dense *src, partition_record< ::cellshard::dense > *record) {
    dense_view host;
    const std::size_t count = (std::size_t) src->rows * (std::size_t) src->cols;
    const std::size_t view_offset = 0;
    const std::size_t val_offset = align_up_bytes(sizeof(dense_view), alignof(__half));
    const std::size_t total_bytes = val_offset + count * sizeof(__half);
    char *storage = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.val = 0;

    if (count != 0) {
        err = cudaMalloc((void **) &storage, total_bytes);
        if (err != cudaSuccess) goto fail;
        host.val = (__half *) (storage + val_offset);
        err = cudaMemcpy(host.val, src->val, count * sizeof(__half), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    } else {
        err = cudaMalloc((void **) &storage, sizeof(dense_view));
        if (err != cudaSuccess) goto fail;
    }
    err = cudaMemcpy(storage + view_offset, &host, sizeof(host), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    record->allocation = storage;
    record->allocation_bytes = total_bytes != 0 ? total_bytes : sizeof(dense_view);
    record->storage = storage;
    record->view = storage + view_offset;
    record->a0 = host.val;
    return cudaSuccess;

fail:
    if (storage != 0) cudaFree(storage);
    zero_record(record);
    return err;
}

// Async upload only makes the memcpy operations asynchronous relative to the
// supplied stream. The function still performs host-side allocation work before
// returning, and the source still lives in host memory.
__host__ __forceinline__ cudaError_t upload_async(const ::cellshard::dense *src,
                                                  partition_record< ::cellshard::dense > *record,
                                                  cudaStream_t stream) {
    dense_view host;
    const std::size_t count = (std::size_t) src->rows * (std::size_t) src->cols;
    const std::size_t view_offset = 0;
    const std::size_t val_offset = align_up_bytes(sizeof(dense_view), alignof(__half));
    const std::size_t total_bytes = val_offset + count * sizeof(__half);
    char *storage = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.val = 0;

    if (count != 0) {
        err = cudaMalloc((void **) &storage, total_bytes);
        if (err != cudaSuccess) goto fail;
        host.val = (__half *) (storage + val_offset);
        err = cudaMemcpyAsync(host.val, src->val, count * sizeof(__half), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto fail;
    } else {
        err = cudaMalloc((void **) &storage, sizeof(dense_view));
        if (err != cudaSuccess) goto fail;
    }
    err = cudaMemcpyAsync(storage + view_offset, &host, sizeof(host), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) goto fail;

    record->allocation = storage;
    record->allocation_bytes = total_bytes != 0 ? total_bytes : sizeof(dense_view);
    record->storage = storage;
    record->view = storage + view_offset;
    record->a0 = host.val;
    return cudaSuccess;

fail:
    if (storage != 0) cudaFree(storage);
    zero_record(record);
    return err;
}

// Compressed upload performs three separate H2D payload copies:
// - major pointer array
// - minor index array
// - value array
//
// Then it copies a small descriptor struct containing those device pointers.
__host__ __forceinline__ cudaError_t upload(const ::cellshard::sparse::compressed *src, partition_record< ::cellshard::sparse::compressed > *record) {
    compressed_view host;
    const std::size_t ptr_count = (std::size_t) sparse::major_dim(src) + 1u;
    const std::size_t view_offset = 0;
    const std::size_t major_offset = align_up_bytes(sizeof(compressed_view), alignof(unsigned int));
    const std::size_t minor_offset = align_up_bytes(major_offset + ptr_count * sizeof(unsigned int), alignof(unsigned int));
    const std::size_t val_offset = align_up_bytes(minor_offset + (std::size_t) src->nnz * sizeof(unsigned int), alignof(__half));
    const std::size_t total_bytes = val_offset + (std::size_t) src->nnz * sizeof(__half);
    char *storage = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.nnz = src->nnz;
    host.axis = src->axis;
    host.majorPtr = 0;
    host.minorIdx = 0;
    host.val = 0;

    err = cudaMalloc((void **) &storage, total_bytes == 0 ? sizeof(compressed_view) : total_bytes);
    if (err != cudaSuccess) goto fail;
    if (ptr_count != 0) {
        host.majorPtr = (unsigned int *) (storage + major_offset);
        err = cudaMemcpy(host.majorPtr, src->majorPtr, ptr_count * sizeof(unsigned int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }

    if (src->nnz != 0) {
        host.minorIdx = (unsigned int *) (storage + minor_offset);
        err = cudaMemcpy(host.minorIdx, src->minorIdx, src->nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
        host.val = (__half *) (storage + val_offset);
        err = cudaMemcpy(host.val, src->val, src->nnz * sizeof(__half), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }
    err = cudaMemcpy(storage + view_offset, &host, sizeof(host), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    record->allocation = storage;
    record->allocation_bytes = total_bytes;
    record->storage = storage;
    record->view = storage + view_offset;
    record->a0 = host.majorPtr;
    record->a1 = host.minorIdx;
    record->a2 = host.val;
    return cudaSuccess;

fail:
    if (storage != 0) cudaFree(storage);
    zero_record(record);
    return err;
}

// Async compressed upload still allocates on the host thread. Only the payload
// transfers and descriptor copy are queued on the supplied stream.
__host__ __forceinline__ cudaError_t upload_async(const ::cellshard::sparse::compressed *src,
                                                  partition_record< ::cellshard::sparse::compressed > *record,
                                                  cudaStream_t stream) {
    compressed_view host;
    const std::size_t ptr_count = (std::size_t) sparse::major_dim(src) + 1u;
    const std::size_t view_offset = 0;
    const std::size_t major_offset = align_up_bytes(sizeof(compressed_view), alignof(unsigned int));
    const std::size_t minor_offset = align_up_bytes(major_offset + ptr_count * sizeof(unsigned int), alignof(unsigned int));
    const std::size_t val_offset = align_up_bytes(minor_offset + (std::size_t) src->nnz * sizeof(unsigned int), alignof(__half));
    const std::size_t total_bytes = val_offset + (std::size_t) src->nnz * sizeof(__half);
    char *storage = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.nnz = src->nnz;
    host.axis = src->axis;
    host.majorPtr = 0;
    host.minorIdx = 0;
    host.val = 0;

    err = cudaMalloc((void **) &storage, total_bytes == 0 ? sizeof(compressed_view) : total_bytes);
    if (err != cudaSuccess) goto fail;
    if (ptr_count != 0) {
        host.majorPtr = (unsigned int *) (storage + major_offset);
        err = cudaMemcpyAsync(host.majorPtr, src->majorPtr, ptr_count * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto fail;
    }

    if (src->nnz != 0) {
        host.minorIdx = (unsigned int *) (storage + minor_offset);
        err = cudaMemcpyAsync(host.minorIdx, src->minorIdx, src->nnz * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto fail;
        host.val = (__half *) (storage + val_offset);
        err = cudaMemcpyAsync(host.val, src->val, src->nnz * sizeof(__half), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto fail;
    }
    err = cudaMemcpyAsync(storage + view_offset, &host, sizeof(host), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) goto fail;

    record->allocation = storage;
    record->allocation_bytes = total_bytes;
    record->storage = storage;
    record->view = storage + view_offset;
    record->a0 = host.majorPtr;
    record->a1 = host.minorIdx;
    record->a2 = host.val;
    return cudaSuccess;

fail:
    if (storage != 0) cudaFree(storage);
    zero_record(record);
    return err;
}

__host__ __forceinline__ cudaError_t upload(const ::cellshard::sparse::blocked_ell *src, partition_record< ::cellshard::sparse::blocked_ell > *record) {
    blocked_ell_view host;
    const std::size_t row_blocks = (std::size_t) sparse::row_block_count(src);
    const std::size_t ell_width = (std::size_t) sparse::ell_width_blocks(src);
    const std::size_t idx_count = row_blocks * ell_width;
    const std::size_t view_offset = 0;
    const std::size_t idx_offset = align_up_bytes(sizeof(blocked_ell_view), alignof(unsigned int));
    const std::size_t val_offset = align_up_bytes(idx_offset + idx_count * sizeof(unsigned int), alignof(__half));
    const std::size_t total_bytes = val_offset + (std::size_t) src->rows * (std::size_t) src->ell_cols * sizeof(__half);
    char *storage = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.nnz = src->nnz;
    host.block_size = src->block_size;
    host.ell_cols = src->ell_cols;
    host.blockColIdx = 0;
    host.val = 0;

    err = cudaMalloc((void **) &storage, total_bytes == 0 ? sizeof(blocked_ell_view) : total_bytes);
    if (err != cudaSuccess) goto fail;
    if (idx_count != 0u) {
        host.blockColIdx = (unsigned int *) (storage + idx_offset);
        err = cudaMemcpy(host.blockColIdx, src->blockColIdx, idx_count * sizeof(unsigned int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }
    if (src->rows != 0u && src->ell_cols != 0u) {
        host.val = (__half *) (storage + val_offset);
        err = cudaMemcpy(host.val, src->val, (std::size_t) src->rows * (std::size_t) src->ell_cols * sizeof(__half), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }
    err = cudaMemcpy(storage + view_offset, &host, sizeof(host), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    record->allocation = storage;
    record->allocation_bytes = total_bytes;
    record->storage = storage;
    record->view = storage + view_offset;
    record->a0 = host.blockColIdx;
    record->a1 = host.val;
    return cudaSuccess;

fail:
    if (storage != 0) cudaFree(storage);
    zero_record(record);
    return err;
}

__host__ __forceinline__ cudaError_t upload_async(const ::cellshard::sparse::blocked_ell *src,
                                                  partition_record< ::cellshard::sparse::blocked_ell > *record,
                                                  cudaStream_t stream) {
    blocked_ell_view host;
    const std::size_t row_blocks = (std::size_t) sparse::row_block_count(src);
    const std::size_t ell_width = (std::size_t) sparse::ell_width_blocks(src);
    const std::size_t idx_count = row_blocks * ell_width;
    const std::size_t view_offset = 0;
    const std::size_t idx_offset = align_up_bytes(sizeof(blocked_ell_view), alignof(unsigned int));
    const std::size_t val_offset = align_up_bytes(idx_offset + idx_count * sizeof(unsigned int), alignof(__half));
    const std::size_t total_bytes = val_offset + (std::size_t) src->rows * (std::size_t) src->ell_cols * sizeof(__half);
    char *storage = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.nnz = src->nnz;
    host.block_size = src->block_size;
    host.ell_cols = src->ell_cols;
    host.blockColIdx = 0;
    host.val = 0;

    err = cudaMalloc((void **) &storage, total_bytes == 0 ? sizeof(blocked_ell_view) : total_bytes);
    if (err != cudaSuccess) goto fail;
    if (idx_count != 0u) {
        host.blockColIdx = (unsigned int *) (storage + idx_offset);
        err = cudaMemcpyAsync(host.blockColIdx, src->blockColIdx, idx_count * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto fail;
    }
    if (src->rows != 0u && src->ell_cols != 0u) {
        host.val = (__half *) (storage + val_offset);
        err = cudaMemcpyAsync(host.val, src->val, (std::size_t) src->rows * (std::size_t) src->ell_cols * sizeof(__half), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto fail;
    }
    err = cudaMemcpyAsync(storage + view_offset, &host, sizeof(host), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) goto fail;

    record->allocation = storage;
    record->allocation_bytes = total_bytes;
    record->storage = storage;
    record->view = storage + view_offset;
    record->a0 = host.blockColIdx;
    record->a1 = host.val;
    return cudaSuccess;

fail:
    if (storage != 0) cudaFree(storage);
    zero_record(record);
    return err;
}

__host__ __forceinline__ cudaError_t upload(const ::cellshard::sparse::sliced_ell *src,
                                            partition_record< ::cellshard::sparse::sliced_ell > *record) {
    sliced_ell_view host;
    const std::size_t slice_offsets_bytes = src != 0 ? (std::size_t) (src->slice_count + 1u) * sizeof(unsigned int) : 0u;
    const std::size_t widths_offset = align_up_bytes(slice_offsets_bytes, alignof(unsigned int));
    const std::size_t widths_bytes = src != 0 ? (std::size_t) src->slice_count * sizeof(unsigned int) : 0u;
    const std::size_t slot_offsets_offset = align_up_bytes(widths_offset + widths_bytes, alignof(unsigned int));
    const std::size_t slot_offsets_bytes = src != 0 ? (std::size_t) src->slice_count * sizeof(unsigned int) : 0u;
    const std::size_t total_slots = src != 0 ? (std::size_t) sparse::total_slots(src) : 0u;
    const std::size_t col_offset = align_up_bytes(slot_offsets_offset + slot_offsets_bytes, alignof(unsigned int));
    const std::size_t col_bytes = total_slots * sizeof(unsigned int);
    const std::size_t val_offset = align_up_bytes(col_offset + col_bytes, alignof(__half));
    const std::size_t val_bytes = total_slots * sizeof(__half);
    const std::size_t view_offset = 0u;
    const std::size_t payload_offset = align_up_bytes(sizeof(sliced_ell_view), alignof(unsigned int));
    const std::size_t payload_bytes = val_offset + val_bytes;
    const std::size_t total_bytes = payload_offset + payload_bytes;
    char *storage = 0;
    char *payload = 0;
    unsigned int *slot_offsets_host = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src != 0 ? src->rows : 0u;
    host.cols = src != 0 ? src->cols : 0u;
    host.nnz = src != 0 ? src->nnz : 0u;
    host.slice_count = src != 0 ? src->slice_count : 0u;
    host.slice_rows = 0u;
    host.slice_row_offsets = 0;
    host.slice_widths = 0;
    host.slice_slot_offsets = 0;
    host.col_idx = 0;
    host.val = 0;

    err = cudaMalloc((void **) &storage, total_bytes == 0 ? sizeof(sliced_ell_view) : total_bytes);
    if (err != cudaSuccess) goto fail;
    payload = storage + payload_offset;

    if (payload_bytes != 0u) {
        if (slice_offsets_bytes != 0u) {
            err = cudaMemcpy(payload, src->slice_row_offsets, slice_offsets_bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) goto fail;
        }
        if (widths_bytes != 0u) {
            err = cudaMemcpy(payload + widths_offset, src->slice_widths, widths_bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) goto fail;
        }
        slot_offsets_host = src->slice_count != 0u ? (unsigned int *) std::malloc(slot_offsets_bytes) : 0;
        if (src->slice_count != 0u && slot_offsets_host == 0) {
            err = cudaErrorMemoryAllocation;
            goto fail;
        }
        if (src->slice_count != 0u) {
            unsigned int running = 0u;
            unsigned int slice = 0u;
            for (slice = 0u; slice < src->slice_count; ++slice) {
                slot_offsets_host[slice] = running;
                running += (src->slice_row_offsets[slice + 1u] - src->slice_row_offsets[slice]) * src->slice_widths[slice];
            }
            err = cudaMemcpy(payload + slot_offsets_offset, slot_offsets_host, slot_offsets_bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) goto fail;
            host.slice_rows = sparse::uniform_slice_rows(src);
            host.slice_slot_offsets = (unsigned int *) (payload + slot_offsets_offset);
        }
        if (col_bytes != 0u) {
            err = cudaMemcpy(payload + col_offset, src->col_idx, col_bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) goto fail;
        }
        if (val_bytes != 0u) {
            err = cudaMemcpy(payload + val_offset, src->val, val_bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) goto fail;
        }
        host.slice_row_offsets = src->slice_row_offsets != 0 ? (unsigned int *) payload : 0;
        host.slice_widths = src->slice_widths != 0 ? (unsigned int *) (payload + widths_offset) : 0;
        host.col_idx = src->col_idx != 0 ? (unsigned int *) (payload + col_offset) : 0;
        host.val = src->val != 0 ? (__half *) (payload + val_offset) : 0;
    }

    err = cudaMemcpy(storage + view_offset, &host, sizeof(host), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    record->allocation = storage;
    record->allocation_bytes = total_bytes;
    record->storage = payload;
    record->view = storage + view_offset;
    record->a0 = host.slice_row_offsets;
    record->a1 = host.slice_widths;
    record->a2 = host.col_idx;
    record->a3 = host.val;
    std::free(slot_offsets_host);
    return cudaSuccess;

fail:
    std::free(slot_offsets_host);
    if (storage != 0) cudaFree(storage);
    zero_record(record);
    return err;
}

__host__ __forceinline__ cudaError_t upload_async(const ::cellshard::sparse::sliced_ell *src,
                                                  partition_record< ::cellshard::sparse::sliced_ell > *record,
                                                  cudaStream_t stream) {
    sliced_ell_view host;
    const std::size_t slice_offsets_bytes = src != 0 ? (std::size_t) (src->slice_count + 1u) * sizeof(unsigned int) : 0u;
    const std::size_t widths_offset = align_up_bytes(slice_offsets_bytes, alignof(unsigned int));
    const std::size_t widths_bytes = src != 0 ? (std::size_t) src->slice_count * sizeof(unsigned int) : 0u;
    const std::size_t slot_offsets_offset = align_up_bytes(widths_offset + widths_bytes, alignof(unsigned int));
    const std::size_t slot_offsets_bytes = src != 0 ? (std::size_t) src->slice_count * sizeof(unsigned int) : 0u;
    const std::size_t total_slots = src != 0 ? (std::size_t) sparse::total_slots(src) : 0u;
    const std::size_t col_offset = align_up_bytes(slot_offsets_offset + slot_offsets_bytes, alignof(unsigned int));
    const std::size_t col_bytes = total_slots * sizeof(unsigned int);
    const std::size_t val_offset = align_up_bytes(col_offset + col_bytes, alignof(__half));
    const std::size_t val_bytes = total_slots * sizeof(__half);
    const std::size_t view_offset = 0u;
    const std::size_t payload_offset = align_up_bytes(sizeof(sliced_ell_view), alignof(unsigned int));
    const std::size_t payload_bytes = val_offset + val_bytes;
    const std::size_t total_bytes = payload_offset + payload_bytes;
    char *storage = 0;
    char *payload = 0;
    unsigned int *slot_offsets_host = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src != 0 ? src->rows : 0u;
    host.cols = src != 0 ? src->cols : 0u;
    host.nnz = src != 0 ? src->nnz : 0u;
    host.slice_count = src != 0 ? src->slice_count : 0u;
    host.slice_rows = 0u;
    host.slice_row_offsets = 0;
    host.slice_widths = 0;
    host.slice_slot_offsets = 0;
    host.col_idx = 0;
    host.val = 0;

    err = cudaMalloc((void **) &storage, total_bytes == 0 ? sizeof(sliced_ell_view) : total_bytes);
    if (err != cudaSuccess) goto fail;
    payload = storage + payload_offset;

    if (payload_bytes != 0u) {
        if (slice_offsets_bytes != 0u) {
            err = cudaMemcpyAsync(payload, src->slice_row_offsets, slice_offsets_bytes, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) goto fail;
        }
        if (widths_bytes != 0u) {
            err = cudaMemcpyAsync(payload + widths_offset, src->slice_widths, widths_bytes, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) goto fail;
        }
        slot_offsets_host = src->slice_count != 0u ? (unsigned int *) std::malloc(slot_offsets_bytes) : 0;
        if (src->slice_count != 0u && slot_offsets_host == 0) {
            err = cudaErrorMemoryAllocation;
            goto fail;
        }
        if (src->slice_count != 0u) {
            unsigned int running = 0u;
            unsigned int slice = 0u;
            for (slice = 0u; slice < src->slice_count; ++slice) {
                slot_offsets_host[slice] = running;
                running += (src->slice_row_offsets[slice + 1u] - src->slice_row_offsets[slice]) * src->slice_widths[slice];
            }
            err = cudaMemcpy(payload + slot_offsets_offset, slot_offsets_host, slot_offsets_bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) goto fail;
            host.slice_rows = sparse::uniform_slice_rows(src);
            host.slice_slot_offsets = (unsigned int *) (payload + slot_offsets_offset);
        }
        if (col_bytes != 0u) {
            err = cudaMemcpyAsync(payload + col_offset, src->col_idx, col_bytes, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) goto fail;
        }
        if (val_bytes != 0u) {
            err = cudaMemcpyAsync(payload + val_offset, src->val, val_bytes, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) goto fail;
        }
        host.slice_row_offsets = src->slice_row_offsets != 0 ? (unsigned int *) payload : 0;
        host.slice_widths = src->slice_widths != 0 ? (unsigned int *) (payload + widths_offset) : 0;
        host.col_idx = src->col_idx != 0 ? (unsigned int *) (payload + col_offset) : 0;
        host.val = src->val != 0 ? (__half *) (payload + val_offset) : 0;
    }

    err = cudaMemcpyAsync(storage + view_offset, &host, sizeof(host), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) goto fail;

    record->allocation = storage;
    record->allocation_bytes = total_bytes;
    record->storage = payload;
    record->view = storage + view_offset;
    record->a0 = host.slice_row_offsets;
    record->a1 = host.slice_widths;
    record->a2 = host.col_idx;
    record->a3 = host.val;
    std::free(slot_offsets_host);
    return cudaSuccess;

fail:
    std::free(slot_offsets_host);
    if (storage != 0) cudaFree(storage);
    zero_record(record);
    return err;
}

// COO upload is another full host->device materialization:
// row index copy, column index copy, value copy, then descriptor copy.
__host__ __forceinline__ cudaError_t upload(const ::cellshard::sparse::coo *src, partition_record< ::cellshard::sparse::coo > *record) {
    coo_view host;
    const std::size_t view_offset = 0;
    const std::size_t row_offset = align_up_bytes(sizeof(coo_view), alignof(unsigned int));
    const std::size_t col_offset = align_up_bytes(row_offset + (std::size_t) src->nnz * sizeof(unsigned int), alignof(unsigned int));
    const std::size_t val_offset = align_up_bytes(col_offset + (std::size_t) src->nnz * sizeof(unsigned int), alignof(__half));
    const std::size_t total_bytes = val_offset + (std::size_t) src->nnz * sizeof(__half);
    char *storage = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.nnz = src->nnz;
    host.rowIdx = 0;
    host.colIdx = 0;
    host.val = 0;

    err = cudaMalloc((void **) &storage, total_bytes == 0 ? sizeof(coo_view) : total_bytes);
    if (err != cudaSuccess) goto fail;
    if (src->nnz != 0) {
        host.rowIdx = (unsigned int *) (storage + row_offset);
        err = cudaMemcpy(host.rowIdx, src->rowIdx, src->nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
        host.colIdx = (unsigned int *) (storage + col_offset);
        err = cudaMemcpy(host.colIdx, src->colIdx, src->nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
        host.val = (__half *) (storage + val_offset);
        err = cudaMemcpy(host.val, src->val, src->nnz * sizeof(__half), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }
    err = cudaMemcpy(storage + view_offset, &host, sizeof(host), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    record->allocation = storage;
    record->allocation_bytes = total_bytes;
    record->storage = storage;
    record->view = storage + view_offset;
    record->a0 = host.rowIdx;
    record->a1 = host.colIdx;
    record->a2 = host.val;
    return cudaSuccess;

fail:
    if (storage != 0) cudaFree(storage);
    zero_record(record);
    return err;
}

// Async COO upload has the same copy volume as upload(); the only difference is
// stream-ordered memcpy submission.
__host__ __forceinline__ cudaError_t upload_async(const ::cellshard::sparse::coo *src,
                                                  partition_record< ::cellshard::sparse::coo > *record,
                                                  cudaStream_t stream) {
    coo_view host;
    const std::size_t view_offset = 0;
    const std::size_t row_offset = align_up_bytes(sizeof(coo_view), alignof(unsigned int));
    const std::size_t col_offset = align_up_bytes(row_offset + (std::size_t) src->nnz * sizeof(unsigned int), alignof(unsigned int));
    const std::size_t val_offset = align_up_bytes(col_offset + (std::size_t) src->nnz * sizeof(unsigned int), alignof(__half));
    const std::size_t total_bytes = val_offset + (std::size_t) src->nnz * sizeof(__half);
    char *storage = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.nnz = src->nnz;
    host.rowIdx = 0;
    host.colIdx = 0;
    host.val = 0;

    err = cudaMalloc((void **) &storage, total_bytes == 0 ? sizeof(coo_view) : total_bytes);
    if (err != cudaSuccess) goto fail;
    if (src->nnz != 0) {
        host.rowIdx = (unsigned int *) (storage + row_offset);
        err = cudaMemcpyAsync(host.rowIdx, src->rowIdx, src->nnz * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto fail;
        host.colIdx = (unsigned int *) (storage + col_offset);
        err = cudaMemcpyAsync(host.colIdx, src->colIdx, src->nnz * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto fail;
        host.val = (__half *) (storage + val_offset);
        err = cudaMemcpyAsync(host.val, src->val, src->nnz * sizeof(__half), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto fail;
    }
    err = cudaMemcpyAsync(storage + view_offset, &host, sizeof(host), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) goto fail;

    record->allocation = storage;
    record->allocation_bytes = total_bytes;
    record->storage = storage;
    record->view = storage + view_offset;
    record->a0 = host.rowIdx;
    record->a1 = host.colIdx;
    record->a2 = host.val;
    return cudaSuccess;

fail:
    if (storage != 0) cudaFree(storage);
    zero_record(record);
    return err;
}

// DIA upload copies offsets and values separately, then publishes one device
// descriptor that points at those allocations.
__host__ __forceinline__ cudaError_t upload(const ::cellshard::sparse::dia *src, partition_record< ::cellshard::sparse::dia > *record) {
    dia_view host;
    const std::size_t view_offset = 0;
    const std::size_t offsets_offset = align_up_bytes(sizeof(dia_view), alignof(int));
    const std::size_t val_offset = align_up_bytes(offsets_offset + (std::size_t) src->num_diagonals * sizeof(int), alignof(__half));
    const std::size_t total_bytes = val_offset + (std::size_t) src->nnz * sizeof(__half);
    char *storage = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.nnz = src->nnz;
    host.num_diagonals = src->num_diagonals;
    host.offsets = 0;
    host.val = 0;

    err = cudaMalloc((void **) &storage, total_bytes == 0 ? sizeof(dia_view) : total_bytes);
    if (err != cudaSuccess) goto fail;
    if (src->num_diagonals != 0) {
        host.offsets = (int *) (storage + offsets_offset);
        err = cudaMemcpy(host.offsets, src->offsets, src->num_diagonals * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }
    if (src->nnz != 0) {
        host.val = (__half *) (storage + val_offset);
        err = cudaMemcpy(host.val, src->val, src->nnz * sizeof(__half), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }
    err = cudaMemcpy(storage + view_offset, &host, sizeof(host), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    record->allocation = storage;
    record->allocation_bytes = total_bytes;
    record->storage = storage;
    record->view = storage + view_offset;
    record->a0 = host.offsets;
    record->a1 = host.val;
    return cudaSuccess;

fail:
    if (storage != 0) cudaFree(storage);
    zero_record(record);
    return err;
}

// Async DIA upload preserves the same allocation/copy structure as the sync
// path; it only changes when the H2D copies retire.
__host__ __forceinline__ cudaError_t upload_async(const ::cellshard::sparse::dia *src,
                                                  partition_record< ::cellshard::sparse::dia > *record,
                                                  cudaStream_t stream) {
    dia_view host;
    const std::size_t view_offset = 0;
    const std::size_t offsets_offset = align_up_bytes(sizeof(dia_view), alignof(int));
    const std::size_t val_offset = align_up_bytes(offsets_offset + (std::size_t) src->num_diagonals * sizeof(int), alignof(__half));
    const std::size_t total_bytes = val_offset + (std::size_t) src->nnz * sizeof(__half);
    char *storage = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.nnz = src->nnz;
    host.num_diagonals = src->num_diagonals;
    host.offsets = 0;
    host.val = 0;

    err = cudaMalloc((void **) &storage, total_bytes == 0 ? sizeof(dia_view) : total_bytes);
    if (err != cudaSuccess) goto fail;
    if (src->num_diagonals != 0) {
        host.offsets = (int *) (storage + offsets_offset);
        err = cudaMemcpyAsync(host.offsets, src->offsets, src->num_diagonals * sizeof(int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto fail;
    }
    if (src->nnz != 0) {
        host.val = (__half *) (storage + val_offset);
        err = cudaMemcpyAsync(host.val, src->val, src->nnz * sizeof(__half), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto fail;
    }
    err = cudaMemcpyAsync(storage + view_offset, &host, sizeof(host), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) goto fail;

    record->allocation = storage;
    record->allocation_bytes = total_bytes;
    record->storage = storage;
    record->view = storage + view_offset;
    record->a0 = host.offsets;
    record->a1 = host.val;
    return cudaSuccess;

fail:
    if (storage != 0) cudaFree(storage);
    zero_record(record);
    return err;
}

template<typename MatrixT>
__host__ __forceinline__ cudaError_t release(partition_record<MatrixT> *record) {
    cudaError_t err = cudaSuccess;

    if (record->allocation != 0) {
        err = cudaFree(record->allocation);
        if (err != cudaSuccess) return err;
        zero_record(record);
        return cudaSuccess;
    }
    if (record->a2 != 0) {
        err = cudaFree(record->a2);
        if (err != cudaSuccess) return err;
    }
    if (record->a3 != 0) {
        err = cudaFree(record->a3);
        if (err != cudaSuccess) return err;
    }
    if (record->a1 != 0) {
        err = cudaFree(record->a1);
        if (err != cudaSuccess) return err;
    }
    if (record->a0 != 0) {
        err = cudaFree(record->a0);
        if (err != cudaSuccess) return err;
    }
    if (record->view != 0) {
        err = cudaFree(record->view);
        if (err != cudaSuccess) return err;
    }
    zero_record(record);
    return cudaSuccess;
}

template<typename MatrixT>
__host__ __forceinline__ cudaError_t upload_partition_current_device(sharded_device<MatrixT> *state,
                                                                     const ::cellshard::sharded<MatrixT> *view,
                                                                     unsigned long partId,
                                                                     int deviceId) {
    cudaError_t err = cudaSuccess;

    // The caller already selected the destination device. Keep that selection
    // stable across as many part uploads as possible so shard uploads only pay
    // one cudaSetDevice() in the common case.
    if (partId >= view->num_partitions || partId >= state->capacity || view->parts[partId] == 0) return cudaErrorInvalidValue;
    if (state->parts[partId].view != 0) {
        if (state->parts[partId].device_id == deviceId) return cudaSuccess;
        err = release_partition(state, partId);
        if (err != cudaSuccess) return err;
        err = cudaSetDevice(deviceId);
        if (err != cudaSuccess) return err;
    }
    err = upload(view->parts[partId], &state->parts[partId]);
    if (err != cudaSuccess) return err;
    state->parts[partId].allocation = state->parts[partId].storage;
    state->parts[partId].group_begin = partId;
    state->parts[partId].group_end = partId + 1;
    state->parts[partId].device_id = deviceId;
    return cudaSuccess;
}

template<typename MatrixT>
__host__ __forceinline__ cudaError_t upload_partition_current_device_async(sharded_device<MatrixT> *state,
                                                                           const ::cellshard::sharded<MatrixT> *view,
                                                                           unsigned long partId,
                                                                           int deviceId,
                                                                           cudaStream_t stream) {
    cudaError_t err = cudaSuccess;

    // Async here means stream-ordered copies after the caller has already
    // selected the destination device. Host-side decision-making stays outside
    // this helper so whole-shard uploads can queue densely on one stream.
    if (partId >= view->num_partitions || partId >= state->capacity || view->parts[partId] == 0) return cudaErrorInvalidValue;
    if (state->parts[partId].view != 0) {
        if (state->parts[partId].device_id == deviceId) return cudaSuccess;
        err = release_partition(state, partId);
        if (err != cudaSuccess) return err;
        err = cudaSetDevice(deviceId);
        if (err != cudaSuccess) return err;
    }
    err = upload_async(view->parts[partId], &state->parts[partId], stream);
    if (err != cudaSuccess) return err;
    state->parts[partId].allocation = state->parts[partId].storage;
    state->parts[partId].group_begin = partId;
    state->parts[partId].group_end = partId + 1;
    state->parts[partId].device_id = deviceId;
    return cudaSuccess;
}

// Byte estimates below are the resident device footprint after upload. They are
// not disk bytes and not host bytes. Use them when balancing shard ownership
// across GPUs.
__host__ __forceinline__ std::size_t device_partition_bytes(const ::cellshard::sharded< ::cellshard::dense > *view, unsigned long partId) {
    const std::size_t val_offset = align_up_bytes(sizeof(dense_view), alignof(__half));
    return val_offset + (std::size_t) view->partition_nnz[partId] * sizeof(__half);
}

__host__ __forceinline__ std::size_t device_partition_bytes(const ::cellshard::sharded< ::cellshard::sparse::compressed > *view, unsigned long partId) {
    const unsigned long ptr_dim = view->partition_aux[partId] == sparse::compressed_by_col ? view->cols : view->partition_rows[partId];
    const std::size_t major_offset = align_up_bytes(sizeof(compressed_view), alignof(unsigned int));
    const std::size_t minor_offset = align_up_bytes(major_offset + (std::size_t) (ptr_dim + 1) * sizeof(unsigned int), alignof(unsigned int));
    const std::size_t val_offset = align_up_bytes(minor_offset + (std::size_t) view->partition_nnz[partId] * sizeof(unsigned int), alignof(__half));
    return val_offset + (std::size_t) view->partition_nnz[partId] * sizeof(__half);
}

__host__ __forceinline__ std::size_t device_partition_bytes(const ::cellshard::sharded< ::cellshard::sparse::blocked_ell > *view, unsigned long partId) {
    const unsigned long aux = view->partition_aux[partId];
    const types::u32 block_size = sparse::unpack_blocked_ell_block_size(aux);
    const unsigned long ell_width = sparse::unpack_blocked_ell_ell_width(aux);
    const std::size_t idx_offset = align_up_bytes(sizeof(blocked_ell_view), alignof(unsigned int));
    const std::size_t val_offset = align_up_bytes(idx_offset + ((std::size_t) ((view->partition_rows[partId] + block_size - 1u) / block_size) * ell_width * sizeof(unsigned int)), alignof(__half));
    return val_offset + (std::size_t) view->partition_rows[partId] * (std::size_t) (ell_width * block_size) * sizeof(__half);
}

__host__ __forceinline__ std::size_t device_partition_bytes(const ::cellshard::sharded< ::cellshard::sparse::sliced_ell > *view, unsigned long partId) {
    const unsigned long aux = view->partition_aux[partId];
    const std::size_t payload_bytes =
        (std::size_t) sparse::unpack_sliced_ell_slice_count(aux) * sizeof(unsigned int)
        + (std::size_t) (sparse::unpack_sliced_ell_slice_count(aux) + 1u) * sizeof(unsigned int)
        + (std::size_t) sparse::unpack_sliced_ell_slice_count(aux) * sizeof(unsigned int)
        + (std::size_t) sparse::unpack_sliced_ell_total_slots(aux) * sizeof(unsigned int)
        + (std::size_t) sparse::unpack_sliced_ell_total_slots(aux) * sizeof(__half);
    const std::size_t payload_offset = align_up_bytes(sizeof(sliced_ell_view), alignof(unsigned int));
    return payload_offset + payload_bytes;
}

__host__ __forceinline__ std::size_t device_partition_bytes(const ::cellshard::sharded< ::cellshard::sparse::coo > *view, unsigned long partId) {
    const std::size_t row_offset = align_up_bytes(sizeof(coo_view), alignof(unsigned int));
    const std::size_t col_offset = align_up_bytes(row_offset + (std::size_t) view->partition_nnz[partId] * sizeof(unsigned int), alignof(unsigned int));
    const std::size_t val_offset = align_up_bytes(col_offset + (std::size_t) view->partition_nnz[partId] * sizeof(unsigned int), alignof(__half));
    return val_offset + (std::size_t) view->partition_nnz[partId] * sizeof(__half);
}

__host__ __forceinline__ std::size_t device_partition_bytes(const ::cellshard::sharded< ::cellshard::sparse::dia > *view, unsigned long partId) {
    const std::size_t offsets_offset = align_up_bytes(sizeof(dia_view), alignof(int));
    const std::size_t val_offset = align_up_bytes(offsets_offset + (std::size_t) view->partition_aux[partId] * sizeof(int), alignof(__half));
    return val_offset + (std::size_t) view->partition_nnz[partId] * sizeof(__half);
}

template<typename MatrixT>
__host__ __forceinline__ std::size_t device_shard_bytes(const ::cellshard::sharded<MatrixT> *view, unsigned long shardId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    std::size_t total = 0;

    if (shardId >= view->num_shards) return 0;
    begin = ::cellshard::first_partition_in_shard(view, shardId);
    end = ::cellshard::last_partition_in_shard(view, shardId);
    for (i = begin; i < end; ++i) total += device_partition_bytes(view, i);
    return total;
}

template<typename MatrixT>
__host__ __forceinline__ int set_shards_by_device_bytes(::cellshard::sharded<MatrixT> *view, std::size_t max_bytes) {
    std::size_t used = 0;
    std::size_t bytes = 0;
    unsigned long shardCount = 0;
    unsigned long i = 0;

    if (max_bytes == 0) return ::cellshard::set_shards_to_partitions(view);
    if (!::cellshard::reserve_shards(view, view->num_partitions)) return 0;

    view->shard_offsets[0] = 0;
    for (i = 0; i < view->num_partitions; ++i) {
        bytes = device_partition_bytes(view, i);
        if (bytes == 0) continue;
        if (used != 0 && used + bytes > max_bytes) {
            ++shardCount;
            view->shard_offsets[shardCount] = view->partition_offsets[i];
            used = 0;
        }
        used += bytes;
    }
    if (view->num_partitions != 0) {
        ++shardCount;
        view->shard_offsets[shardCount] = view->rows;
    }
    view->num_shards = shardCount;
    ::cellshard::rebuild_shard_parts(view);
    return 1;
}

// upload_partition assumes the host part is already materialized. It does not touch
// disk. The cost here is device allocation plus H2D copy.
template<typename MatrixT>
__host__ __forceinline__ cudaError_t upload_partition(sharded_device<MatrixT> *state, const ::cellshard::sharded<MatrixT> *view, unsigned long partId, int deviceId) {
    cudaError_t err = cudaSuccess;

    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) return err;
    return upload_partition_current_device(state, view, partId, deviceId);
}

// Async upload_partition still inspects host state synchronously before enqueueing
// copies. It is not a fire-and-forget data loader.
template<typename MatrixT>
__host__ __forceinline__ cudaError_t upload_partition_async(sharded_device<MatrixT> *state,
                                                            const ::cellshard::sharded<MatrixT> *view,
                                                            unsigned long partId,
                                                            int deviceId,
                                                            cudaStream_t stream) {
    cudaError_t err = cudaSuccess;

    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) return err;
    return upload_partition_current_device_async(state, view, partId, deviceId, stream);
}

template<typename MatrixT>
__host__ __forceinline__ cudaError_t release_partition(sharded_device<MatrixT> *state, unsigned long partId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long owner = 0;
    unsigned long i = 0;
    cudaError_t err = cudaSuccess;

    if (partId >= state->capacity || state->parts[partId].view == 0) return cudaSuccess;
    // Release is cheap only when it can hand the packed allocation back to the
    // small cache below. A cold cudaFree path is still visible in timelines.
    err = cudaSetDevice(state->parts[partId].device_id >= 0 ? state->parts[partId].device_id : 0);
    if (err != cudaSuccess) return err;
    begin = state->parts[partId].group_begin;
    end = state->parts[partId].group_end;
    if (end <= begin || end > state->capacity) {
        return release(&state->parts[partId]);
    }
    owner = begin;
    if (state->parts[owner].allocation != 0) {
        err = cache_allocation(state,
                               state->parts[owner].device_id,
                               state->parts[owner].allocation,
                               state->parts[owner].allocation_bytes);
        if (err != cudaSuccess) return err;
    } else if (state->parts[owner].storage != 0) {
        err = cudaFree(state->parts[owner].storage);
        if (err != cudaSuccess) return err;
    }
    for (i = begin; i < end; ++i) zero_record(&state->parts[i]);
    return cudaSuccess;
}

// Compressed parts already live in one packed host allocation
// [majorPtr][minorIdx][val]. When staging a full shard, mirror that layout on
// device and pack all descriptors into one front-matter block. That changes the
// hot stage path from:
// - 1 cudaMalloc per part
// - 4 H2D copies per part
// to:
// - 1 cudaMalloc per shard
// - 1 descriptor copy per shard
// - 1 payload copy per part
//
// The payload copies remain because the host parts are separate allocations, but
// allocator churn and descriptor traffic drop sharply.
__host__ __forceinline__ std::size_t compressed_payload_bytes(const ::cellshard::sparse::compressed *src) {
    const std::size_t ptr_count = (std::size_t) sparse::major_dim(src) + 1u;
    const std::size_t minor_offset = align_up_bytes(ptr_count * sizeof(unsigned int), alignof(unsigned int));
    const std::size_t val_offset = align_up_bytes(minor_offset + (std::size_t) src->nnz * sizeof(unsigned int), alignof(__half));
    return val_offset + (std::size_t) src->nnz * sizeof(__half);
}

__host__ __forceinline__ cudaError_t upload_compressed_shard_current_device_async(
    sharded_device< ::cellshard::sparse::compressed > *state,
    const ::cellshard::sharded< ::cellshard::sparse::compressed > *view,
    unsigned long shardId,
    int deviceId,
    cudaStream_t stream
) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long part = 0;
    std::size_t descriptor_bytes = 0;
    std::size_t allocation_bytes = 0;
    std::size_t total_bytes = 0;
    char *allocation = 0;
    char *descriptor_base = 0;
    char *payload_base = 0;
    std::vector<compressed_view> host_views;
    cudaError_t err = cudaSuccess;

    if (shardId >= view->num_shards) return cudaErrorInvalidValue;
    begin = ::cellshard::first_partition_in_shard(view, shardId);
    end = ::cellshard::last_partition_in_shard(view, shardId);
    if (begin >= end) return cudaSuccess;

    for (part = begin; part < end; ++part) {
        if (part >= state->capacity || view->parts[part] == 0) return cudaErrorInvalidValue;
        if (state->parts[part].view == 0 || state->parts[part].device_id != deviceId) {
            break;
        }
    }
    if (part == end) return cudaSuccess;

    for (part = begin; part < end; ++part) {
        if (state->parts[part].view != 0) {
            err = release_partition(state, part);
            if (err != cudaSuccess) return err;
        }
    }

    host_views.resize((std::size_t) (end - begin));
    descriptor_bytes = align_up_bytes(host_views.size() * sizeof(compressed_view), alignof(unsigned int));
    total_bytes = descriptor_bytes;
    for (part = begin; part < end; ++part) {
        total_bytes = align_up_bytes(total_bytes, alignof(unsigned int));
        total_bytes += compressed_payload_bytes(view->parts[part]);
    }

    allocation = (char *) take_cached_allocation(state, deviceId, total_bytes, &allocation_bytes);
    if (allocation == 0) {
        err = cudaMalloc((void **) &allocation, total_bytes);
        if (err != cudaSuccess) return err;
        allocation_bytes = total_bytes;
    }

    descriptor_base = allocation;
    payload_base = allocation + descriptor_bytes;
    for (part = begin; part < end; ++part) {
        const ::cellshard::sparse::compressed *src = view->parts[part];
        const std::size_t part_payload_bytes = compressed_payload_bytes(src);
        const std::size_t ptr_count = (std::size_t) sparse::major_dim(src) + 1u;
        const std::size_t minor_offset = align_up_bytes(ptr_count * sizeof(unsigned int), alignof(unsigned int));
        const std::size_t val_offset = align_up_bytes(minor_offset + (std::size_t) src->nnz * sizeof(unsigned int), alignof(__half));
        compressed_view &dst = host_views[(std::size_t) (part - begin)];

        payload_base = (char *) align_up_bytes((std::size_t) payload_base, alignof(unsigned int));
        dst.rows = src->rows;
        dst.cols = src->cols;
        dst.nnz = src->nnz;
        dst.axis = src->axis;
        dst.majorPtr = (unsigned int *) payload_base;
        dst.minorIdx = (unsigned int *) (payload_base + minor_offset);
        dst.val = (__half *) (payload_base + val_offset);

        err = cudaMemcpyAsync(payload_base,
                              src->storage,
                              part_payload_bytes,
                              cudaMemcpyHostToDevice,
                              stream);
        if (err != cudaSuccess) goto fail;
        payload_base += part_payload_bytes;
    }

    err = cudaMemcpyAsync(descriptor_base,
                          host_views.data(),
                          host_views.size() * sizeof(compressed_view),
                          cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) goto fail;

    for (part = begin; part < end; ++part) {
        const unsigned long slot = part - begin;
        state->parts[part].allocation = part == begin ? allocation : 0;
        state->parts[part].allocation_bytes = part == begin ? allocation_bytes : 0;
        state->parts[part].storage = part == begin ? allocation : 0;
        state->parts[part].view = descriptor_base + slot * sizeof(compressed_view);
        state->parts[part].a0 = host_views[(std::size_t) slot].majorPtr;
        state->parts[part].a1 = host_views[(std::size_t) slot].minorIdx;
        state->parts[part].a2 = host_views[(std::size_t) slot].val;
        state->parts[part].group_begin = begin;
        state->parts[part].group_end = end;
        state->parts[part].device_id = deviceId;
    }
    return cudaSuccess;

fail:
    if (allocation != 0) cudaFree(allocation);
    for (part = begin; part < end; ++part) zero_record(&state->parts[part]);
    return err;
}

__host__ __forceinline__ std::size_t blocked_ell_payload_bytes(const ::cellshard::sparse::blocked_ell *src) {
    const std::size_t idx_count = (std::size_t) sparse::row_block_count(src) * (std::size_t) sparse::ell_width_blocks(src);
    const std::size_t val_offset = align_up_bytes(idx_count * sizeof(unsigned int), alignof(__half));
    return val_offset + (std::size_t) src->rows * (std::size_t) src->ell_cols * sizeof(__half);
}

__host__ __forceinline__ cudaError_t upload_blocked_ell_shard_current_device_async(
    sharded_device< ::cellshard::sparse::blocked_ell > *state,
    const ::cellshard::sharded< ::cellshard::sparse::blocked_ell > *view,
    unsigned long shardId,
    int deviceId,
    cudaStream_t stream
) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long part = 0;
    std::size_t descriptor_bytes = 0;
    std::size_t allocation_bytes = 0;
    std::size_t total_bytes = 0;
    char *allocation = 0;
    char *descriptor_base = 0;
    char *payload_base = 0;
    std::vector<blocked_ell_view> host_views;
    cudaError_t err = cudaSuccess;

    if (shardId >= view->num_shards) return cudaErrorInvalidValue;
    begin = ::cellshard::first_partition_in_shard(view, shardId);
    end = ::cellshard::last_partition_in_shard(view, shardId);
    if (begin >= end) return cudaSuccess;

    for (part = begin; part < end; ++part) {
        if (part >= state->capacity || view->parts[part] == 0) return cudaErrorInvalidValue;
        if (state->parts[part].view == 0 || state->parts[part].device_id != deviceId) break;
    }
    if (part == end) return cudaSuccess;

    for (part = begin; part < end; ++part) {
        if (state->parts[part].view != 0) {
            err = release_partition(state, part);
            if (err != cudaSuccess) return err;
        }
    }

    host_views.resize((std::size_t) (end - begin));
    descriptor_bytes = align_up_bytes(host_views.size() * sizeof(blocked_ell_view), alignof(unsigned int));
    total_bytes = descriptor_bytes;
    for (part = begin; part < end; ++part) {
        total_bytes = align_up_bytes(total_bytes, alignof(unsigned int));
        total_bytes += blocked_ell_payload_bytes(view->parts[part]);
    }

    allocation = (char *) take_cached_allocation(state, deviceId, total_bytes, &allocation_bytes);
    if (allocation == 0) {
        err = cudaMalloc((void **) &allocation, total_bytes);
        if (err != cudaSuccess) return err;
        allocation_bytes = total_bytes;
    }

    descriptor_base = allocation;
    payload_base = allocation + descriptor_bytes;
    for (part = begin; part < end; ++part) {
        const ::cellshard::sparse::blocked_ell *src = view->parts[part];
        const std::size_t part_payload_bytes = blocked_ell_payload_bytes(src);
        const std::size_t idx_count = (std::size_t) sparse::row_block_count(src) * (std::size_t) sparse::ell_width_blocks(src);
        const std::size_t val_offset = align_up_bytes(idx_count * sizeof(unsigned int), alignof(__half));
        blocked_ell_view &dst = host_views[(std::size_t) (part - begin)];

        payload_base = (char *) align_up_bytes((std::size_t) payload_base, alignof(unsigned int));
        dst.rows = src->rows;
        dst.cols = src->cols;
        dst.nnz = src->nnz;
        dst.block_size = src->block_size;
        dst.ell_cols = src->ell_cols;
        dst.blockColIdx = (unsigned int *) payload_base;
        dst.val = (__half *) (payload_base + val_offset);

        err = cudaMemcpyAsync(payload_base, src->storage, part_payload_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto fail;
        payload_base += part_payload_bytes;
    }

    err = cudaMemcpyAsync(descriptor_base, host_views.data(), host_views.size() * sizeof(blocked_ell_view), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) goto fail;

    for (part = begin; part < end; ++part) {
        const unsigned long slot = part - begin;
        state->parts[part].allocation = part == begin ? allocation : 0;
        state->parts[part].allocation_bytes = part == begin ? allocation_bytes : 0;
        state->parts[part].storage = part == begin ? allocation : 0;
        state->parts[part].view = descriptor_base + slot * sizeof(blocked_ell_view);
        state->parts[part].a0 = host_views[(std::size_t) slot].blockColIdx;
        state->parts[part].a1 = host_views[(std::size_t) slot].val;
        state->parts[part].group_begin = begin;
        state->parts[part].group_end = end;
        state->parts[part].device_id = deviceId;
    }
    return cudaSuccess;

fail:
    if (allocation != 0) cudaFree(allocation);
    for (part = begin; part < end; ++part) zero_record(&state->parts[part]);
    return err;
}

// Shard upload is a loop over per-part uploads. There is no packed multi-part
// transfer yet, so launch/copy count is proportional to parts per shard.
template<typename MatrixT>
__host__ __forceinline__ cudaError_t upload_shard(sharded_device<MatrixT> *state, const ::cellshard::sharded<MatrixT> *view, unsigned long shardId, int deviceId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    cudaError_t err = cudaSuccess;

    if (shardId >= view->num_shards) return cudaErrorInvalidValue;
    begin = ::cellshard::first_partition_in_shard(view, shardId);
    end = ::cellshard::last_partition_in_shard(view, shardId);
    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) return err;
    for (i = begin; i < end; ++i) {
        err = upload_partition_current_device(state, view, i, deviceId);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

__host__ __forceinline__ cudaError_t upload_shard(
    sharded_device< ::cellshard::sparse::compressed > *state,
    const ::cellshard::sharded< ::cellshard::sparse::compressed > *view,
    unsigned long shardId,
    int deviceId
) {
    cudaError_t err = cudaSuccess;

    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) return err;
    err = upload_compressed_shard_current_device_async(state, view, shardId, deviceId, (cudaStream_t) 0);
    if (err != cudaSuccess) return err;
    return cudaStreamSynchronize((cudaStream_t) 0);
}

__host__ __forceinline__ cudaError_t upload_shard(
    sharded_device< ::cellshard::sparse::blocked_ell > *state,
    const ::cellshard::sharded< ::cellshard::sparse::blocked_ell > *view,
    unsigned long shardId,
    int deviceId
) {
    cudaError_t err = cudaSuccess;

    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) return err;
    err = upload_blocked_ell_shard_current_device_async(state, view, shardId, deviceId, (cudaStream_t) 0);
    if (err != cudaSuccess) return err;
    return cudaStreamSynchronize((cudaStream_t) 0);
}

// Async shard upload preserves the same per-part behavior and copy count as the
// sync path. It simply queues those copies on one stream.
template<typename MatrixT>
__host__ __forceinline__ cudaError_t upload_shard_async(sharded_device<MatrixT> *state,
                                                        const ::cellshard::sharded<MatrixT> *view,
                                                        unsigned long shardId,
                                                        int deviceId,
                                                        cudaStream_t stream) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    cudaError_t err = cudaSuccess;

    if (shardId >= view->num_shards) return cudaErrorInvalidValue;
    begin = ::cellshard::first_partition_in_shard(view, shardId);
    end = ::cellshard::last_partition_in_shard(view, shardId);
    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) return err;
    for (i = begin; i < end; ++i) {
        err = upload_partition_current_device_async(state, view, i, deviceId, stream);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

__host__ __forceinline__ cudaError_t upload_shard_async(
    sharded_device< ::cellshard::sparse::compressed > *state,
    const ::cellshard::sharded< ::cellshard::sparse::compressed > *view,
    unsigned long shardId,
    int deviceId,
    cudaStream_t stream
) {
    cudaError_t err = cudaSuccess;

    if (shardId >= view->num_shards) return cudaErrorInvalidValue;
    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) return err;
    return upload_compressed_shard_current_device_async(state, view, shardId, deviceId, stream);
}

__host__ __forceinline__ cudaError_t upload_shard_async(
    sharded_device< ::cellshard::sparse::blocked_ell > *state,
    const ::cellshard::sharded< ::cellshard::sparse::blocked_ell > *view,
    unsigned long shardId,
    int deviceId,
    cudaStream_t stream
) {
    cudaError_t err = cudaSuccess;

    if (shardId >= view->num_shards) return cudaErrorInvalidValue;
    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) return err;
    return upload_blocked_ell_shard_current_device_async(state, view, shardId, deviceId, stream);
}

template<typename MatrixT>
__host__ __forceinline__ cudaError_t release_shard(sharded_device<MatrixT> *state, const ::cellshard::sharded<MatrixT> *view, unsigned long shardId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    cudaError_t err = cudaSuccess;

    if (shardId >= view->num_shards) return cudaErrorInvalidValue;
    begin = ::cellshard::first_partition_in_shard(view, shardId);
    end = ::cellshard::last_partition_in_shard(view, shardId);
    // Release loops over resident parts; there is no separate shard-level free
    // path beyond grouped packed allocations handled inside release_partition().
    for (i = begin; i < end; ++i) {
        if (i >= state->capacity || state->parts[i].view == 0) continue;
        err = release_partition(state, i);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

// stage_partition is the highest-risk convenience path in the file:
// - if the partition is absent on host, it performs synchronous source-backed fetch
// - it materializes a host partition object
// - it allocates device memory
// - it copies host payload to device
// - it may then free the host part immediately
//
// Nothing here is zero-copy and nothing here is hidden from the caller anymore.
template<typename MatrixT>
__host__ __forceinline__ cudaError_t stage_partition(sharded_device<MatrixT> *state,
                              ::cellshard::sharded<MatrixT> *view,
                              const ::cellshard::shard_storage *files,
                              unsigned long partId,
                              int deviceId,
                              int drop_host_after_upload) {
    cudaError_t err = cudaSuccess;

    if (partId >= view->num_partitions || partId >= state->capacity) return cudaErrorInvalidValue;
    if (state->parts[partId].view != 0 && state->parts[partId].device_id == deviceId) {
        if (drop_host_after_upload && view->parts[partId] != 0) {
            if (!::cellshard::drop_partition(view, partId)) return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }
    if (view->parts[partId] == 0) {
        if (!::cellshard::fetch_partition(view, files, partId)) return cudaErrorInvalidValue;
    }
    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) return err;
    err = upload_partition_current_device(state, view, partId, deviceId);
    if (err != cudaSuccess) return err;
    if (drop_host_after_upload) {
        if (!::cellshard::drop_partition(view, partId)) return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

// The async staging path still does synchronous host fetch when the part is not
// resident in view->parts[]. Only the upload half is stream-ordered.
template<typename MatrixT>
__host__ __forceinline__ cudaError_t stage_partition_async(sharded_device<MatrixT> *state,
                                                           ::cellshard::sharded<MatrixT> *view,
                                                           const ::cellshard::shard_storage *files,
                                                           unsigned long partId,
                                                           int deviceId,
                                                           cudaStream_t stream,
                                                           int drop_host_after_upload) {
    cudaError_t err = cudaSuccess;

    if (partId >= view->num_partitions || partId >= state->capacity) return cudaErrorInvalidValue;
    if (state->parts[partId].view != 0 && state->parts[partId].device_id == deviceId) {
        if (drop_host_after_upload && view->parts[partId] != 0) {
            if (!::cellshard::drop_partition(view, partId)) return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }
    if (view->parts[partId] == 0) {
        if (!::cellshard::fetch_partition(view, files, partId)) return cudaErrorInvalidValue;
    }
    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) return err;
    err = upload_partition_current_device_async(state, view, partId, deviceId, stream);
    if (err != cudaSuccess) return err;
    if (drop_host_after_upload) {
        if (!::cellshard::drop_partition(view, partId)) return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

// Shard staging now fetches any cold host parts for the whole shard first and
// then uploads with one device selection across the shard.
template<typename MatrixT>
__host__ __forceinline__ cudaError_t stage_shard(sharded_device<MatrixT> *state,
                               ::cellshard::sharded<MatrixT> *view,
                               const ::cellshard::shard_storage *files,
                               unsigned long shardId,
                               int deviceId,
                               int drop_host_after_upload) {
    cudaError_t err = cudaSuccess;

    if (shardId >= view->num_shards) return cudaErrorInvalidValue;
    if (!::cellshard::shard_loaded(view, shardId)) {
        if (!::cellshard::fetch_shard(view, files, shardId)) return cudaErrorInvalidValue;
    }
    err = upload_shard(state, view, shardId, deviceId);
    if (err != cudaSuccess) return err;
    if (drop_host_after_upload) {
        if (!::cellshard::drop_shard(view, shardId)) return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

// Async shard staging now front-loads any cold host fetch for the whole shard,
// then uploads the resident part set with one device selection.
template<typename MatrixT>
__host__ __forceinline__ cudaError_t stage_shard_async(sharded_device<MatrixT> *state,
                                                       ::cellshard::sharded<MatrixT> *view,
                                                       const ::cellshard::shard_storage *files,
                                                       unsigned long shardId,
                                                       int deviceId,
                                                       cudaStream_t stream,
                                                       int drop_host_after_upload) {
    cudaError_t err = cudaSuccess;

    if (shardId >= view->num_shards) return cudaErrorInvalidValue;
    if (!::cellshard::shard_loaded(view, shardId)) {
        if (!::cellshard::fetch_shard(view, files, shardId)) return cudaErrorInvalidValue;
    }
    err = upload_shard_async(state, view, shardId, deviceId, stream);
    if (err != cudaSuccess) return err;
    if (drop_host_after_upload) {
        if (!::cellshard::drop_shard(view, shardId)) return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

// swap_shard always stages the incoming shard before it releases the outgoing
// shard. The peak device footprint therefore includes both shards transiently.
template<typename MatrixT>
__host__ __forceinline__ cudaError_t swap_shard(sharded_device<MatrixT> *state,
                              ::cellshard::sharded<MatrixT> *view,
                              const ::cellshard::shard_storage *files,
                              unsigned long outShardId,
                              unsigned long inShardId,
                              int deviceId,
                              int drop_host_after_upload,
                              int drop_host_after_release) {
    cudaError_t err = cudaSuccess;

    err = stage_shard(state, view, files, inShardId, deviceId, drop_host_after_upload);
    if (err != cudaSuccess) return err;
    err = release_shard(state, view, outShardId);
    if (err != cudaSuccess) return err;
    if (drop_host_after_release) {
        if (!::cellshard::drop_shard(view, outShardId)) return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

// The async swap path preserves the same transient double-residency behavior as
// the sync path. It is not an in-place exchange.
template<typename MatrixT>
__host__ __forceinline__ cudaError_t swap_shard_async(sharded_device<MatrixT> *state,
                                                      ::cellshard::sharded<MatrixT> *view,
                                                      const ::cellshard::shard_storage *files,
                                                      unsigned long outShardId,
                                                      unsigned long inShardId,
                                                      int deviceId,
                                                      cudaStream_t stream,
                                                      int drop_host_after_upload,
                                                      int drop_host_after_release) {
    cudaError_t err = cudaSuccess;

    err = stage_shard_async(state, view, files, inShardId, deviceId, stream, drop_host_after_upload);
    if (err != cudaSuccess) return err;
    err = release_shard(state, view, outShardId);
    if (err != cudaSuccess) return err;
    if (drop_host_after_release) {
        if (!::cellshard::drop_shard(view, outShardId)) return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

} // namespace device
} // namespace cellshard
