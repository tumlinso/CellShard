#pragma once

#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include "sharded_host.cuh"

namespace cellshard {
namespace device {

template<typename MatrixT>
struct alignas(16) part_record {
    void *view;
    void *a0;
    void *a1;
    void *a2;
    int device_id;
};

template<typename MatrixT>
struct sharded_device {
    unsigned long capacity;
    part_record<MatrixT> *parts;
};

template<typename MatrixT>
inline cudaError_t release(part_record<MatrixT> *record);

template<typename MatrixT>
inline cudaError_t release_part(sharded_device<MatrixT> *state, unsigned long partId);

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
}

template<typename MatrixT>
__host__ __forceinline__ void clear(sharded_device<MatrixT> *state) {
    unsigned long i = 0;
    if (state->parts != 0) {
        for (i = 0; i < state->capacity; ++i) {
            if (state->parts[i].view != 0) release_part(state, i);
        }
    }
    std::free(state->parts);
    state->capacity = 0;
    state->parts = 0;
}

template<typename MatrixT>
__host__ __forceinline__ int reserve(sharded_device<MatrixT> *state, unsigned long capacity) {
    part_record<MatrixT> *records = 0;
    unsigned long i = 0;

    if (capacity <= state->capacity) return 1;
    records = (part_record<MatrixT> *) std::malloc((std::size_t) capacity * sizeof(part_record<MatrixT>));
    if (records == 0) return 0;
    std::memset(records, 0, (std::size_t) capacity * sizeof(part_record<MatrixT>));
    for (i = 0; i < capacity; ++i) records[i].device_id = -1;
    for (i = 0; i < state->capacity; ++i) records[i] = state->parts[i];
    std::free(state->parts);
    state->parts = records;
    state->capacity = capacity;
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ void zero_record(part_record<MatrixT> *record) {
    record->view = 0;
    record->a0 = 0;
    record->a1 = 0;
    record->a2 = 0;
    record->device_id = -1;
}

__host__ __forceinline__ cudaError_t upload(const ::cellshard::dense *src, part_record< ::cellshard::dense > *record) {
    dense_view host;
    dense_view *deviceView = 0;
    const std::size_t count = (std::size_t) src->rows * (std::size_t) src->cols;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.val = 0;

    if (count != 0) {
        err = cudaMalloc((void **) &host.val, count * sizeof(__half));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.val, src->val, count * sizeof(__half), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }

    err = cudaMalloc((void **) &deviceView, sizeof(host));
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(deviceView, &host, sizeof(host), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    record->view = deviceView;
    record->a0 = host.val;
    return cudaSuccess;

fail:
    if (deviceView != 0) cudaFree(deviceView);
    if (host.val != 0) cudaFree(host.val);
    zero_record(record);
    return err;
}

__host__ __forceinline__ cudaError_t upload(const ::cellshard::sparse::compressed *src, part_record< ::cellshard::sparse::compressed > *record) {
    compressed_view host;
    compressed_view *deviceView = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.nnz = src->nnz;
    host.axis = src->axis;
    host.majorPtr = 0;
    host.minorIdx = 0;
    host.val = 0;

    if (sparse::major_dim(src) != 0) {
        err = cudaMalloc((void **) &host.majorPtr, ((std::size_t) sparse::major_dim(src) + 1) * sizeof(unsigned int));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.majorPtr, src->majorPtr, ((std::size_t) sparse::major_dim(src) + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }

    if (src->nnz != 0) {
        err = cudaMalloc((void **) &host.minorIdx, src->nnz * sizeof(unsigned int));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.minorIdx, src->minorIdx, src->nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
        err = cudaMalloc((void **) &host.val, src->nnz * sizeof(__half));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.val, src->val, src->nnz * sizeof(__half), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }

    err = cudaMalloc((void **) &deviceView, sizeof(host));
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(deviceView, &host, sizeof(host), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    record->view = deviceView;
    record->a0 = host.majorPtr;
    record->a1 = host.minorIdx;
    record->a2 = host.val;
    return cudaSuccess;

fail:
    if (deviceView != 0) cudaFree(deviceView);
    if (host.val != 0) cudaFree(host.val);
    if (host.minorIdx != 0) cudaFree(host.minorIdx);
    if (host.majorPtr != 0) cudaFree(host.majorPtr);
    zero_record(record);
    return err;
}

__host__ __forceinline__ cudaError_t upload(const ::cellshard::sparse::coo *src, part_record< ::cellshard::sparse::coo > *record) {
    coo_view host;
    coo_view *deviceView = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.nnz = src->nnz;
    host.rowIdx = 0;
    host.colIdx = 0;
    host.val = 0;

    if (src->nnz != 0) {
        err = cudaMalloc((void **) &host.rowIdx, src->nnz * sizeof(unsigned int));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.rowIdx, src->rowIdx, src->nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
        err = cudaMalloc((void **) &host.colIdx, src->nnz * sizeof(unsigned int));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.colIdx, src->colIdx, src->nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
        err = cudaMalloc((void **) &host.val, src->nnz * sizeof(__half));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.val, src->val, src->nnz * sizeof(__half), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }

    err = cudaMalloc((void **) &deviceView, sizeof(host));
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(deviceView, &host, sizeof(host), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    record->view = deviceView;
    record->a0 = host.rowIdx;
    record->a1 = host.colIdx;
    record->a2 = host.val;
    return cudaSuccess;

fail:
    if (deviceView != 0) cudaFree(deviceView);
    if (host.val != 0) cudaFree(host.val);
    if (host.colIdx != 0) cudaFree(host.colIdx);
    if (host.rowIdx != 0) cudaFree(host.rowIdx);
    zero_record(record);
    return err;
}

__host__ __forceinline__ cudaError_t upload(const ::cellshard::sparse::dia *src, part_record< ::cellshard::sparse::dia > *record) {
    dia_view host;
    dia_view *deviceView = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.nnz = src->nnz;
    host.num_diagonals = src->num_diagonals;
    host.offsets = 0;
    host.val = 0;

    if (src->num_diagonals != 0) {
        err = cudaMalloc((void **) &host.offsets, src->num_diagonals * sizeof(int));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.offsets, src->offsets, src->num_diagonals * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }
    if (src->nnz != 0) {
        err = cudaMalloc((void **) &host.val, src->nnz * sizeof(__half));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.val, src->val, src->nnz * sizeof(__half), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }

    err = cudaMalloc((void **) &deviceView, sizeof(host));
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(deviceView, &host, sizeof(host), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    record->view = deviceView;
    record->a0 = host.offsets;
    record->a1 = host.val;
    return cudaSuccess;

fail:
    if (deviceView != 0) cudaFree(deviceView);
    if (host.val != 0) cudaFree(host.val);
    if (host.offsets != 0) cudaFree(host.offsets);
    zero_record(record);
    return err;
}

template<typename MatrixT>
__host__ __forceinline__ cudaError_t release(part_record<MatrixT> *record) {
    cudaError_t err = cudaSuccess;

    if (record->a2 != 0) {
        err = cudaFree(record->a2);
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

__host__ __forceinline__ std::size_t device_part_bytes(const ::cellshard::sharded< ::cellshard::dense > *view, unsigned long partId) {
    return sizeof(dense_view) + (std::size_t) view->part_nnz[partId] * sizeof(__half);
}

__host__ __forceinline__ std::size_t device_part_bytes(const ::cellshard::sharded< ::cellshard::sparse::compressed > *view, unsigned long partId) {
    const unsigned long ptr_dim = view->part_aux[partId] == sparse::compressed_by_col ? view->cols : view->part_rows[partId];
    return sizeof(compressed_view)
        + (std::size_t) (ptr_dim + 1) * sizeof(unsigned int)
        + (std::size_t) view->part_nnz[partId] * sizeof(unsigned int)
        + (std::size_t) view->part_nnz[partId] * sizeof(__half);
}

__host__ __forceinline__ std::size_t device_part_bytes(const ::cellshard::sharded< ::cellshard::sparse::coo > *view, unsigned long partId) {
    return sizeof(coo_view)
        + (std::size_t) view->part_nnz[partId] * sizeof(unsigned int)
        + (std::size_t) view->part_nnz[partId] * sizeof(unsigned int)
        + (std::size_t) view->part_nnz[partId] * sizeof(__half);
}

__host__ __forceinline__ std::size_t device_part_bytes(const ::cellshard::sharded< ::cellshard::sparse::dia > *view, unsigned long partId) {
    return sizeof(dia_view)
        + (std::size_t) view->part_aux[partId] * sizeof(int)
        + (std::size_t) view->part_nnz[partId] * sizeof(__half);
}

template<typename MatrixT>
__host__ __forceinline__ std::size_t device_shard_bytes(const ::cellshard::sharded<MatrixT> *view, unsigned long shardId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    std::size_t total = 0;

    if (shardId >= view->num_shards) return 0;
    begin = ::cellshard::first_part_in_shard(view, shardId);
    end = ::cellshard::last_part_in_shard(view, shardId);
    for (i = begin; i < end; ++i) total += device_part_bytes(view, i);
    return total;
}

template<typename MatrixT>
__host__ __forceinline__ int set_shards_by_device_bytes(::cellshard::sharded<MatrixT> *view, std::size_t max_bytes) {
    std::size_t used = 0;
    std::size_t bytes = 0;
    unsigned long shardCount = 0;
    unsigned long i = 0;

    if (max_bytes == 0) return ::cellshard::set_shards_to_parts(view);
    if (!::cellshard::reserve_shards(view, view->num_parts)) return 0;

    view->shard_offsets[0] = 0;
    for (i = 0; i < view->num_parts; ++i) {
        bytes = device_part_bytes(view, i);
        if (bytes == 0) continue;
        if (used != 0 && used + bytes > max_bytes) {
            ++shardCount;
            view->shard_offsets[shardCount] = view->part_offsets[i];
            used = 0;
        }
        used += bytes;
    }
    if (view->num_parts != 0) {
        ++shardCount;
        view->shard_offsets[shardCount] = view->rows;
    }
    view->num_shards = shardCount;
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ cudaError_t upload_part(sharded_device<MatrixT> *state, const ::cellshard::sharded<MatrixT> *view, unsigned long partId, int deviceId) {
    cudaError_t err = cudaSuccess;

    if (partId >= view->num_parts || partId >= state->capacity || view->parts[partId] == 0) return cudaErrorInvalidValue;
    if (state->parts[partId].view != 0) {
        if (state->parts[partId].device_id == deviceId) return cudaSuccess;
        err = release_part(state, partId);
        if (err != cudaSuccess) return err;
    }
    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) return err;
    err = upload(view->parts[partId], &state->parts[partId]);
    if (err != cudaSuccess) return err;
    state->parts[partId].device_id = deviceId;
    return cudaSuccess;
}

template<typename MatrixT>
__host__ __forceinline__ cudaError_t release_part(sharded_device<MatrixT> *state, unsigned long partId) {
    cudaError_t err = cudaSuccess;

    if (partId >= state->capacity || state->parts[partId].view == 0) return cudaSuccess;
    err = cudaSetDevice(state->parts[partId].device_id >= 0 ? state->parts[partId].device_id : 0);
    if (err != cudaSuccess) return err;
    return release(&state->parts[partId]);
}

template<typename MatrixT>
__host__ __forceinline__ cudaError_t upload_shard(sharded_device<MatrixT> *state, const ::cellshard::sharded<MatrixT> *view, unsigned long shardId, int deviceId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    cudaError_t err = cudaSuccess;

    if (shardId >= view->num_shards) return cudaErrorInvalidValue;
    begin = ::cellshard::first_part_in_shard(view, shardId);
    end = ::cellshard::last_part_in_shard(view, shardId);
    for (i = begin; i < end; ++i) {
        err = upload_part(state, view, i, deviceId);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

template<typename MatrixT>
__host__ __forceinline__ cudaError_t release_shard(sharded_device<MatrixT> *state, const ::cellshard::sharded<MatrixT> *view, unsigned long shardId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    cudaError_t err = cudaSuccess;

    if (shardId >= view->num_shards) return cudaErrorInvalidValue;
    begin = ::cellshard::first_part_in_shard(view, shardId);
    end = ::cellshard::last_part_in_shard(view, shardId);
    for (i = begin; i < end; ++i) {
        err = release_part(state, i);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

template<typename MatrixT>
__host__ __forceinline__ cudaError_t stage_part(sharded_device<MatrixT> *state,
                              ::cellshard::sharded<MatrixT> *view,
                              const ::cellshard::shard_storage *files,
                              unsigned long partId,
                              int deviceId,
                              int drop_host_after_upload) {
    cudaError_t err = cudaSuccess;

    if (partId >= view->num_parts || partId >= state->capacity) return cudaErrorInvalidValue;
    if (state->parts[partId].view != 0 && state->parts[partId].device_id == deviceId) {
        if (drop_host_after_upload && view->parts[partId] != 0) {
            if (!::cellshard::drop_part(view, partId)) return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }
    if (view->parts[partId] == 0) {
        if (!::cellshard::fetch_part(view, files, partId)) return cudaErrorInvalidValue;
    }
    err = upload_part(state, view, partId, deviceId);
    if (err != cudaSuccess) return err;
    if (drop_host_after_upload) {
        if (!::cellshard::drop_part(view, partId)) return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

template<typename MatrixT>
__host__ __forceinline__ cudaError_t stage_shard(sharded_device<MatrixT> *state,
                               ::cellshard::sharded<MatrixT> *view,
                               const ::cellshard::shard_storage *files,
                               unsigned long shardId,
                               int deviceId,
                               int drop_host_after_upload) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    cudaError_t err = cudaSuccess;

    if (shardId >= view->num_shards) return cudaErrorInvalidValue;
    begin = ::cellshard::first_part_in_shard(view, shardId);
    end = ::cellshard::last_part_in_shard(view, shardId);
    for (i = begin; i < end; ++i) {
        err = stage_part(state, view, files, i, deviceId, drop_host_after_upload);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

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

} // namespace device
} // namespace cellshard
