#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <Cellerator/dist/distributed.cuh>

#include "../device/sharded_device.cuh"

namespace cellshard {
namespace distributed {

struct shard_weight {
    unsigned long shard_id;
    std::size_t bytes;
};

inline int compare_shard_weight_desc(const void *lhs, const void *rhs) {
    const shard_weight *a = (const shard_weight *) lhs;
    const shard_weight *b = (const shard_weight *) rhs;
    if (a->bytes > b->bytes) return -1;
    if (a->bytes < b->bytes) return 1;
    if (a->shard_id < b->shard_id) return -1;
    if (a->shard_id > b->shard_id) return 1;
    return 0;
}

struct shard_map {
    unsigned long shard_count;
    int *device_slot;
    std::size_t *device_bytes;
};

template<typename MatrixT>
struct device_fleet {
    unsigned int count;
    ::cellshard::device::sharded_device<MatrixT> *states;
};

inline void init(shard_map *map) {
    map->shard_count = 0;
    map->device_slot = 0;
    map->device_bytes = 0;
}

inline void clear(shard_map *map) {
    std::free(map->device_slot);
    std::free(map->device_bytes);
    init(map);
}

inline int reserve(shard_map *map, unsigned long shard_count, unsigned int device_count) {
    int *slots = 0;
    std::size_t *bytes = 0;

    clear(map);
    if (shard_count != 0) {
        slots = (int *) std::calloc((std::size_t) shard_count, sizeof(int));
        if (slots == 0) return 0;
    }
    if (device_count != 0) {
        bytes = (std::size_t *) std::calloc((std::size_t) device_count, sizeof(std::size_t));
        if (bytes == 0) {
            std::free(slots);
            return 0;
        }
    }
    map->shard_count = shard_count;
    map->device_slot = slots;
    map->device_bytes = bytes;
    return 1;
}

template<typename MatrixT>
inline int assign_shards_round_robin(shard_map *map,
                                     const ::cellshard::sharded<MatrixT> *view,
                                     const ::cellerator::dist::local_context *ctx) {
    unsigned long i = 0;

    if (map == 0 || view == 0 || ctx == 0 || ctx->device_count == 0) return 0;
    if (!reserve(map, view->num_shards, ctx->device_count)) return 0;
    for (i = 0; i < view->num_shards; ++i) {
        map->device_slot[i] = (int) (i % ctx->device_count);
        map->device_bytes[map->device_slot[i]] += ::cellshard::device::device_shard_bytes(view, i);
    }
    return 1;
}

template<typename MatrixT>
inline int assign_shards_by_bytes(shard_map *map,
                                  const ::cellshard::sharded<MatrixT> *view,
                                  const ::cellerator::dist::local_context *ctx) {
    shard_weight *weights = 0;
    unsigned long i = 0;
    int ok = 0;

    if (map == 0 || view == 0 || ctx == 0 || ctx->device_count == 0) return 0;
    if (!reserve(map, view->num_shards, ctx->device_count)) return 0;
    // Largest-first greedy placement is materially better than input-order
    // assignment for skewed shard sizes. The real bottleneck here is eventual
    // resident device footprint, so sort by device_shard_bytes() before the
    // per-device load-balancing pass.
    if (view->num_shards != 0) {
        weights = (shard_weight *) std::calloc((std::size_t) view->num_shards, sizeof(shard_weight));
        if (weights == 0) return 0;
    }
    for (i = 0; i < view->num_shards; ++i) {
        weights[i].shard_id = i;
        weights[i].bytes = ::cellshard::device::device_shard_bytes(view, i);
    }
    std::qsort(weights, (std::size_t) view->num_shards, sizeof(shard_weight), compare_shard_weight_desc);
    for (i = 0; i < view->num_shards; ++i) {
        unsigned int best = 0;
        unsigned int d = 1;
        const unsigned long shard_id = weights[i].shard_id;
        const std::size_t shard_bytes = weights[i].bytes;
        for (d = 1; d < ctx->device_count; ++d) {
            if (map->device_bytes[d] < map->device_bytes[best]) best = d;
        }
        map->device_slot[shard_id] = (int) best;
        map->device_bytes[best] += shard_bytes;
    }
    ok = 1;
    std::free(weights);
    return ok;
}

template<typename MatrixT>
inline void init(device_fleet<MatrixT> *fleet) {
    fleet->count = 0;
    fleet->states = 0;
}

template<typename MatrixT>
inline void clear(device_fleet<MatrixT> *fleet) {
    unsigned int i = 0;
    if (fleet->states != 0) {
        for (i = 0; i < fleet->count; ++i) ::cellshard::device::clear(fleet->states + i);
    }
    std::free(fleet->states);
    init(fleet);
}

template<typename MatrixT>
inline int reserve(device_fleet<MatrixT> *fleet, unsigned int count) {
    ::cellshard::device::sharded_device<MatrixT> *states = 0;
    unsigned int i = 0;

    clear(fleet);
    if (count == 0) return 1;
    states = (::cellshard::device::sharded_device<MatrixT> *) std::calloc((std::size_t) count, sizeof(::cellshard::device::sharded_device<MatrixT>));
    if (states == 0) return 0;
    for (i = 0; i < count; ++i) ::cellshard::device::init(states + i);
    fleet->count = count;
    fleet->states = states;
    return 1;
}

template<typename MatrixT>
inline int reserve_parts(device_fleet<MatrixT> *fleet, unsigned long capacity) {
    unsigned int i = 0;
    if (fleet == 0) return 0;
    for (i = 0; i < fleet->count; ++i) {
        if (!::cellshard::device::reserve(fleet->states + i, capacity)) return 0;
    }
    return 1;
}

template<typename MatrixT>
inline cudaError_t stage_shard_on_owner(device_fleet<MatrixT> *fleet,
                                        const ::cellerator::dist::local_context *ctx,
                                        shard_map *map,
                                        ::cellshard::sharded<MatrixT> *view,
                                        const ::cellshard::shard_storage *storage,
                                        unsigned long shardId,
                                        int drop_host_after_upload) {
    const int slot = (map != 0 && shardId < map->shard_count && map->device_slot != 0) ? map->device_slot[shardId] : -1;
    cudaStream_t stream = 0;

    if (fleet == 0 || ctx == 0 || view == 0) return cudaErrorInvalidValue;
    if (slot < 0 || (unsigned int) slot >= fleet->count || (unsigned int) slot >= ctx->device_count) return cudaErrorInvalidValue;
    if (ctx->streams != 0) stream = ctx->streams[slot];
    // This calls directly into stage_shard_async(), so it may trigger:
    // - synchronous source-backed host fetch for cold parts
    // - device allocation
    // - H2D copy on the owner's stream
    return ::cellshard::device::stage_shard_async(fleet->states + slot,
                                                  view,
                                                  storage,
                                                  shardId,
                                                  ctx->device_ids[slot],
                                                  stream,
                                                  drop_host_after_upload);
}

template<typename MatrixT>
inline cudaError_t stage_all_shards_on_owners(device_fleet<MatrixT> *fleet,
                                              const ::cellerator::dist::local_context *ctx,
                                              shard_map *map,
                                              ::cellshard::sharded<MatrixT> *view,
                                              const ::cellshard::shard_storage *storage,
                                              int drop_host_after_upload) {
    std::vector<std::thread> workers;
    std::atomic<int> first_error((int) cudaSuccess);

    if (view == 0) return cudaErrorInvalidValue;
    // Queue owner-local work concurrently across GPUs. A single host thread
    // walking every device serially can leave copy engines underfed even when
    // each device has its own stream and independent shard queue.
    workers.reserve(ctx->device_count);
    for (unsigned int slot = 0; slot < ctx->device_count; ++slot) {
        workers.emplace_back([&, slot]() {
            unsigned long i = 0;
            for (i = 0; i < view->num_shards; ++i) {
                cudaError_t err = cudaSuccess;
                if (first_error.load() != (int) cudaSuccess) return;
                if (map == 0 || map->device_slot == 0 || i >= map->shard_count) {
                    int expected = (int) cudaSuccess;
                    first_error.compare_exchange_strong(expected, (int) cudaErrorInvalidValue);
                    return;
                }
                if ((unsigned int) map->device_slot[i] != slot) continue;
                err = stage_shard_on_owner(fleet, ctx, map, view, storage, i, drop_host_after_upload);
                if (err != cudaSuccess) {
                    int expected = (int) cudaSuccess;
                    first_error.compare_exchange_strong(expected, (int) err);
                    return;
                }
            }
        });
    }
    for (std::thread &worker : workers) worker.join();
    return (cudaError_t) first_error.load();
}

} // namespace distributed
} // namespace cellshard
