#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#if defined(__has_include)
#if __has_include(<nccl.h>)
#include <nccl.h>
#define CELLSHARD_HAS_NCCL 1
#elif __has_include("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/12.9/nccl/include/nccl.h")
#include "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/12.9/nccl/include/nccl.h"
#define CELLSHARD_HAS_NCCL 1
#endif
#endif

#ifndef CELLSHARD_HAS_NCCL
#define CELLSHARD_HAS_NCCL 0
#endif

#include "sharded_device.cuh"

namespace cellshard {
namespace distributed {

// local_context is intentionally minimal:
// - one visible device id per local GPU
// - one optional stream per device
// - a dense peer-access capability table
// - optional NCCL communicators
//
// This is not a scheduler and not a hidden runtime. It only holds enough state
// to make explicit multi-GPU shard placement cheap for the caller.
struct local_context {
    unsigned int device_count;
    int *device_ids;
    cudaStream_t *streams;
    unsigned char *peer_access;
#if CELLSHARD_HAS_NCCL
    ncclComm_t *comms;
#endif
};

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

inline void init(local_context *ctx) {
    ctx->device_count = 0;
    ctx->device_ids = 0;
    ctx->streams = 0;
    ctx->peer_access = 0;
#if CELLSHARD_HAS_NCCL
    ctx->comms = 0;
#endif
}

inline void clear(local_context *ctx) {
    unsigned int i = 0;

    if (ctx->streams != 0) {
        for (i = 0; i < ctx->device_count; ++i) {
            if (ctx->streams[i] != 0) {
                cudaSetDevice(ctx->device_ids != 0 ? ctx->device_ids[i] : (int) i);
                cudaStreamDestroy(ctx->streams[i]);
            }
        }
    }
#if CELLSHARD_HAS_NCCL
    if (ctx->comms != 0) {
        for (i = 0; i < ctx->device_count; ++i) {
            if (ctx->comms[i] != 0) ncclCommDestroy(ctx->comms[i]);
        }
    }
    std::free(ctx->comms);
#endif
    std::free(ctx->peer_access);
    std::free(ctx->streams);
    std::free(ctx->device_ids);
    init(ctx);
}

inline cudaError_t discover_local(local_context *ctx, int create_streams, unsigned int stream_flags) {
    int count = 0;
    unsigned int i = 0;
    cudaError_t err = cudaSuccess;

    clear(ctx);
    // Enumerate only CUDA-visible devices. The caller controls visibility with
    // CUDA_VISIBLE_DEVICES or equivalent before process launch.
    err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return err;
    if (count <= 0) return cudaSuccess;

    ctx->device_ids = (int *) std::calloc((std::size_t) count, sizeof(int));
    ctx->peer_access = (unsigned char *) std::calloc((std::size_t) count * (std::size_t) count, sizeof(unsigned char));
    if (ctx->device_ids == 0 || ctx->peer_access == 0) {
        clear(ctx);
        return cudaErrorMemoryAllocation;
    }
    if (create_streams) {
        ctx->streams = (cudaStream_t *) std::calloc((std::size_t) count, sizeof(cudaStream_t));
        if (ctx->streams == 0) {
            clear(ctx);
            return cudaErrorMemoryAllocation;
        }
    }
#if CELLSHARD_HAS_NCCL
    ctx->comms = (ncclComm_t *) std::calloc((std::size_t) count, sizeof(ncclComm_t));
    if (ctx->comms == 0) {
        clear(ctx);
        return cudaErrorMemoryAllocation;
    }
#endif

    ctx->device_count = (unsigned int) count;
    for (i = 0; i < ctx->device_count; ++i) {
        ctx->device_ids[i] = (int) i;
        if (ctx->streams != 0) {
            // One stream per device keeps staging explicit and graph-friendly.
            err = cudaSetDevice(ctx->device_ids[i]);
            if (err != cudaSuccess) {
                clear(ctx);
                return err;
            }
            err = cudaStreamCreateWithFlags(ctx->streams + i, stream_flags);
            if (err != cudaSuccess) {
                clear(ctx);
                return err;
            }
        }
    }

    for (i = 0; i < ctx->device_count; ++i) {
        unsigned int j = 0;
        err = cudaSetDevice(ctx->device_ids[i]);
        if (err != cudaSuccess) {
            clear(ctx);
            return err;
        }
        for (j = 0; j < ctx->device_count; ++j) {
            int can_access = 0;
            err = cudaDeviceCanAccessPeer(&can_access, ctx->device_ids[i], ctx->device_ids[j]);
            if (err != cudaSuccess) {
                clear(ctx);
                return err;
            }
            ctx->peer_access[(std::size_t) i * ctx->device_count + j] = (unsigned char) (can_access != 0);
        }
    }
    return cudaSuccess;
}

inline int peer_access_supported(const local_context *ctx, unsigned int src_slot, unsigned int dst_slot) {
    if (ctx == 0 || ctx->peer_access == 0) return 0;
    if (src_slot >= ctx->device_count || dst_slot >= ctx->device_count) return 0;
    return ctx->peer_access[(std::size_t) src_slot * ctx->device_count + dst_slot] != 0;
}

inline cudaError_t enable_peer_access(local_context *ctx) {
    unsigned int i = 0;
    cudaError_t err = cudaSuccess;

    if (ctx == 0) return cudaErrorInvalidValue;
    for (i = 0; i < ctx->device_count; ++i) {
        unsigned int j = 0;
        err = cudaSetDevice(ctx->device_ids[i]);
        if (err != cudaSuccess) return err;
        for (j = 0; j < ctx->device_count; ++j) {
            if (i == j) continue;
            if (!peer_access_supported(ctx, i, j)) continue;
            // Peer access only enables direct addressability. It does not move
            // any bytes by itself.
            err = cudaDeviceEnablePeerAccess(ctx->device_ids[j], 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError();
                err = cudaSuccess;
            }
            if (err != cudaSuccess) return err;
        }
    }
    return cudaSuccess;
}

#if CELLSHARD_HAS_NCCL
inline ncclResult_t init_local_nccl(local_context *ctx) {
    if (ctx == 0 || ctx->device_count == 0 || ctx->device_ids == 0 || ctx->comms == 0) return ncclInvalidArgument;
    return ncclCommInitAll(ctx->comms, (int) ctx->device_count, ctx->device_ids);
}
#endif

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
                                     const local_context *ctx) {
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
                                  const local_context *ctx) {
    unsigned long i = 0;

    if (map == 0 || view == 0 || ctx == 0 || ctx->device_count == 0) return 0;
    if (!reserve(map, view->num_shards, ctx->device_count)) return 0;
    for (i = 0; i < view->num_shards; ++i) {
        unsigned int best = 0;
        unsigned int d = 1;
        // Balance by eventual resident device footprint, not source bytes on
        // disk. This is the metric that matters once shards are uploaded.
        const std::size_t shard_bytes = ::cellshard::device::device_shard_bytes(view, i);
        for (d = 1; d < ctx->device_count; ++d) {
            if (map->device_bytes[d] < map->device_bytes[best]) best = d;
        }
        map->device_slot[i] = (int) best;
        map->device_bytes[best] += shard_bytes;
    }
    return 1;
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
                                        const local_context *ctx,
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
    // - synchronous packfile fetch on host for cold parts
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
                                              const local_context *ctx,
                                              shard_map *map,
                                              ::cellshard::sharded<MatrixT> *view,
                                              const ::cellshard::shard_storage *storage,
                                              int drop_host_after_upload) {
    unsigned long i = 0;
    cudaError_t err = cudaSuccess;

    if (view == 0) return cudaErrorInvalidValue;
    // This is currently a host loop over shards. It is explicit by design so
    // the caller can replace it with a more specialized schedule later.
    for (i = 0; i < view->num_shards; ++i) {
        err = stage_shard_on_owner(fleet, ctx, map, view, storage, i, drop_host_after_upload);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

inline cudaError_t synchronize(const local_context *ctx) {
    unsigned int i = 0;
    cudaError_t err = cudaSuccess;

    if (ctx == 0) return cudaErrorInvalidValue;
    for (i = 0; i < ctx->device_count; ++i) {
        err = cudaSetDevice(ctx->device_ids[i]);
        if (err != cudaSuccess) return err;
        if (ctx->streams != 0 && ctx->streams[i] != 0) err = cudaStreamSynchronize(ctx->streams[i]);
        else err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

} // namespace distributed
} // namespace cellshard
