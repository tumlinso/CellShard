#include "shared.hh"

inline int bind_dataset_h5_with_role(shard_storage *s, const char *path, std::uint32_t role) {
    std::size_t len = 0;
    char *copy = 0;
    dataset_h5_state *state = 0;

    if (s == 0) return 0;
    if (s->close_backend != 0) s->close_backend(s);
    std::free(s->source_path);
    s->source_path = 0;
    if (path == 0) return 1;

    len = std::strlen(path);
    copy = (char *) std::malloc(len + 1u);
    state = (dataset_h5_state *) std::calloc(1u, sizeof(dataset_h5_state));
    if (copy == 0 || state == 0) {
        std::free(copy);
        std::free(state);
        return 0;
    }
    std::memcpy(copy, path, len + 1u);
    dataset_h5_state_init(state);
    s->source_path = copy;
    s->backend = shard_storage_backend_dataset_h5;
    s->backend_state = state;
    s->open_backend = open_dataset_h5_backend;
    s->close_backend = close_dataset_h5_backend;
    if (!set_shard_storage_role(s, role)) {
        clear(s);
        return 0;
    }
    return 1;
}

int bind_dataset_h5_builder(shard_storage *s, const char *path) {
    return bind_dataset_h5_with_role(s, path, shard_storage_role_builder);
}

int bind_dataset_h5_owner_runtime(shard_storage *s, const char *path) {
    return bind_dataset_h5_with_role(s, path, shard_storage_role_owner_runtime);
}

int bind_dataset_h5(shard_storage *s, const char *path) {
    return bind_dataset_h5_owner_runtime(s, path);
}

int adopt_dataset_h5_executor_role(shard_storage *s) {
    dataset_h5_state *state = dataset_h5_state_from_storage(s);
    if (state == 0) return 0;
    close_dataset_h5_open_handles(state);
    return set_shard_storage_role(s, shard_storage_role_executor);
}

int bind_dataset_h5_cache(shard_storage *s, const char *cache_root) {
    dataset_h5_state *state = dataset_h5_state_from_storage(s);

    if (state == 0) return 0;
    if (state->cache_root != 0 && cache_root != 0 && std::strcmp(state->cache_root, cache_root) != 0) {
        invalidate_dataset_h5_cache(s);
        std::free(state->cache_instance_dir);
        std::free(state->cache_manifest_path);
        state->cache_instance_dir = 0;
        state->cache_manifest_path = 0;
    }
    if (cache_root == 0 || *cache_root == 0) return 1;
    if (shard_storage_has_capability(s, shard_storage_cap_materialize_canonical_pack | shard_storage_cap_materialize_execution_pack)) {
        if (!ensure_directory_exists(cache_root)) return 0;
    } else if (!directory_exists(cache_root)) {
        return 0;
    }
    return assign_owned_string(&state->cache_root, cache_root);
}

int get_dataset_h5_execution_metadata(const shard_storage *s,
                                      dataset_execution_view *execution) {
    const dataset_h5_state *state = dataset_h5_state_from_storage(s);
    if (execution == 0) return 0;
    std::memset(execution, 0, sizeof(*execution));
    if (state == 0) return 0;
    execution->partition_count = (std::uint32_t) state->num_partitions;
    execution->partition_execution_formats = state->partition_execution_formats;
    execution->partition_blocked_ell_block_sizes = state->partition_blocked_ell_block_sizes;
    execution->partition_blocked_ell_bucket_counts = state->partition_blocked_ell_bucket_counts;
    execution->partition_blocked_ell_fill_ratios = state->partition_blocked_ell_fill_ratios;
    execution->partition_execution_bytes = state->partition_execution_bytes;
    execution->partition_blocked_ell_bytes = state->partition_blocked_ell_bytes;
    execution->partition_bucketed_blocked_ell_bytes = state->partition_bucketed_blocked_ell_bytes;
    execution->partition_sliced_ell_slice_counts = state->partition_sliced_ell_slice_counts;
    execution->partition_sliced_ell_slice_rows = state->partition_sliced_ell_slice_rows;
    execution->partition_sliced_ell_bytes = state->partition_sliced_ell_bytes;
    execution->partition_bucketed_sliced_ell_bytes = state->partition_bucketed_sliced_ell_bytes;
    execution->shard_count = (std::uint32_t) state->num_shards;
    execution->shard_execution_formats = state->shard_execution_formats;
    execution->shard_blocked_ell_block_sizes = state->shard_blocked_ell_block_sizes;
    execution->shard_bucketed_partition_counts = state->shard_bucketed_partition_counts;
    execution->shard_bucketed_segment_counts = state->shard_bucketed_segment_counts;
    execution->shard_blocked_ell_fill_ratios = state->shard_blocked_ell_fill_ratios;
    execution->shard_execution_bytes = state->shard_execution_bytes;
    execution->shard_bucketed_blocked_ell_bytes = state->shard_bucketed_blocked_ell_bytes;
    execution->shard_sliced_ell_slice_counts = state->shard_sliced_ell_slice_counts;
    execution->shard_sliced_ell_slice_rows = state->shard_sliced_ell_slice_rows;
    execution->shard_bucketed_sliced_ell_bytes = state->shard_bucketed_sliced_ell_bytes;
    execution->shard_preferred_pair_ids = state->shard_preferred_pair_ids;
    execution->shard_owner_node_ids = state->shard_owner_node_ids;
    execution->shard_owner_rank_ids = state->shard_owner_rank_ids;
    execution->preferred_base_format = state->preferred_base_format;
    return 1;
}

int get_dataset_h5_runtime_service(const shard_storage *s,
                                   dataset_runtime_service_view *runtime_service) {
    const dataset_h5_state *state = dataset_h5_state_from_storage(s);
    if (runtime_service == 0) return 0;
    init(runtime_service);
    if (state == 0) return 0;
    *runtime_service = state->runtime_service;
    return 1;
}

int set_dataset_h5_cache_budget_bytes(shard_storage *s, std::uint64_t bytes) {
    dataset_h5_state *state = dataset_h5_state_from_storage(s);
    dataset_h5_cache_runtime *runtime = 0;
    if (state == 0) return 0;
    if (!require_storage_capability(s,
                                    shard_storage_cap_materialize_canonical_pack,
                                    "set dataset h5 cache budget")) {
        return 0;
    }
    if (!ensure_dataset_cache_layout(s)) return 0;
    runtime = cache_runtime(state);
    if (runtime == 0) return 0;
    {
        std::lock_guard<std::mutex> lock(runtime->state_mutex);
        state->cache_budget_bytes = bytes;
        state->cache_budget_explicit = 1;
        maybe_evict_cached_shards_locked(state, (unsigned long) state->num_shards);
    }
    return 1;
}

int set_dataset_h5_cache_predictor_enabled(shard_storage *s, int enabled) {
    dataset_h5_state *state = dataset_h5_state_from_storage(s);
    if (state == 0) return 0;
    state->predictor_enabled = enabled != 0 ? 1 : 0;
    return 1;
}

int pin_dataset_h5_cache_shard(shard_storage *s, unsigned long shard_id) {
    dataset_h5_state *state = dataset_h5_state_from_storage(s);
    dataset_h5_cache_runtime *runtime = 0;
    if (state == 0) return 0;
    if (!ensure_cached_shard_ready(s, shard_id)) return 0;
    runtime = cache_runtime(state);
    if (runtime == 0) return 0;
    {
        std::lock_guard<std::mutex> lock(runtime->state_mutex);
        state->shard_pin_count[shard_id] += 1u;
    }
    return 1;
}

int unpin_dataset_h5_cache_shard(shard_storage *s, unsigned long shard_id) {
    dataset_h5_state *state = dataset_h5_state_from_storage(s);
    dataset_h5_cache_runtime *runtime = 0;
    if (state == 0) return 0;
    if (!require_storage_capability(s,
                                    shard_storage_cap_materialize_canonical_pack,
                                    "unpin dataset h5 cache shard")) {
        return 0;
    }
    if (!ensure_dataset_cache_layout(s) || shard_id >= state->num_shards) return 0;
    runtime = cache_runtime(state);
    if (runtime == 0) return 0;
    {
        std::lock_guard<std::mutex> lock(runtime->state_mutex);
        if (state->shard_pin_count[shard_id] != 0u) state->shard_pin_count[shard_id] -= 1u;
        maybe_evict_cached_shards_locked(state, (unsigned long) state->num_shards);
    }
    return 1;
}

int evict_dataset_h5_cache_shard(shard_storage *s, unsigned long shard_id) {
    dataset_h5_state *state = dataset_h5_state_from_storage(s);
    dataset_h5_cache_runtime *runtime = 0;
    if (state == 0) return 0;
    if (!require_storage_capability(s,
                                    shard_storage_cap_materialize_canonical_pack,
                                    "evict dataset h5 cache shard")) {
        return 0;
    }
    if (!ensure_dataset_cache_layout(s) || shard_id >= state->num_shards) return 0;
    runtime = cache_runtime(state);
    if (runtime == 0) return 0;
    {
        std::lock_guard<std::mutex> lock(runtime->state_mutex);
        evict_cached_shard_locked(state, shard_id);
    }
    return 1;
}

int invalidate_dataset_h5_cache(shard_storage *s) {
    dataset_h5_state *state = dataset_h5_state_from_storage(s);
    dataset_h5_cache_runtime *runtime = 0;
    unsigned long shard_id = 0ul;
    char path[4096];
    if (state == 0) return 0;
    if (!require_storage_capability(s,
                                    shard_storage_cap_materialize_canonical_pack,
                                    "invalidate dataset h5 cache")) {
        return 0;
    }
    if (!ensure_dataset_cache_layout(s)) return 0;
    runtime = cache_runtime(state);
    if (runtime == 0) return 0;
    {
        std::lock_guard<std::mutex> lock(runtime->state_mutex);
        for (shard_id = 0ul; shard_id < (unsigned long) state->num_shards; ++shard_id) {
            evict_cached_shard_locked(state, shard_id);
            if (build_execution_pack_path(state, shard_id, path, sizeof(path))) ::unlink(path);
            if (build_execution_pack_temp_path(state, shard_id, path, sizeof(path))) ::unlink(path);
        }
        state->access_clock = 0u;
        state->last_requested_shard = std::numeric_limits<std::uint64_t>::max();
    }
    if (state->cache_manifest_path != 0) ::unlink(state->cache_manifest_path);
    return 1;
}

} // namespace cellshard
