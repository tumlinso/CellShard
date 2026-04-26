#include "shared.hh"

int fetch_dataset_blocked_ell_h5_partition(sharded<sparse::blocked_ell> *m,
                                           const shard_storage *s,
                                           unsigned long partition_id) {
    shard_storage *storage = const_cast<shard_storage *>(s);
    dataset_h5_state *state = dataset_h5_state_from_storage(storage);
    const unsigned long shard_id = state != 0 && state->partition_shard_ids != 0
        ? (unsigned long) state->partition_shard_ids[partition_id]
        : 0ul;
    if (m == 0 || storage == 0 || state == 0 || partition_id >= m->num_partitions) return 0;
    if (ensure_cspack_ready(storage, state, shard_id)) {
        return load_blocked_ell_part_from_cspack(m, state, partition_id);
    }
    if (!shard_storage_has_capability(storage, shard_storage_cap_canonical_read) || !open_dataset_h5_backend(storage)) return 0;
    if (blocked_ell_uses_execution_payload(state)) return load_blocked_ell_part_from_optimized_shard(m, state, partition_id);
    return load_or_materialize_blocked_ell_parts(m, state, shard_id, partition_id, partition_id + 1u, 1, 0, 0);
}

int fetch_dataset_blocked_ell_h5_shard(sharded<sparse::blocked_ell> *m,
                                       const shard_storage *s,
                                       unsigned long shard_id) {
    unsigned long begin = 0ul;
    unsigned long end = 0ul;
    shard_storage *storage = const_cast<shard_storage *>(s);
    dataset_h5_state *state = dataset_h5_state_from_storage(storage);
    if (m == 0 || storage == 0 || state == 0 || shard_id >= m->num_shards) return 0;
    begin = first_partition_in_shard(m, shard_id);
    end = last_partition_in_shard(m, shard_id);
    if (ensure_cspack_ready(storage, state, shard_id)) {
        for (unsigned long i = begin; i < end; ++i) {
            if (!load_blocked_ell_part_from_cspack(m, state, i)) return 0;
        }
        return 1;
    }
    if (!shard_storage_has_capability(storage, shard_storage_cap_canonical_read) || !open_dataset_h5_backend(storage)) return 0;
    if (blocked_ell_uses_execution_payload(state)) {
        for (unsigned long i = begin; i < end; ++i) {
            if (!load_blocked_ell_part_from_optimized_shard(m, state, i)) return 0;
        }
        return 1;
    }
    if (!load_or_materialize_blocked_ell_parts(m, state, shard_id, begin, end, 1, 0, 0)) return 0;
    for (unsigned long i = begin; i < end; ++i) {
        if (m->parts[i] == nullptr) return 0;
    }
    return 1;
}

int fetch_dataset_quantized_blocked_ell_h5_partition(sharded<sparse::quantized_blocked_ell> *m,
                                                     const shard_storage *s,
                                                     unsigned long partition_id) {
    return fetch_cached_partition_common(m, s, partition_id, load_quantized_blocked_ell_part_from_cspack);
}

int fetch_dataset_quantized_blocked_ell_h5_shard(sharded<sparse::quantized_blocked_ell> *m,
                                                 const shard_storage *s,
                                                 unsigned long shard_id) {
    return fetch_cached_shard_common(m, s, shard_id, load_quantized_blocked_ell_part_from_cspack);
}

int fetch_dataset_sliced_ell_h5_partition(sharded<sparse::sliced_ell> *m,
                                          const shard_storage *s,
                                          unsigned long partition_id) {
    shard_storage *storage = const_cast<shard_storage *>(s);
    dataset_h5_state *state = dataset_h5_state_from_storage(storage);
    const unsigned long shard_id = state != 0 && state->partition_shard_ids != 0
        ? (unsigned long) state->partition_shard_ids[partition_id]
        : 0ul;
    if (m == 0 || storage == 0 || state == 0 || partition_id >= m->num_partitions) return 0;
    if (!ensure_cspack_ready(storage, state, shard_id)) return 0;
    return load_sliced_ell_part_from_cspack(m, state, partition_id);
}

int fetch_dataset_sliced_ell_h5_shard(sharded<sparse::sliced_ell> *m,
                                      const shard_storage *s,
                                      unsigned long shard_id) {
    const unsigned long begin = first_partition_in_shard(m, shard_id);
    const unsigned long end = last_partition_in_shard(m, shard_id);
    shard_storage *storage = const_cast<shard_storage *>(s);
    dataset_h5_state *state = dataset_h5_state_from_storage(storage);
    if (m == 0 || storage == 0 || state == 0 || shard_id >= m->num_shards) return 0;
    if (!ensure_cspack_ready(storage, state, shard_id)) return 0;
    for (unsigned long i = begin; i < end; ++i) {
        if (!load_sliced_ell_part_from_cspack(m, state, i)) return 0;
    }
    return 1;
}

#if CELLSHARD_ENABLE_CUDA
int acquire_dataset_sliced_ell_h5_bucketed_partition_device(dataset_sliced_bucketed_device_partition_view *out,
                                                            const sharded<sparse::sliced_ell> *m,
                                                            const shard_storage *s,
                                                            unsigned long partition_id,
                                                            int device_id,
                                                            std::uint64_t cache_budget_bytes) {
    dataset_runtime_service_view runtime_service{};
    bucketed_sliced_ell_partition fetched;
    device::partition_record<sparse::sliced_ell> *uploaded_segments = 0;
    std::uint64_t expected_bytes = 0u;
    std::uint64_t uploaded_bytes = 0u;
    sliced_execution_device_cache_state *state = 0;
    std::lock_guard<std::mutex> lock(sliced_execution_device_caches_mutex());

    init(&runtime_service);
    init(&fetched);
    if (out == 0 || m == 0 || s == 0 || s->source_path == 0 || partition_id >= m->num_partitions) return 0;
    init(out);
    if (!get_dataset_h5_runtime_service(s, &runtime_service)) return 0;

    state = find_sliced_execution_device_cache(s->source_path, device_id);
    if (state == 0) {
        sliced_execution_device_cache_state fresh;
        fresh.source_path = s->source_path;
        fresh.device_id = device_id;
        sliced_execution_device_caches().push_back(std::move(fresh));
        state = &sliced_execution_device_caches().back();
    }

    state->byte_budget = cache_budget_bytes != 0u
        ? cache_budget_bytes
        : default_sliced_execution_device_cache_budget(device_id);
    if (!reserve_sliced_execution_device_cache_capacity(state, (std::size_t) m->num_partitions)) return 0;

    if (state->execution_plan_generation != runtime_service.execution_plan_generation
        || state->pack_generation != runtime_service.pack_generation
        || state->service_epoch != runtime_service.service_epoch) {
        clear_sliced_execution_device_cache_state(state);
        if (!reserve_sliced_execution_device_cache_capacity(state, (std::size_t) m->num_partitions)) return 0;
        state->execution_plan_generation = runtime_service.execution_plan_generation;
        state->pack_generation = runtime_service.pack_generation;
        state->service_epoch = runtime_service.service_epoch;
    }

    {
        sliced_execution_device_cache_entry &entry = state->entries[(std::size_t) partition_id];
        if (entry.valid) {
            entry.last_use_tick = ++state->use_tick;
            out->partition_id = partition_id;
            out->segment_count = entry.host_partition.segment_count;
            out->host_partition = &entry.host_partition;
            out->device_segments = entry.device_segments;
            out->resident_bytes = entry.resident_bytes;
            out->execution_plan_generation = state->execution_plan_generation;
            out->pack_generation = state->pack_generation;
            out->service_epoch = state->service_epoch;
            return 1;
        }
    }

    if (!fetch_dataset_sliced_ell_h5_bucketed_partition(&fetched, m, s, partition_id)) return 0;
    expected_bytes = bucketed_sliced_device_bytes(&fetched);
    while (state->byte_budget != 0u
           && state->resident_bytes != 0u
           && state->resident_bytes + expected_bytes > state->byte_budget) {
        if (!evict_one_sliced_execution_device_cache_entry(state, partition_id)) break;
    }

    uploaded_segments = fetched.segment_count != 0u
        ? (device::partition_record<sparse::sliced_ell> *) std::calloc((std::size_t) fetched.segment_count,
                                                                        sizeof(device::partition_record<sparse::sliced_ell>))
        : 0;
    if (fetched.segment_count != 0u && uploaded_segments == 0) goto fail;
    if (cudaSetDevice(device_id) != cudaSuccess) goto fail;
    for (std::uint32_t segment = 0u; segment < fetched.segment_count; ++segment) {
        if (device::upload(fetched.segments + segment, uploaded_segments + segment) != cudaSuccess) goto fail;
        uploaded_bytes += (std::uint64_t) uploaded_segments[segment].allocation_bytes;
    }

    {
        sliced_execution_device_cache_entry &entry = state->entries[(std::size_t) partition_id];
        if (entry.valid) state->resident_bytes -= entry.resident_bytes;
        clear_sliced_execution_device_cache_entry(&entry, state->device_id);
        entry.host_partition = fetched;
        init(&fetched);
        entry.device_segments = uploaded_segments;
        uploaded_segments = 0;
        entry.resident_bytes = uploaded_bytes != 0u ? uploaded_bytes : expected_bytes;
        entry.last_use_tick = ++state->use_tick;
        entry.valid = 1;
        state->resident_bytes += entry.resident_bytes;

        out->partition_id = partition_id;
        out->segment_count = entry.host_partition.segment_count;
        out->host_partition = &entry.host_partition;
        out->device_segments = entry.device_segments;
        out->resident_bytes = entry.resident_bytes;
        out->execution_plan_generation = state->execution_plan_generation;
        out->pack_generation = state->pack_generation;
        out->service_epoch = state->service_epoch;
    }
    return 1;

fail:
    if (uploaded_segments != 0) {
        (void) cudaSetDevice(device_id >= 0 ? device_id : 0);
        for (std::uint32_t segment = 0u; segment < fetched.segment_count; ++segment) {
            (void) device::release(uploaded_segments + segment);
        }
    }
    std::free(uploaded_segments);
    clear(&fetched);
    return 0;
}

int release_dataset_sliced_ell_h5_bucketed_partition_device(dataset_sliced_bucketed_device_partition_view *view) {
    if (view == 0) return 0;
    init(view);
    return 1;
}

int clear_dataset_sliced_ell_h5_bucketed_device_cache(const char *source_path,
                                                      int device_id) {
    std::lock_guard<std::mutex> lock(sliced_execution_device_caches_mutex());
    if (source_path == 0) return 0;
    if (sliced_execution_device_cache_state *state = find_sliced_execution_device_cache(source_path, device_id)) {
        clear_sliced_execution_device_cache_state(state);
        return 1;
    }
    return 0;
}

int clear_all_dataset_sliced_ell_h5_bucketed_device_caches() {
    std::lock_guard<std::mutex> lock(sliced_execution_device_caches_mutex());
    std::vector<sliced_execution_device_cache_state> &caches = sliced_execution_device_caches();
    for (std::size_t i = 0u; i < caches.size(); ++i) clear_sliced_execution_device_cache_state(&caches[i]);
    return 1;
}
#endif

int prefetch_dataset_blocked_ell_h5_partition_cache(const sharded<sparse::blocked_ell> *m,
                                                    shard_storage *s,
                                                    unsigned long partition_id) {
    dataset_h5_state *state = dataset_h5_state_from_storage(s);
    const unsigned long shard_id = state != 0 && state->partition_shard_ids != 0
        ? (unsigned long) state->partition_shard_ids[partition_id]
        : 0ul;
    if (m == 0 || s == 0 || state == 0 || partition_id >= m->num_partitions) return 0;
    return ensure_cspack_ready(s, state, shard_id);
}

int prefetch_dataset_blocked_ell_h5_shard_cache(const sharded<sparse::blocked_ell> *m,
                                                shard_storage *s,
                                                unsigned long shard_id) {
    dataset_h5_state *state = dataset_h5_state_from_storage(s);
    if (m == 0 || s == 0 || state == 0 || shard_id >= m->num_shards) return 0;
    return ensure_cspack_ready(s, state, shard_id);
}

int prefetch_dataset_quantized_blocked_ell_h5_partition_cache(const sharded<sparse::quantized_blocked_ell> *m,
                                                              shard_storage *s,
                                                              unsigned long partition_id) {
    return prefetch_partition_cache_common(m, s, partition_id);
}

int prefetch_dataset_quantized_blocked_ell_h5_shard_cache(const sharded<sparse::quantized_blocked_ell> *m,
                                                          shard_storage *s,
                                                          unsigned long shard_id) {
    return prefetch_shard_cache_common(m, s, shard_id);
}

int prefetch_dataset_sliced_ell_h5_partition_cache(const sharded<sparse::sliced_ell> *m,
                                                   shard_storage *s,
                                                   unsigned long partition_id) {
    dataset_h5_state *state = dataset_h5_state_from_storage(s);
    const unsigned long shard_id = state != 0 && state->partition_shard_ids != 0
        ? (unsigned long) state->partition_shard_ids[partition_id]
        : 0ul;
    if (m == 0 || s == 0 || state == 0 || partition_id >= m->num_partitions) return 0;
    return ensure_cspack_ready(s, state, shard_id);
}

int prefetch_dataset_sliced_ell_h5_shard_cache(const sharded<sparse::sliced_ell> *m,
                                               shard_storage *s,
                                               unsigned long shard_id) {
    dataset_h5_state *state = dataset_h5_state_from_storage(s);
    if (m == 0 || s == 0 || state == 0 || shard_id >= m->num_shards) return 0;
    return ensure_cspack_ready(s, state, shard_id);
}

int fetch_dataset_sliced_ell_h5_bucketed_partition(bucketed_sliced_ell_partition *out,
                                                   const sharded<sparse::sliced_ell> *m,
                                                   const shard_storage *s,
                                                   unsigned long partition_id) {
    shard_storage *storage = const_cast<shard_storage *>(s);
    dataset_h5_state *state = dataset_h5_state_from_storage(storage);
    unsigned long shard_id = 0ul;

    if (out == 0 || m == 0 || state == 0 || partition_id >= m->num_partitions) return 0;
    shard_id = (unsigned long) state->partition_shard_ids[partition_id];
    if (!ensure_cspack_ready(storage, state, shard_id)) return 0;
    if (load_sliced_bucketed_partition_from_cspack(state, shard_id, partition_id, out)) return 1;
    if (!shard_storage_has_capability(storage, shard_storage_cap_canonical_read)) return 0;
    return load_bucketed_sliced_ell_partition_payload(state, partition_id, out);
}

int fetch_dataset_blocked_ell_h5_pack_partition(bucketed_blocked_ell_partition *out,
                                                     const sharded<sparse::blocked_ell> *m,
                                                     const shard_storage *s,
                                                     unsigned long partition_id) {
    shard_storage *storage = const_cast<shard_storage *>(s);
    dataset_h5_state *state = dataset_h5_state_from_storage(storage);
    unsigned long shard_id = 0ul;

    if (out == 0 || m == 0 || state == 0 || partition_id >= m->num_partitions) return 0;
    shard_id = (unsigned long) state->partition_shard_ids[partition_id];
    if (!ensure_cspack_ready(storage, state, shard_id)) return 0;
    if (load_blocked_pack_partition_from_cspack(state, shard_id, partition_id, out)) return 1;
    if (!shard_storage_has_capability(storage, shard_storage_cap_materialize_pack | shard_storage_cap_canonical_read)) {
        return 0;
    }
    if (blocked_ell_uses_execution_payload(state)) {
        const std::uint64_t begin = state->shard_part_begin[shard_id];
        const std::uint64_t local_partition = partition_id - begin;
        if (!open_dataset_h5_backend(storage) || !load_optimized_blocked_ell_shard_payload(state, shard_id)) return 0;
        if (local_partition >= state->loaded_optimized_shard.partition_count) return 0;
        return clone_bucketed_partition(out, state->loaded_optimized_shard.partitions + local_partition);
    }
    if (!fetch_dataset_blocked_ell_h5_partition(const_cast<sharded<sparse::blocked_ell> *>(m), storage, partition_id)) return 0;
    return build_bucketed_execution_partition(out,
                                              m->parts[partition_id],
                                              state->partition_blocked_ell_bucket_counts != 0
                                                  ? state->partition_blocked_ell_bucket_counts[partition_id]
                                                  : 1u,
                                              0);
}

int build_bucketed_blocked_ell_partition(bucketed_blocked_ell_partition *out,
                                         const sparse::blocked_ell *part,
                                         std::uint32_t requested_bucket_count,
                                         std::uint64_t *bucketed_bytes_out) {
    return build_bucketed_execution_partition(out, part, requested_bucket_count, bucketed_bytes_out);
}

int build_bucketed_sliced_ell_partition(bucketed_sliced_ell_partition *out,
                                        const sparse::sliced_ell *part,
                                        std::uint32_t requested_bucket_count,
                                        std::uint64_t *bucketed_bytes_out) {
    return build_bucketed_sliced_execution_partition(out, part, requested_bucket_count, bucketed_bytes_out);
}

} // namespace cellshard
