#pragma once

inline int load_sliced_execution_partition_blob(std::FILE *fp, bucketed_sliced_ell_partition *part);

inline int ensure_blocked_ell_payload_open(dataset_h5_state *state) {
    if (state == 0 || state->file < 0) return 0;
    if (state->payload_blocked_ell >= 0
        && state->d_blocked_ell_block_idx >= 0
        && state->d_blocked_ell_values >= 0) {
        return 1;
    }
    if (state->payload_blocked_ell < 0) {
        state->payload_blocked_ell = H5Gopen2(state->file, payload_blocked_ell_group, H5P_DEFAULT);
        if (state->payload_blocked_ell < 0) return 0;
    }
    if (state->d_blocked_ell_block_idx < 0) {
        state->d_blocked_ell_block_idx = H5Dopen2(state->payload_blocked_ell, "block_col_idx", H5P_DEFAULT);
        if (state->d_blocked_ell_block_idx < 0) return 0;
    }
    if (state->d_blocked_ell_values < 0) {
        state->d_blocked_ell_values = H5Dopen2(state->payload_blocked_ell, "values", H5P_DEFAULT);
        if (state->d_blocked_ell_values < 0) return 0;
    }
    return 1;
}

inline int ensure_sliced_ell_payload_open(dataset_h5_state *state) {
    if (state == 0 || state->file < 0) return 0;
    if (state->payload_sliced_ell >= 0) return 1;
    state->payload_sliced_ell = H5Gopen2(state->file, payload_sliced_ell_group, H5P_DEFAULT);
    return state->payload_sliced_ell >= 0;
}

inline int ensure_quantized_blocked_ell_payload_open(dataset_h5_state *state) {
    if (state == 0 || state->file < 0) return 0;
    if (state->payload_quantized_blocked_ell >= 0) return 1;
    state->payload_quantized_blocked_ell = H5Gopen2(state->file, payload_quantized_blocked_ell_group, H5P_DEFAULT);
    return state->payload_quantized_blocked_ell >= 0;
}

inline int ensure_optimized_blocked_ell_payload_open(dataset_h5_state *state) {
    if (state == 0 || state->file < 0) return 0;
    if (state->payload_optimized_blocked_ell >= 0) return 1;
    state->payload_optimized_blocked_ell = H5Gopen2(state->file, payload_blocked_ell_group, H5P_DEFAULT);
    if (state->payload_optimized_blocked_ell < 0) {
        state->payload_optimized_blocked_ell = H5Gopen2(state->file, payload_optimized_blocked_ell_group, H5P_DEFAULT);
    }
    return state->payload_optimized_blocked_ell >= 0;
}

inline int deserialize_optimized_shard(const unsigned char *data,
                                       std::size_t bytes,
                                       bucketed_blocked_ell_shard *shard);

inline int deserialize_optimized_sliced_shard(const unsigned char *data,
                                              std::size_t bytes,
                                              bucketed_sliced_ell_shard *shard);

inline int load_blocked_ell_shard_payload(dataset_h5_state *state,
                                          std::uint64_t shard_id) {
    const std::uint64_t idx_begin = state != 0 && state->shard_block_idx_offsets != 0 ? state->shard_block_idx_offsets[shard_id] : 0u;
    const std::uint64_t idx_end = state != 0 && state->shard_block_idx_offsets != 0 ? state->shard_block_idx_offsets[shard_id + 1u] : 0u;
    const std::uint64_t value_begin = state != 0 && state->shard_value_offsets != 0 ? state->shard_value_offsets[shard_id] : 0u;
    const std::uint64_t value_end = state != 0 && state->shard_value_offsets != 0 ? state->shard_value_offsets[shard_id + 1u] : 0u;
    const std::size_t block_idx_count = idx_end >= idx_begin ? (std::size_t) (idx_end - idx_begin) : 0u;
    const std::size_t value_count = value_end >= value_begin ? (std::size_t) (value_end - value_begin) : 0u;

    if (state == 0 || shard_id >= state->num_shards) return 0;
    if (state->loaded_blocked_ell_shard_id == shard_id) return 1;
    if (!ensure_blocked_ell_payload_open(state)) return 0;
    if (!reserve_blocked_ell_shard_scratch(state, block_idx_count, value_count)) return 0;
    if (block_idx_count != 0u
        && !read_hyperslab_1d(state->d_blocked_ell_block_idx,
                              H5T_NATIVE_UINT32,
                              idx_begin,
                              (std::uint64_t) block_idx_count,
                              state->blocked_ell_block_idx_scratch)) {
        return 0;
    }
    if (value_count != 0u
        && !read_hyperslab_1d(state->d_blocked_ell_values,
                              H5T_NATIVE_UINT16,
                              value_begin,
                              (std::uint64_t) value_count,
                              state->blocked_ell_value_scratch)) {
        return 0;
    }
    state->loaded_blocked_ell_shard_id = shard_id;
    return 1;
}

inline int load_optimized_blocked_ell_shard_payload(dataset_h5_state *state,
                                                    std::uint64_t shard_id) {
    std::vector<unsigned char> blob;
    char dataset_name[64];
    if (state == 0 || shard_id >= state->num_shards) return 0;
    if (state->loaded_optimized_shard_id == shard_id) return 1;
    if (!ensure_optimized_blocked_ell_payload_open(state)) return 0;
    if (!build_optimized_shard_dataset_name((unsigned long) shard_id, dataset_name, sizeof(dataset_name))) return 0;
    if (!read_blob_dataset(state->payload_optimized_blocked_ell, dataset_name, &blob)) return 0;
    clear(&state->loaded_optimized_shard);
    init(&state->loaded_optimized_shard);
    if (!deserialize_optimized_shard(blob.data(), blob.size(), &state->loaded_optimized_shard)) {
        std::fprintf(stderr, "cellshard: failed to deserialize optimized blocked shard %llu\n", (unsigned long long) shard_id);
        return 0;
    }
    state->loaded_optimized_shard_id = shard_id;
    return 1;
}

inline int load_bucketed_sliced_ell_partition_payload(dataset_h5_state *state,
                                                      unsigned long partition_id,
                                                      bucketed_sliced_ell_partition *part) {
    std::vector<unsigned char> blob;
    char dataset_name[64];
    std::FILE *fp = 0;
    int ok = 0;

    if (state == 0 || part == 0 || partition_id >= state->num_partitions) return 0;
    if (!ensure_sliced_ell_payload_open(state)) return 0;
    if (!build_partition_blob_dataset_name(partition_id, dataset_name, sizeof(dataset_name))) return 0;
    if (!read_blob_dataset(state->payload_sliced_ell, dataset_name, &blob)) return 0;
    fp = fmemopen(blob.data(), blob.size(), "rb");
    if (fp == 0) return 0;
    clear(part);
    init(part);
    ok = load_sliced_execution_partition_blob(fp, part);
    std::fclose(fp);
    return ok;
}

inline int load_quantized_blocked_ell_partition_payload(dataset_h5_state *state,
                                                        unsigned long partition_id,
                                                        sparse::quantized_blocked_ell *part) {
    std::vector<unsigned char> blob;
    char dataset_name[64];
    std::FILE *fp = 0;
    int ok = 0;

    if (state == 0 || part == 0 || partition_id >= state->num_partitions) return 0;
    if (!ensure_quantized_blocked_ell_payload_open(state)) return 0;
    if (!build_partition_blob_dataset_name(partition_id, dataset_name, sizeof(dataset_name))) return 0;
    if (!read_blob_dataset(state->payload_quantized_blocked_ell, dataset_name, &blob)) return 0;
    fp = fmemopen(blob.data(), blob.size(), "rb");
    if (fp == 0) return 0;
    sparse::clear(part);
    sparse::init(part);
    ok = ::cellshard::load(fp, part);
    std::fclose(fp);
    return ok;
}

inline int prepare_blocked_ell_parts_from_state(const dataset_h5_state *state,
                                                unsigned long begin,
                                                unsigned long end,
                                                sparse::blocked_ell **parts_out) {
    unsigned long partition_id = 0;

    if (state == 0 || parts_out == 0 || begin > end || end > state->num_partitions) return 0;
    for (partition_id = begin; partition_id < end; ++partition_id) {
        const unsigned long aux = (unsigned long) state->partition_aux[partition_id];
        const types::u32 block_size = sparse::unpack_blocked_ell_block_size(aux);
        const types::u32 ell_cols = sparse::unpack_blocked_ell_cols(aux);
        sparse::blocked_ell *part = new sparse::blocked_ell;
        sparse::init(part,
                     (types::dim_t) state->partition_rows[partition_id],
                     (types::dim_t) state->cols,
                     (types::nnz_t) state->partition_nnz[partition_id],
                     block_size,
                     ell_cols);
        if (!sparse::allocate(part)) {
            sparse::clear(part);
            delete part;
            return 0;
        }
        parts_out[partition_id - begin] = part;
    }
    return 1;
}

inline void clear_blocked_ell_parts(sparse::blocked_ell **parts, unsigned long count) {
    unsigned long i = 0;
    if (parts == 0) return;
    for (i = 0; i < count; ++i) {
        if (parts[i] != 0) {
            sparse::clear(parts[i]);
            delete parts[i];
            parts[i] = 0;
        }
    }
}

inline int fill_blocked_ell_parts_from_loaded_shard(const dataset_h5_state *state,
                                                    unsigned long shard_id,
                                                    unsigned long begin,
                                                    unsigned long end,
                                                    sparse::blocked_ell **parts) {
    const unsigned long partition_count = end - begin;
    const std::uint64_t shard_idx_base = state->shard_block_idx_offsets[shard_id];
    const std::uint64_t shard_value_base = state->shard_value_offsets[shard_id];
    unsigned long local = 0;

    if (state == 0 || parts == 0 || begin > end || end > state->num_partitions) return 0;
    if (state->loaded_blocked_ell_shard_id != shard_id) return 0;

#pragma omp parallel for if(partition_count >= 4ul)
    for (local = 0; local < partition_count; ++local) {
        const unsigned long partition_id = begin + local;
        const unsigned long aux = (unsigned long) state->partition_aux[partition_id];
        const std::size_t block_idx_count = blocked_ell_part_block_index_count(state->partition_rows[partition_id], aux);
        const std::size_t value_count = blocked_ell_part_value_count(state->partition_rows[partition_id], aux);
        const std::uint64_t idx_offset = state->partition_block_idx_offsets[partition_id] - shard_idx_base;
        const std::uint64_t value_offset = state->partition_value_offsets[partition_id] - shard_value_base;

        if (block_idx_count != 0u) {
            std::memcpy(parts[local]->blockColIdx,
                        state->blocked_ell_block_idx_scratch + idx_offset,
                        block_idx_count * sizeof(types::idx_t));
        }
        if (value_count != 0u) {
            std::memcpy(parts[local]->val,
                        state->blocked_ell_value_scratch + value_offset,
                        value_count * sizeof(real::storage_t));
        }
    }

    return 1;
}

inline int duplicate_u32_array(std::uint32_t **dst,
                               const std::uint32_t *src,
                               std::size_t count) {
    std::uint32_t *copy = 0;
    if (dst == 0) return 0;
    *dst = 0;
    if (count == 0u) return 1;
    copy = (std::uint32_t *) std::malloc(count * sizeof(std::uint32_t));
    if (copy == 0) return 0;
    std::memcpy(copy, src, count * sizeof(std::uint32_t));
    *dst = copy;
    return 1;
}

inline int assign_partition_col_maps(bucketed_blocked_ell_partition *part,
                                     const std::uint32_t *exec_to_canonical_cols,
                                     const std::uint32_t *canonical_to_exec_cols,
                                     std::uint32_t cols) {
    if (part == 0) return 0;
    std::free(part->exec_to_canonical_cols);
    std::free(part->canonical_to_exec_cols);
    part->exec_to_canonical_cols = 0;
    part->canonical_to_exec_cols = 0;
    if (cols == 0u) return 1;
    if (!duplicate_u32_array(&part->exec_to_canonical_cols, exec_to_canonical_cols, cols)) return 0;
    if (!duplicate_u32_array(&part->canonical_to_exec_cols, canonical_to_exec_cols, cols)) {
        std::free(part->exec_to_canonical_cols);
        part->exec_to_canonical_cols = 0;
        return 0;
    }
    return 1;
}

inline int clone_bucketed_partition(bucketed_blocked_ell_partition *dst,
                                    const bucketed_blocked_ell_partition *src) {
    std::uint32_t segment = 0u;
    if (dst == 0 || src == 0) return 0;
    clear(dst);
    init(dst);
    dst->rows = src->rows;
    dst->cols = src->cols;
    dst->nnz = src->nnz;
    dst->segment_count = src->segment_count;
    dst->segments = dst->segment_count != 0u ? (sparse::blocked_ell *) std::calloc((std::size_t) dst->segment_count, sizeof(sparse::blocked_ell)) : 0;
    dst->segment_row_offsets = (std::uint32_t *) std::calloc((std::size_t) dst->segment_count + 1u, sizeof(std::uint32_t));
    if ((dst->segment_count != 0u && (dst->segments == 0 || dst->segment_row_offsets == 0))
        || !duplicate_u32_array(&dst->exec_to_canonical_rows, src->exec_to_canonical_rows, src->rows)
        || !duplicate_u32_array(&dst->canonical_to_exec_rows, src->canonical_to_exec_rows, src->rows)
        || !duplicate_u32_array(&dst->exec_to_canonical_cols, src->exec_to_canonical_cols, src->cols)
        || !duplicate_u32_array(&dst->canonical_to_exec_cols, src->canonical_to_exec_cols, src->cols)) {
        clear(dst);
        return 0;
    }
    if (dst->segment_count != 0u) {
        std::memcpy(dst->segment_row_offsets,
                    src->segment_row_offsets,
                    ((std::size_t) dst->segment_count + 1u) * sizeof(std::uint32_t));
    }
    for (segment = 0u; segment < dst->segment_count; ++segment) {
        sparse::init(dst->segments + segment,
                     src->segments[segment].rows,
                     src->segments[segment].cols,
                     src->segments[segment].nnz,
                     src->segments[segment].block_size,
                     src->segments[segment].ell_cols);
        if (!sparse::allocate(dst->segments + segment)) {
            clear(dst);
            return 0;
        }
        if (src->segments[segment].blockColIdx != 0) {
            std::memcpy(dst->segments[segment].blockColIdx,
                        src->segments[segment].blockColIdx,
                        blocked_ell_part_block_index_count(src->segments[segment].rows,
                                                           ::cellshard::partition_aux(src->segments + segment))
                            * sizeof(types::idx_t));
        }
        if (src->segments[segment].val != 0) {
            std::memcpy(dst->segments[segment].val,
                        src->segments[segment].val,
                        (std::size_t) src->segments[segment].rows * src->segments[segment].ell_cols * sizeof(real::storage_t));
        }
    }
    return 1;
}

inline int clone_bucketed_sliced_partition(bucketed_sliced_ell_partition *dst,
                                           const bucketed_sliced_ell_partition *src) {
    std::uint32_t segment = 0u;
    if (dst == 0 || src == 0) return 0;
    clear(dst);
    init(dst);
    dst->rows = src->rows;
    dst->cols = src->cols;
    dst->nnz = src->nnz;
    dst->segment_count = src->segment_count;
    dst->canonical_slice_count = src->canonical_slice_count;
    dst->segments = dst->segment_count != 0u ? (sparse::sliced_ell *) std::calloc((std::size_t) dst->segment_count, sizeof(sparse::sliced_ell)) : 0;
    dst->segment_row_offsets = (std::uint32_t *) std::calloc((std::size_t) dst->segment_count + 1u, sizeof(std::uint32_t));
    dst->canonical_slice_row_offsets =
        (std::uint32_t *) std::calloc((std::size_t) dst->canonical_slice_count + 1u, sizeof(std::uint32_t));
    dst->canonical_slice_widths =
        dst->canonical_slice_count != 0u ? (std::uint32_t *) std::calloc((std::size_t) dst->canonical_slice_count, sizeof(std::uint32_t)) : 0;
    if ((dst->segment_count != 0u && (dst->segments == 0 || dst->segment_row_offsets == 0))
        || (((std::size_t) dst->canonical_slice_count + 1u) != 0u && dst->canonical_slice_row_offsets == 0)
        || (dst->canonical_slice_count != 0u && dst->canonical_slice_widths == 0)
        || !duplicate_u32_array(&dst->exec_to_canonical_rows, src->exec_to_canonical_rows, src->rows)
        || !duplicate_u32_array(&dst->canonical_to_exec_rows, src->canonical_to_exec_rows, src->rows)) {
        clear(dst);
        return 0;
    }
    if (dst->segment_count != 0u) {
        std::memcpy(dst->segment_row_offsets,
                    src->segment_row_offsets,
                    ((std::size_t) dst->segment_count + 1u) * sizeof(std::uint32_t));
    }
    if (dst->canonical_slice_row_offsets != 0) {
        std::memcpy(dst->canonical_slice_row_offsets,
                    src->canonical_slice_row_offsets,
                    ((std::size_t) dst->canonical_slice_count + 1u) * sizeof(std::uint32_t));
    }
    if (dst->canonical_slice_widths != 0) {
        std::memcpy(dst->canonical_slice_widths,
                    src->canonical_slice_widths,
                    (std::size_t) dst->canonical_slice_count * sizeof(std::uint32_t));
    }
    for (segment = 0u; segment < dst->segment_count; ++segment) {
        const sparse::sliced_ell *src_segment = src->segments + segment;
        sparse::init(dst->segments + segment, src_segment->rows, src_segment->cols, src_segment->nnz);
        if (!sparse::allocate(dst->segments + segment,
                              src_segment->slice_count,
                              src_segment->slice_row_offsets,
                              src_segment->slice_widths)) {
            clear(dst);
            return 0;
        }
        if (src_segment->col_idx != 0) {
            std::memcpy(dst->segments[segment].col_idx,
                        src_segment->col_idx,
                        sliced_ell_part_total_slots(::cellshard::partition_aux(src_segment)) * sizeof(types::idx_t));
        }
        if (src_segment->val != 0) {
            std::memcpy(dst->segments[segment].val,
                        src_segment->val,
                        sliced_ell_part_total_slots(::cellshard::partition_aux(src_segment)) * sizeof(real::storage_t));
        }
    }
    return 1;
}

#if CELLSHARD_ENABLE_CUDA
struct sliced_execution_device_cache_entry {
    bucketed_sliced_ell_partition host_partition;
    device::partition_record<sparse::sliced_ell> *device_segments;
    std::uint64_t resident_bytes;
    std::uint64_t last_use_tick;
    int valid;

    sliced_execution_device_cache_entry()
        : device_segments(0),
          resident_bytes(0u),
          last_use_tick(0u),
          valid(0) {
        init(&host_partition);
    }
};

struct sliced_execution_device_cache_state {
    std::string source_path;
    int device_id = -1;
    std::uint64_t byte_budget = 0u;
    std::uint64_t resident_bytes = 0u;
    std::uint64_t use_tick = 0u;
    std::uint64_t execution_plan_generation = 0u;
    std::uint64_t pack_generation = 0u;
    std::uint64_t service_epoch = 0u;
    std::vector<sliced_execution_device_cache_entry> entries;
};

inline std::size_t sliced_ell_device_bytes(const sparse::sliced_ell *src) {
    const std::size_t slice_offsets_bytes = src != 0 ? (std::size_t) (src->slice_count + 1u) * sizeof(unsigned int) : 0u;
    const std::size_t widths_offset = device::align_up_bytes(slice_offsets_bytes, alignof(unsigned int));
    const std::size_t widths_bytes = src != 0 ? (std::size_t) src->slice_count * sizeof(unsigned int) : 0u;
    const std::size_t slot_offsets_offset = device::align_up_bytes(widths_offset + widths_bytes, alignof(unsigned int));
    const std::size_t slot_offsets_bytes = src != 0 ? (std::size_t) src->slice_count * sizeof(unsigned int) : 0u;
    const std::size_t total_slots = src != 0 ? (std::size_t) sparse::total_slots(src) : 0u;
    const std::size_t col_offset = device::align_up_bytes(slot_offsets_offset + slot_offsets_bytes, alignof(unsigned int));
    const std::size_t col_bytes = total_slots * sizeof(unsigned int);
    const std::size_t val_offset = device::align_up_bytes(col_offset + col_bytes, alignof(__half));
    const std::size_t val_bytes = total_slots * sizeof(__half);
    const std::size_t payload_offset = device::align_up_bytes(sizeof(device::sliced_ell_view), alignof(unsigned int));
    return payload_offset + val_offset + val_bytes;
}

inline std::uint64_t bucketed_sliced_device_bytes(const bucketed_sliced_ell_partition *part) {
    std::uint64_t total = 0u;
    std::uint32_t segment = 0u;
    if (part == 0) return 0u;
    for (segment = 0u; segment < part->segment_count; ++segment) {
        total += (std::uint64_t) sliced_ell_device_bytes(part->segments + segment);
    }
    return total;
}

inline void clear_sliced_execution_device_cache_entry(sliced_execution_device_cache_entry *entry,
                                                      int device_id) {
    std::uint32_t segment = 0u;
    if (entry == 0) return;
    if (entry->device_segments != 0) {
        (void) cudaSetDevice(device_id >= 0 ? device_id : 0);
        for (segment = 0u; segment < entry->host_partition.segment_count; ++segment) {
            (void) device::release(entry->device_segments + segment);
        }
    }
    std::free(entry->device_segments);
    entry->device_segments = 0;
    clear(&entry->host_partition);
    entry->resident_bytes = 0u;
    entry->last_use_tick = 0u;
    entry->valid = 0;
}

inline void clear_sliced_execution_device_cache_state(sliced_execution_device_cache_state *state) {
    if (state == 0) return;
    for (std::size_t i = 0u; i < state->entries.size(); ++i) {
        clear_sliced_execution_device_cache_entry(&state->entries[i], state->device_id);
    }
    state->resident_bytes = 0u;
    state->use_tick = 0u;
    state->execution_plan_generation = 0u;
    state->pack_generation = 0u;
    state->service_epoch = 0u;
}

inline std::uint64_t default_sliced_execution_device_cache_budget(int device_id) {
    std::size_t free_bytes = 0u;
    std::size_t total_bytes = 0u;
    if (cudaSetDevice(device_id) != cudaSuccess) {
        cudaGetLastError();
        return 512ull << 20u;
    }
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        cudaGetLastError();
        return 512ull << 20u;
    }
    const std::uint64_t quarter_free = (std::uint64_t) free_bytes / 4u;
    const std::uint64_t cap = 2ull << 30u;
    const std::uint64_t floor = 512ull << 20u;
    if (quarter_free == 0u) return floor;
    return std::max<std::uint64_t>(floor, std::min<std::uint64_t>(quarter_free, cap));
}

inline std::vector<sliced_execution_device_cache_state> &sliced_execution_device_caches() {
    static std::vector<sliced_execution_device_cache_state> caches;
    return caches;
}

inline std::mutex &sliced_execution_device_caches_mutex() {
    static std::mutex mtx;
    return mtx;
}

inline sliced_execution_device_cache_state *find_sliced_execution_device_cache(const char *source_path,
                                                                               int device_id) {
    std::vector<sliced_execution_device_cache_state> &caches = sliced_execution_device_caches();
    for (std::size_t i = 0u; i < caches.size(); ++i) {
        if (caches[i].device_id == device_id && caches[i].source_path == (source_path != 0 ? source_path : "")) {
            return &caches[i];
        }
    }
    return 0;
}

inline int evict_one_sliced_execution_device_cache_entry(sliced_execution_device_cache_state *state,
                                                         unsigned long protected_partition_id) {
    unsigned long victim = (unsigned long) -1;
    if (state == 0) return 0;
    for (unsigned long part_id = 0u; part_id < state->entries.size(); ++part_id) {
        const sliced_execution_device_cache_entry &entry = state->entries[part_id];
        if (!entry.valid || part_id == protected_partition_id) continue;
        if (victim == (unsigned long) -1 || entry.last_use_tick < state->entries[victim].last_use_tick) {
            victim = part_id;
        }
    }
    if (victim == (unsigned long) -1) return 0;
    state->resident_bytes -= state->entries[victim].resident_bytes;
    clear_sliced_execution_device_cache_entry(&state->entries[victim], state->device_id);
    return 1;
}

inline int reserve_sliced_execution_device_cache_capacity(sliced_execution_device_cache_state *state,
                                                          std::size_t partition_count) {
    if (state == 0) return 0;
    if (state->entries.size() == partition_count) return 1;
    clear_sliced_execution_device_cache_state(state);
    state->entries.clear();
    state->entries.resize(partition_count);
    return 1;
}
#endif

inline int build_identity_col_maps(std::uint32_t cols,
                                   std::uint32_t **exec_to_canonical_cols,
                                   std::uint32_t **canonical_to_exec_cols) {
    std::uint32_t *exec_to_canonical = 0;
    std::uint32_t *canonical_to_exec = 0;
    std::uint32_t col = 0u;
    if (exec_to_canonical_cols == 0 || canonical_to_exec_cols == 0) return 0;
    *exec_to_canonical_cols = 0;
    *canonical_to_exec_cols = 0;
    if (cols == 0u) return 1;
    exec_to_canonical = (std::uint32_t *) std::malloc((std::size_t) cols * sizeof(std::uint32_t));
    canonical_to_exec = (std::uint32_t *) std::malloc((std::size_t) cols * sizeof(std::uint32_t));
    if (exec_to_canonical == 0 || canonical_to_exec == 0) {
        std::free(exec_to_canonical);
        std::free(canonical_to_exec);
        return 0;
    }
    for (col = 0u; col < cols; ++col) {
        exec_to_canonical[col] = col;
        canonical_to_exec[col] = col;
    }
    *exec_to_canonical_cols = exec_to_canonical;
    *canonical_to_exec_cols = canonical_to_exec;
    return 1;
}

inline int build_identity_u32_maps(std::uint32_t count,
                                   std::uint32_t **exec_to_canonical,
                                   std::uint32_t **canonical_to_exec) {
    std::uint32_t *forward = 0;
    std::uint32_t *inverse = 0;
    if (exec_to_canonical == 0 || canonical_to_exec == 0) return 0;
    *exec_to_canonical = 0;
    *canonical_to_exec = 0;
    if (count == 0u) return 1;
    forward = (std::uint32_t *) std::malloc((std::size_t) count * sizeof(std::uint32_t));
    inverse = (std::uint32_t *) std::malloc((std::size_t) count * sizeof(std::uint32_t));
    if (forward == 0 || inverse == 0) {
        std::free(forward);
        std::free(inverse);
        return 0;
    }
    for (std::uint32_t i = 0u; i < count; ++i) {
        forward[i] = i;
        inverse[i] = i;
    }
    *exec_to_canonical = forward;
    *canonical_to_exec = inverse;
    return 1;
}

struct shard_column_signature {
    std::uint32_t canonical_col;
    std::uint32_t support;
    std::uint64_t hash_a;
    std::uint64_t hash_b;
    std::uint32_t min_row_block;
};

inline std::uint64_t mix_signature(std::uint64_t seed, std::uint64_t value) {
    return fnv1a_mix(seed, &value, sizeof(value));
}

inline int build_shard_column_maps(sparse::blocked_ell *const *parts,
                                   std::uint32_t partition_count,
                                   std::uint32_t cols,
                                   std::uint32_t **exec_to_canonical_cols,
                                   std::uint32_t **canonical_to_exec_cols) {
    std::vector<shard_column_signature> signatures;
    std::uint32_t *exec_to_canonical = 0;
    std::uint32_t *canonical_to_exec = 0;
    std::uint32_t partition = 0u;
    std::uint32_t global_row_block = 0u;

    if (exec_to_canonical_cols == 0 || canonical_to_exec_cols == 0) return 0;
    *exec_to_canonical_cols = 0;
    *canonical_to_exec_cols = 0;
    if (cols == 0u) return 1;

    signatures.resize((std::size_t) cols);
    for (std::uint32_t col = 0u; col < cols; ++col) {
        signatures[(std::size_t) col].canonical_col = col;
        signatures[(std::size_t) col].support = 0u;
        signatures[(std::size_t) col].hash_a = 1469598103934665603ull;
        signatures[(std::size_t) col].hash_b = 1099511628211ull;
        signatures[(std::size_t) col].min_row_block = std::numeric_limits<std::uint32_t>::max();
    }

    for (partition = 0u; partition < partition_count; ++partition) {
        const sparse::blocked_ell *part = parts != 0 ? parts[partition] : 0;
        const std::uint32_t row_block_count = part != 0 ? sparse::row_block_count(part) : 0u;
        const std::uint32_t block_size = part != 0 ? part->block_size : 0u;
        const std::uint32_t width_blocks = part != 0 ? sparse::ell_width_blocks(part) : 0u;
        if (part == 0 || block_size == 0u) return 0;
        for (std::uint32_t row_block = 0u; row_block < row_block_count; ++row_block, ++global_row_block) {
            const std::uint32_t row_begin = row_block * block_size;
            const std::uint32_t rows_in_block =
                row_begin < part->rows ? std::min<std::uint32_t>(block_size, part->rows - row_begin) : 0u;
            for (std::uint32_t slot = 0u; slot < width_blocks; ++slot) {
                const types::idx_t block_col = part->blockColIdx[(std::size_t) row_block * width_blocks + slot];
                if (block_col == sparse::blocked_ell_invalid_col) continue;
                for (std::uint32_t col_in_block = 0u; col_in_block < block_size; ++col_in_block) {
                    const std::uint32_t col = (std::uint32_t) block_col * block_size + col_in_block;
                    bool seen = false;
                    if (col >= cols) continue;
                    for (std::uint32_t row_in_block = 0u; row_in_block < rows_in_block; ++row_in_block) {
                        const std::size_t offset =
                            (std::size_t) (row_block * block_size + row_in_block) * part->ell_cols
                            + (std::size_t) slot * block_size + col_in_block;
                        if (__half2float(part->val[offset]) != 0.0f) {
                            seen = true;
                            break;
                        }
                    }
                    if (!seen) continue;
                    signatures[(std::size_t) col].support += 1u;
                    signatures[(std::size_t) col].hash_a =
                        mix_signature(signatures[(std::size_t) col].hash_a, (std::uint64_t) global_row_block + 1u);
                    signatures[(std::size_t) col].hash_b =
                        mix_signature(signatures[(std::size_t) col].hash_b, ((std::uint64_t) global_row_block + 1u) * 1315423911ull);
                    signatures[(std::size_t) col].min_row_block =
                        std::min(signatures[(std::size_t) col].min_row_block, global_row_block);
                }
            }
        }
    }

    std::stable_sort(signatures.begin(),
                     signatures.end(),
                     [](const shard_column_signature &lhs, const shard_column_signature &rhs) {
                         if (lhs.support == 0u || rhs.support == 0u) {
                             if (lhs.support != rhs.support) return lhs.support > rhs.support;
                         }
                         if (lhs.min_row_block != rhs.min_row_block) return lhs.min_row_block < rhs.min_row_block;
                         if (lhs.hash_a != rhs.hash_a) return lhs.hash_a < rhs.hash_a;
                         if (lhs.hash_b != rhs.hash_b) return lhs.hash_b < rhs.hash_b;
                         if (lhs.support != rhs.support) return lhs.support > rhs.support;
                         return lhs.canonical_col < rhs.canonical_col;
                     });

    exec_to_canonical = (std::uint32_t *) std::malloc((std::size_t) cols * sizeof(std::uint32_t));
    canonical_to_exec = (std::uint32_t *) std::malloc((std::size_t) cols * sizeof(std::uint32_t));
    if (exec_to_canonical == 0 || canonical_to_exec == 0) {
        std::free(exec_to_canonical);
        std::free(canonical_to_exec);
        return 0;
    }
    for (std::uint32_t exec_col = 0u; exec_col < cols; ++exec_col) {
        const std::uint32_t canonical_col = signatures[(std::size_t) exec_col].canonical_col;
        exec_to_canonical[exec_col] = canonical_col;
        canonical_to_exec[canonical_col] = exec_col;
    }
    *exec_to_canonical_cols = exec_to_canonical;
    *canonical_to_exec_cols = canonical_to_exec;
    return 1;
}

inline int blocked_ell_to_canonical_coo(const sparse::blocked_ell *part,
                                        sparse::coo *out) {
    const std::uint32_t block_size = part != 0 ? part->block_size : 0u;
    const std::uint32_t width_blocks = part != 0 ? sparse::ell_width_blocks(part) : 0u;
    std::size_t actual_nnz = 0u;
    std::size_t emitted = 0u;
    if (part == 0 || out == 0) return 0;
    sparse::clear(out);
    for (std::uint32_t row = 0u; row < part->rows; ++row) {
        const std::uint32_t row_block = row / block_size;
        for (std::uint32_t slot = 0u; slot < width_blocks; ++slot) {
            const types::idx_t block_col = part->blockColIdx[(std::size_t) row_block * width_blocks + slot];
            if (block_col == sparse::blocked_ell_invalid_col) continue;
            for (std::uint32_t col_in_block = 0u; col_in_block < block_size; ++col_in_block) {
                const std::uint32_t col = (std::uint32_t) block_col * block_size + col_in_block;
                const real::storage_t value =
                    part->val[(std::size_t) row * part->ell_cols + (std::size_t) slot * block_size + col_in_block];
                if (__half2float(value) == 0.0f || col >= part->cols) continue;
                ++actual_nnz;
            }
        }
    }
    sparse::init(out, part->rows, part->cols, (types::nnz_t) actual_nnz);
    if (!sparse::allocate(out)) return 0;
    for (std::uint32_t row = 0u; row < part->rows; ++row) {
        const std::uint32_t row_block = row / block_size;
        for (std::uint32_t slot = 0u; slot < width_blocks; ++slot) {
            const types::idx_t block_col = part->blockColIdx[(std::size_t) row_block * width_blocks + slot];
            if (block_col == sparse::blocked_ell_invalid_col) continue;
            for (std::uint32_t col_in_block = 0u; col_in_block < block_size; ++col_in_block) {
                const std::uint32_t col = (std::uint32_t) block_col * block_size + col_in_block;
                const real::storage_t value =
                    part->val[(std::size_t) row * part->ell_cols + (std::size_t) slot * block_size + col_in_block];
                if (__half2float(value) == 0.0f || col >= part->cols) continue;
                out->rowIdx[emitted] = row;
                out->colIdx[emitted] = col;
                out->val[emitted] = value;
                ++emitted;
            }
        }
    }
    if (emitted != out->nnz) {
        sparse::clear(out);
        return 0;
    }
    return 1;
}

inline int extract_sampled_rows_from_coo(const sparse::coo *src,
                                         std::uint32_t global_row_base,
                                         std::uint32_t sample_stride,
                                         sparse::coo *out) {
    std::vector<std::uint32_t> row_map;
    std::size_t sampled_nnz = 0u;
    std::uint32_t sampled_rows = 0u;
    std::size_t emitted = 0u;
    if (src == 0 || out == 0 || sample_stride == 0u) return 0;
    sparse::clear(out);
    sparse::init(out, 0u, src->cols, 0u);
    if (src->rows == 0u || src->nnz == 0u) return 1;
    row_map.assign((std::size_t) src->rows, std::numeric_limits<std::uint32_t>::max());
    for (std::uint32_t row = 0u; row < src->rows; ++row) {
        if (((global_row_base + row) % sample_stride) != 0u) continue;
        row_map[(std::size_t) row] = sampled_rows++;
    }
    if (sampled_rows == 0u) return 1;
    for (types::nnz_t i = 0u; i < src->nnz; ++i) {
        if (row_map[(std::size_t) src->rowIdx[i]] != std::numeric_limits<std::uint32_t>::max()) ++sampled_nnz;
    }
    sparse::init(out, sampled_rows, src->cols, (types::nnz_t) sampled_nnz);
    if (!sparse::allocate(out)) return 0;
    for (types::nnz_t i = 0u; i < src->nnz; ++i) {
        const std::uint32_t sampled_row = row_map[(std::size_t) src->rowIdx[i]];
        if (sampled_row == std::numeric_limits<std::uint32_t>::max()) continue;
        out->rowIdx[emitted] = sampled_row;
        out->colIdx[emitted] = src->colIdx[i];
        out->val[emitted] = src->val[i];
        ++emitted;
    }
    return emitted == sampled_nnz;
}

inline int build_sampled_shard_coo(sparse::blocked_ell *const *parts,
                                   std::uint32_t partition_count,
                                   std::uint32_t cols,
                                   std::uint32_t max_sample_rows,
                                   sparse::coo *out) {
    std::uint64_t total_rows = 0u;
    std::uint32_t sample_stride = 1u;
    std::uint32_t global_row_base = 0u;
    sparse::coo canonical;
    sparse::coo sampled_part;
    int ok = 0;
    if (out == 0) return 0;
    sparse::clear(out);
    sparse::init(out, 0u, cols, 0u);
    sparse::init(&canonical);
    sparse::init(&sampled_part);
    for (std::uint32_t partition = 0u; partition < partition_count; ++partition) {
        const sparse::blocked_ell *part = parts != 0 ? parts[partition] : 0;
        if (part == 0) goto done;
        total_rows += part->rows;
    }
    if (total_rows == 0u) {
        ok = 1;
        goto done;
    }
    sample_stride = max_sample_rows == 0u
        ? 1u
        : (std::uint32_t) std::max<std::uint64_t>(1u, (total_rows + (std::uint64_t) max_sample_rows - 1u) / (std::uint64_t) max_sample_rows);
    for (std::uint32_t partition = 0u; partition < partition_count; ++partition) {
        const sparse::blocked_ell *part = parts != 0 ? parts[partition] : 0;
        sparse::clear(&canonical);
        sparse::clear(&sampled_part);
        sparse::init(&canonical);
        sparse::init(&sampled_part);
        if (part == 0) goto done;
        if (!blocked_ell_to_canonical_coo(part, &canonical)) goto done;
        if (!extract_sampled_rows_from_coo(&canonical, global_row_base, sample_stride, &sampled_part)) goto done;
        if (!sparse::append_rows(out, &sampled_part)) goto done;
        global_row_base += part->rows;
    }
    ok = 1;

done:
    sparse::clear(&canonical);
    sparse::clear(&sampled_part);
    if (!ok) {
        sparse::clear(out);
        sparse::init(out);
    }
    return ok;
}

inline int apply_coo_permutation(const sparse::coo *src,
                                 const std::uint32_t *canonical_to_exec_rows,
                                 const std::uint32_t *canonical_to_exec_cols,
                                 sparse::coo *out) {
    if (src == 0 || out == 0) return 0;
    sparse::clear(out);
    sparse::init(out, src->rows, src->cols, src->nnz);
    if (!sparse::allocate(out)) return 0;
    for (types::nnz_t i = 0u; i < src->nnz; ++i) {
        const std::uint32_t row = src->rowIdx[i];
        const std::uint32_t col = src->colIdx[i];
        const std::uint32_t exec_row = canonical_to_exec_rows != 0 ? canonical_to_exec_rows[row] : row;
        const std::uint32_t exec_col = canonical_to_exec_cols != 0 ? canonical_to_exec_cols[col] : col;
        if (exec_row >= src->rows || exec_col >= src->cols) {
            sparse::clear(out);
            sparse::init(out);
            return 0;
        }
        out->rowIdx[i] = exec_row;
        out->colIdx[i] = exec_col;
        out->val[i] = src->val[i];
    }
    return 1;
}

inline int build_single_segment_execution_partition(bucketed_blocked_ell_partition *out,
                                                    const sparse::blocked_ell *part,
                                                    const std::uint32_t *exec_to_canonical_rows,
                                                    const std::uint32_t *canonical_to_exec_rows,
                                                    const std::uint32_t *exec_to_canonical_cols,
                                                    const std::uint32_t *canonical_to_exec_cols,
                                                    std::uint64_t *execution_bytes_out) {
    const std::size_t block_idx_count = part != 0
        ? blocked_ell_part_block_index_count(part->rows, ::cellshard::partition_aux(part))
        : 0u;
    const std::size_t value_count = part != 0 ? (std::size_t) part->rows * (std::size_t) part->ell_cols : 0u;
    if (out == 0 || part == 0) return 0;
    clear(out);
    init(out);
    out->rows = part->rows;
    out->cols = part->cols;
    out->nnz = part->nnz;
    out->segment_count = 1u;
    out->segments = (sparse::blocked_ell *) std::calloc(1u, sizeof(sparse::blocked_ell));
    out->segment_row_offsets = (std::uint32_t *) std::calloc(2u, sizeof(std::uint32_t));
    if (out->segments == 0 || out->segment_row_offsets == 0
        || !duplicate_u32_array(&out->exec_to_canonical_rows, exec_to_canonical_rows, out->rows)
        || !duplicate_u32_array(&out->canonical_to_exec_rows, canonical_to_exec_rows, out->rows)
        || !duplicate_u32_array(&out->exec_to_canonical_cols, exec_to_canonical_cols, out->cols)
        || !duplicate_u32_array(&out->canonical_to_exec_cols, canonical_to_exec_cols, out->cols)) {
        clear(out);
        return 0;
    }
    sparse::init(out->segments + 0u,
                 part->rows,
                 part->cols,
                 part->nnz,
                 part->block_size,
                 part->ell_cols);
    if (!sparse::allocate(out->segments + 0u)) {
        clear(out);
        return 0;
    }
    out->segment_row_offsets[0] = 0u;
    out->segment_row_offsets[1] = out->rows;
    if (block_idx_count != 0u && part->blockColIdx != 0) {
        std::memcpy(out->segments[0].blockColIdx,
                    part->blockColIdx,
                    block_idx_count * sizeof(types::idx_t));
    }
    if (value_count != 0u && part->val != 0) {
        std::memcpy(out->segments[0].val,
                    part->val,
                    value_count * sizeof(real::storage_t));
    }
    if (execution_bytes_out != 0) {
        *execution_bytes_out = (std::uint64_t) packed_bytes((const sparse::blocked_ell *) 0,
                                                            out->segments[0].rows,
                                                            out->segments[0].cols,
                                                            out->segments[0].nnz,
                                                            ::cellshard::partition_aux(out->segments + 0u),
                                                            sizeof(real::storage_t));
    }
    return 1;
}

inline int bucketed_partition_to_canonical_coo(const bucketed_blocked_ell_partition *part,
                                               const std::uint32_t *exec_to_canonical_cols,
                                               sparse::coo *out) {
    std::size_t actual_nnz = 0u;
    std::size_t emitted = 0u;
    if (part == 0 || out == 0) return 0;
    sparse::clear(out);
    for (std::uint32_t segment = 0u; segment < part->segment_count; ++segment) {
        const sparse::blocked_ell *seg = part->segments + segment;
        const std::uint32_t block_size = seg->block_size;
        const std::uint32_t width_blocks = sparse::ell_width_blocks(seg);
        const std::uint32_t exec_row_base = part->segment_row_offsets[segment];
        for (std::uint32_t row = 0u; row < seg->rows; ++row) {
            const std::uint32_t row_block = row / block_size;
            const std::uint32_t canonical_row = part->exec_to_canonical_rows[exec_row_base + row];
            for (std::uint32_t slot = 0u; slot < width_blocks; ++slot) {
                const types::idx_t block_col = seg->blockColIdx[(std::size_t) row_block * width_blocks + slot];
                if (block_col == sparse::blocked_ell_invalid_col) continue;
                for (std::uint32_t col_in_block = 0u; col_in_block < block_size; ++col_in_block) {
                    const std::uint32_t exec_col = (std::uint32_t) block_col * block_size + col_in_block;
                    const real::storage_t value =
                        seg->val[(std::size_t) row * seg->ell_cols + (std::size_t) slot * block_size + col_in_block];
                    if (__half2float(value) == 0.0f || exec_col >= part->cols) continue;
                    ++actual_nnz;
                }
            }
        }
    }
    sparse::init(out, part->rows, part->cols, (types::nnz_t) actual_nnz);
    if (!sparse::allocate(out)) return 0;
    for (std::uint32_t segment = 0u; segment < part->segment_count; ++segment) {
        const sparse::blocked_ell *seg = part->segments + segment;
        const std::uint32_t block_size = seg->block_size;
        const std::uint32_t width_blocks = sparse::ell_width_blocks(seg);
        const std::uint32_t exec_row_base = part->segment_row_offsets[segment];
        for (std::uint32_t row = 0u; row < seg->rows; ++row) {
            const std::uint32_t row_block = row / block_size;
            const std::uint32_t canonical_row = part->exec_to_canonical_rows[exec_row_base + row];
            for (std::uint32_t slot = 0u; slot < width_blocks; ++slot) {
                const types::idx_t block_col = seg->blockColIdx[(std::size_t) row_block * width_blocks + slot];
                if (block_col == sparse::blocked_ell_invalid_col) continue;
                for (std::uint32_t col_in_block = 0u; col_in_block < block_size; ++col_in_block) {
                    const std::uint32_t exec_col = (std::uint32_t) block_col * block_size + col_in_block;
                    const real::storage_t value =
                        seg->val[(std::size_t) row * seg->ell_cols + (std::size_t) slot * block_size + col_in_block];
                    if (__half2float(value) == 0.0f || exec_col >= part->cols) continue;
                    out->rowIdx[emitted] = canonical_row;
                    out->colIdx[emitted] = exec_to_canonical_cols != 0 ? exec_to_canonical_cols[exec_col] : exec_col;
                    out->val[emitted] = value;
                    ++emitted;
                }
            }
        }
    }
    if (emitted != out->nnz) {
        sparse::clear(out);
        return 0;
    }
    return 1;
}

inline int load_or_materialize_blocked_ell_parts(sharded<sparse::blocked_ell> *m,
                                                 dataset_h5_state *state,
                                                 unsigned long shard_id,
                                                 unsigned long begin,
                                                 unsigned long end,
                                                 int assign_to_matrix,
                                                 int store_to_cache,
                                                 int require_cache_hit_only) {
    const unsigned long partition_count = end - begin;
    sparse::blocked_ell **parts = 0;
    unsigned long i = 0;
    int ok = 0;

    if (m == 0 || state == 0 || begin > end || end > m->num_partitions) return 0;
    if (partition_count == 0ul) return 1;

    if (require_cache_hit_only) return 0;
    if (!load_blocked_ell_shard_payload(state, shard_id)) return 0;

    parts = (sparse::blocked_ell **) std::calloc((std::size_t) partition_count, sizeof(sparse::blocked_ell *));
    if (parts == 0) return 0;
    if (!prepare_blocked_ell_parts_from_state(state, begin, end, parts)) goto done;
    if (!fill_blocked_ell_parts_from_loaded_shard(state, shard_id, begin, end, parts)) goto done;
    (void) store_to_cache;

    if (assign_to_matrix) {
        for (i = 0; i < partition_count; ++i) {
            if (m->parts[begin + i] != 0) destroy(m->parts[begin + i]);
            m->parts[begin + i] = parts[i];
            parts[i] = 0;
        }
    }

    ok = 1;

done:
    clear_blocked_ell_parts(parts, partition_count);
    std::free(parts);
    return ok;
}

struct execution_bucket_layout {
    std::vector<std::uint32_t> row_block_order;
    std::vector<std::uint32_t> row_block_widths;
    std::vector<std::uint32_t> segment_row_block_offsets;
    std::vector<std::uint32_t> segment_width_blocks;
};

inline std::uint32_t count_valid_block_slots(const sparse::blocked_ell *part, std::uint32_t row_block) {
    const std::uint32_t width = sparse::ell_width_blocks(part);
    std::uint32_t count = 0u;
    std::uint32_t slot = 0u;
    if (part == 0 || part->blockColIdx == 0 || row_block >= sparse::row_block_count(part)) return 0u;
    for (slot = 0u; slot < width; ++slot) {
        if (part->blockColIdx[(std::size_t) row_block * width + slot] != sparse::blocked_ell_invalid_col) ++count;
    }
    return count;
}

inline std::uint32_t rows_in_row_block(const sparse::blocked_ell *part, std::uint32_t row_block) {
    const std::uint32_t row_begin = part != 0 ? row_block * part->block_size : 0u;
    if (part == 0 || row_begin >= part->rows) return 0u;
    return std::min<std::uint32_t>(part->block_size, part->rows - row_begin);
}

inline int build_execution_bucket_layout(const sparse::blocked_ell *part,
                                         std::uint32_t requested_bucket_count,
                                         execution_bucket_layout *layout) {
    const std::uint32_t row_block_count = part != 0 ? sparse::row_block_count(part) : 0u;
    const std::uint32_t full_block_rows = part != 0 ? part->block_size : 0u;
    const std::uint32_t sortable_row_blocks =
        row_block_count != 0u && rows_in_row_block(part, row_block_count - 1u) != full_block_rows
            ? row_block_count - 1u
            : row_block_count;
    std::uint32_t bucket_count = 0u;
    std::uint32_t bucket = 0u;
    if (part == 0 || layout == 0) return 0;
    layout->row_block_order.clear();
    layout->row_block_widths.clear();
    layout->segment_row_block_offsets.clear();
    layout->segment_width_blocks.clear();
    if (row_block_count == 0u) {
        layout->segment_row_block_offsets.push_back(0u);
        return 1;
    }
    layout->row_block_order.resize(row_block_count);
    layout->row_block_widths.resize(row_block_count);
    for (std::uint32_t rb = 0u; rb < row_block_count; ++rb) {
        layout->row_block_order[rb] = rb;
        layout->row_block_widths[rb] = count_valid_block_slots(part, rb);
    }
    bucket_count = std::max<std::uint32_t>(1u, std::min<std::uint32_t>(requested_bucket_count, row_block_count));
    std::stable_sort(layout->row_block_order.begin(),
                     layout->row_block_order.begin() + sortable_row_blocks,
                     [&](std::uint32_t lhs, std::uint32_t rhs) {
                         const std::uint32_t lhs_width = layout->row_block_widths[lhs];
                         const std::uint32_t rhs_width = layout->row_block_widths[rhs];
                         if (lhs_width != rhs_width) return lhs_width < rhs_width;
                         return lhs < rhs;
                     });
    layout->segment_row_block_offsets.reserve((std::size_t) bucket_count + 1u);
    layout->segment_width_blocks.reserve(bucket_count);
    layout->segment_row_block_offsets.push_back(0u);
    for (bucket = 0u; bucket < bucket_count; ++bucket) {
        const std::uint32_t rb_begin = (bucket * row_block_count) / bucket_count;
        const std::uint32_t rb_end = ((bucket + 1u) * row_block_count) / bucket_count;
        std::uint32_t seg_width = 0u;
        for (std::uint32_t pos = rb_begin; pos < rb_end; ++pos) {
            seg_width = std::max(seg_width, layout->row_block_widths[layout->row_block_order[pos]]);
        }
        layout->segment_width_blocks.push_back(seg_width);
        layout->segment_row_block_offsets.push_back(rb_end);
    }
    if (layout->segment_width_blocks.size() > 1u) {
        bool all_same = true;
        for (std::size_t i = 1; i < layout->segment_width_blocks.size(); ++i) {
            if (layout->segment_width_blocks[i] != layout->segment_width_blocks[0]) {
                all_same = false;
                break;
            }
        }
        if (all_same) {
            layout->segment_row_block_offsets.assign(2u, 0u);
            layout->segment_row_block_offsets[1] = row_block_count;
            layout->segment_width_blocks.assign(1u, layout->segment_width_blocks[0]);
        }
    }
    return 1;
}

inline int allocate_bucketed_execution_partition(bucketed_blocked_ell_partition *out,
                                                 const sparse::blocked_ell *part,
                                                 const execution_bucket_layout &layout) {
    std::uint32_t segment = 0u;
    std::uint32_t row_cursor = 0u;
    if (out == 0 || part == 0) return 0;
    init(out);
    out->rows = part->rows;
    out->cols = part->cols;
    out->nnz = part->nnz;
    out->segment_count = (std::uint32_t) layout.segment_width_blocks.size();
    out->segments = out->segment_count != 0u ? (sparse::blocked_ell *) std::calloc((std::size_t) out->segment_count, sizeof(sparse::blocked_ell)) : 0;
    out->segment_row_offsets = (std::uint32_t *) std::calloc((std::size_t) out->segment_count + 1u, sizeof(std::uint32_t));
    out->exec_to_canonical_rows = out->rows != 0u ? (std::uint32_t *) std::calloc((std::size_t) out->rows, sizeof(std::uint32_t)) : 0;
    out->canonical_to_exec_rows = out->rows != 0u ? (std::uint32_t *) std::calloc((std::size_t) out->rows, sizeof(std::uint32_t)) : 0;
    if ((out->segment_count != 0u && (out->segments == 0 || out->segment_row_offsets == 0))
        || (out->rows != 0u && (out->exec_to_canonical_rows == 0 || out->canonical_to_exec_rows == 0))) {
        clear(out);
        return 0;
    }
    for (segment = 0u; segment < out->segment_count; ++segment) {
        const std::uint32_t rb_begin = layout.segment_row_block_offsets[segment];
        const std::uint32_t rb_end = layout.segment_row_block_offsets[segment + 1u];
        std::uint32_t seg_rows = 0u;
        std::uint32_t seg_nnz = 0u;
        for (std::uint32_t pos = rb_begin; pos < rb_end; ++pos) {
            const std::uint32_t rb = layout.row_block_order[pos];
            seg_rows += rows_in_row_block(part, rb);
            seg_nnz += rows_in_row_block(part, rb) * layout.row_block_widths[rb] * part->block_size;
        }
        out->segment_row_offsets[segment] = row_cursor;
        row_cursor += seg_rows;
        sparse::init(out->segments + segment,
                     seg_rows,
                     part->cols,
                     seg_nnz,
                     part->block_size,
                     layout.segment_width_blocks[segment] * part->block_size);
        if (!sparse::allocate(out->segments + segment)) {
            clear(out);
            return 0;
        }
        std::memset(out->segments[segment].storage, 0, sparse::bytes(out->segments + segment) - sizeof(sparse::blocked_ell));
        {
            const std::size_t idx_count = (std::size_t) sparse::row_block_count(out->segments + segment)
                * (std::size_t) sparse::ell_width_blocks(out->segments + segment);
            for (std::size_t idx = 0u; idx < idx_count; ++idx) out->segments[segment].blockColIdx[idx] = sparse::blocked_ell_invalid_col;
        }
    }
    out->segment_row_offsets[out->segment_count] = row_cursor;
    return row_cursor == out->rows;
}

inline std::uint64_t execution_segment_bytes(const sparse::blocked_ell *part) {
    return part != 0
        ? (std::uint64_t) packed_bytes((const sparse::blocked_ell *) 0,
                                       part->rows,
                                       part->cols,
                                       part->nnz,
                                       ::cellshard::partition_aux(part),
                                       sizeof(real::storage_t))
        : 0u;
}

inline int fill_bucketed_execution_partition(bucketed_blocked_ell_partition *out,
                                             const sparse::blocked_ell *part,
                                             const execution_bucket_layout &layout) {
    const std::uint32_t block_size = part != 0 ? part->block_size : 0u;
    const std::uint32_t src_width = part != 0 ? sparse::ell_width_blocks(part) : 0u;
    std::uint32_t segment = 0u;
    if (out == 0 || part == 0) return 0;
    for (segment = 0u; segment < out->segment_count; ++segment) {
        const std::uint32_t rb_begin = layout.segment_row_block_offsets[segment];
        const std::uint32_t rb_end = layout.segment_row_block_offsets[segment + 1u];
        sparse::blocked_ell *dst = out->segments + segment;
        std::uint32_t dst_rb = 0u;
        std::uint32_t exec_row = out->segment_row_offsets[segment];
        for (std::uint32_t pos = rb_begin; pos < rb_end; ++pos, ++dst_rb) {
            const std::uint32_t src_rb = layout.row_block_order[pos];
            const std::uint32_t src_rows = rows_in_row_block(part, src_rb);
            const std::size_t src_slot_base = (std::size_t) src_rb * src_width;
            std::size_t dst_slot = 0u;
            for (std::uint32_t row_in_block = 0u; row_in_block < src_rows; ++row_in_block) {
                const std::uint32_t canonical_row = src_rb * block_size + row_in_block;
                out->exec_to_canonical_rows[exec_row + row_in_block] = canonical_row;
                out->canonical_to_exec_rows[canonical_row] = exec_row + row_in_block;
            }
            for (std::uint32_t src_slot = 0u; src_slot < src_width; ++src_slot) {
                const types::idx_t block_col = part->blockColIdx[src_slot_base + src_slot];
                if (block_col == sparse::blocked_ell_invalid_col) continue;
                dst->blockColIdx[(std::size_t) dst_rb * sparse::ell_width_blocks(dst) + dst_slot] = block_col;
                for (std::uint32_t row_in_block = 0u; row_in_block < src_rows; ++row_in_block) {
                    const std::size_t src_offset =
                        (std::size_t) (src_rb * block_size + row_in_block) * part->ell_cols + (std::size_t) src_slot * block_size;
                    const std::size_t dst_offset =
                        (std::size_t) (dst_rb * block_size + row_in_block) * dst->ell_cols + dst_slot * block_size;
                    std::memcpy(dst->val + dst_offset, part->val + src_offset, (std::size_t) block_size * sizeof(real::storage_t));
                }
                ++dst_slot;
            }
            exec_row += src_rows;
        }
    }
    return 1;
}

inline int build_bucketed_execution_partition(bucketed_blocked_ell_partition *out,
                                              const sparse::blocked_ell *part,
                                              std::uint32_t requested_bucket_count,
                                              std::uint64_t *bucketed_bytes_out) {
    execution_bucket_layout layout;
    std::uint64_t candidate_bytes = 0u;
    std::uint64_t original_bytes = 0u;
    if (out == 0 || part == 0) return 0;
    if (!build_execution_bucket_layout(part, requested_bucket_count, &layout)) return 0;
    if (!allocate_bucketed_execution_partition(out, part, layout)) return 0;
    if (!fill_bucketed_execution_partition(out, part, layout)) {
        clear(out);
        return 0;
    }
    for (std::uint32_t segment = 0u; segment < out->segment_count; ++segment) {
        candidate_bytes += execution_segment_bytes(out->segments + segment);
    }
    original_bytes = execution_segment_bytes(part);
    if (out->segment_count > 1u && candidate_bytes >= original_bytes) {
        clear(out);
        if (!build_execution_bucket_layout(part, 1u, &layout)) return 0;
        if (!allocate_bucketed_execution_partition(out, part, layout)) return 0;
        if (!fill_bucketed_execution_partition(out, part, layout)) {
            clear(out);
            return 0;
        }
        candidate_bytes = original_bytes;
    }
    std::free(out->exec_to_canonical_cols);
    std::free(out->canonical_to_exec_cols);
    out->exec_to_canonical_cols = 0;
    out->canonical_to_exec_cols = 0;
    if (!build_identity_col_maps(out->cols,
                                 &out->exec_to_canonical_cols,
                                 &out->canonical_to_exec_cols)) {
        clear(out);
        return 0;
    }
    if (bucketed_bytes_out != 0) *bucketed_bytes_out = candidate_bytes;
    return 1;
}

inline int choose_bucket_count_for_part(const sparse::blocked_ell *part,
                                        std::uint32_t *bucket_count_out,
                                        std::uint64_t *bucketed_bytes_out) {
    const std::uint32_t row_blocks = part != 0 ? sparse::row_block_count(part) : 0u;
    const std::uint32_t max_buckets = std::min<std::uint32_t>(8u, row_blocks);
    bucketed_blocked_ell_partition trial;
    std::uint32_t best_buckets = 1u;
    std::uint64_t best_bytes = std::numeric_limits<std::uint64_t>::max();
    if (part == 0 || bucket_count_out == 0 || bucketed_bytes_out == 0) return 0;
    init(&trial);
    for (std::uint32_t buckets = 1u; buckets <= std::max<std::uint32_t>(1u, max_buckets); ++buckets) {
        std::uint64_t bytes = 0u;
        clear(&trial);
        init(&trial);
        if (!build_bucketed_execution_partition(&trial, part, buckets, &bytes)) {
            clear(&trial);
            return 0;
        }
        if (bytes < best_bytes || (bytes == best_bytes && buckets < best_buckets)) {
            best_bytes = bytes;
            best_buckets = buckets;
        }
    }
    clear(&trial);
    *bucket_count_out = best_buckets;
    *bucketed_bytes_out = best_bytes;
    return 1;
}

struct sliced_execution_bucket_layout {
    std::vector<std::uint32_t> row_order;
    std::vector<std::uint32_t> row_widths;
    std::vector<std::uint32_t> segment_row_offsets;
    std::vector<std::uint32_t> segment_widths;
};

inline std::uint32_t count_valid_sliced_slots(const sparse::sliced_ell *part, std::uint32_t row) {
    return part != 0 ? sparse::row_nnz(part, row) : 0u;
}

inline std::uint64_t packed_single_slice_segment_bytes(std::uint32_t rows, std::uint32_t width) {
    return (std::uint64_t) packed_sliced_ell_bytes(1u,
                                                   rows == 0u || width == 0u ? 0u : rows * width,
                                                   sizeof(real::storage_t));
}

struct sliced_bucket_dp_line {
    std::int64_t slope;
    std::int64_t intercept;
    std::uint32_t cut_rows;
};

inline std::int64_t eval_sliced_bucket_dp_line(const sliced_bucket_dp_line &line, std::uint32_t x) {
    return line.intercept + line.slope * (std::int64_t) x;
}

inline bool sliced_bucket_dp_line_is_redundant(const sliced_bucket_dp_line &lhs,
                                               const sliced_bucket_dp_line &mid,
                                               const sliced_bucket_dp_line &rhs) {
    const __int128 left = (__int128) (rhs.intercept - lhs.intercept) * (__int128) (lhs.slope - mid.slope);
    const __int128 right = (__int128) (mid.intercept - lhs.intercept) * (__int128) (lhs.slope - rhs.slope);
    return left <= right;
}

inline int build_sliced_execution_bucket_layout(const sparse::sliced_ell *part,
                                                std::uint32_t requested_bucket_count,
                                                sliced_execution_bucket_layout *layout) {
    const std::uint32_t row_count = part != 0 ? part->rows : 0u;
    const std::uint32_t max_segments = std::max<std::uint32_t>(1u, std::min<std::uint32_t>(requested_bucket_count, row_count));
    const std::uint64_t segment_fixed_bytes = packed_single_slice_segment_bytes(0u, 0u);
    const std::uint64_t slot_bytes = sizeof(types::idx_t) + sizeof(real::storage_t);
    static constexpr std::uint64_t inf_cost = std::numeric_limits<std::uint64_t>::max() / 4u;
    std::vector<std::uint64_t> prev_costs, curr_costs;
    std::vector<std::vector<std::uint32_t>> cut_rows;
    std::vector<std::uint32_t> sorted_widths;
    if (part == 0 || layout == 0) return 0;
    layout->row_order.clear();
    layout->row_widths.clear();
    layout->segment_row_offsets.clear();
    layout->segment_widths.clear();
    if (row_count == 0u) {
        layout->segment_row_offsets.push_back(0u);
        return 1;
    }
    layout->row_order.resize(row_count);
    layout->row_widths.resize(row_count);
    for (std::uint32_t row = 0u; row < row_count; ++row) {
        layout->row_order[row] = row;
        layout->row_widths[row] = count_valid_sliced_slots(part, row);
    }
    std::stable_sort(layout->row_order.begin(),
                     layout->row_order.end(),
                     [&](std::uint32_t lhs, std::uint32_t rhs) {
                         const std::uint32_t lhs_width = layout->row_widths[lhs];
                         const std::uint32_t rhs_width = layout->row_widths[rhs];
                         if (lhs_width != rhs_width) return lhs_width < rhs_width;
                         return lhs < rhs;
                     });
    sorted_widths.resize(row_count);
    for (std::uint32_t row = 0u; row < row_count; ++row) {
        sorted_widths[row] = layout->row_widths[layout->row_order[row]];
    }

    prev_costs.assign((std::size_t) row_count + 1u, inf_cost);
    curr_costs.assign((std::size_t) row_count + 1u, inf_cost);
    cut_rows.assign((std::size_t) max_segments + 1u, std::vector<std::uint32_t>((std::size_t) row_count + 1u, 0u));
    prev_costs[0] = 0u;

    for (std::uint32_t segments = 1u; segments <= max_segments; ++segments) {
        std::vector<sliced_bucket_dp_line> hull;
        std::size_t head = 0u;
        curr_costs.assign((std::size_t) row_count + 1u, inf_cost);
        hull.reserve((std::size_t) row_count + 1u);
        if (prev_costs[segments - 1u] != inf_cost) {
            const std::uint32_t prefix_rows = segments - 1u;
            hull.push_back({
                -((std::int64_t) slot_bytes * (std::int64_t) prefix_rows),
                (std::int64_t) prev_costs[prefix_rows],
                prefix_rows
            });
        }

        for (std::uint32_t rows = segments; rows <= row_count; ++rows) {
            const std::uint32_t width = sorted_widths[(std::size_t) rows - 1u];
            while (head + 1u < hull.size()
                   && eval_sliced_bucket_dp_line(hull[head + 1u], width)
                          < eval_sliced_bucket_dp_line(hull[head], width)) {
                ++head;
            }
            if (head < hull.size()) {
                const sliced_bucket_dp_line best = hull[head];
                curr_costs[rows] =
                    segment_fixed_bytes
                    + slot_bytes * (std::uint64_t) width * (std::uint64_t) rows
                    + (std::uint64_t) eval_sliced_bucket_dp_line(best, width);
                cut_rows[segments][rows] = best.cut_rows;
            }
            if (prev_costs[rows] != inf_cost) {
                const sliced_bucket_dp_line line = {
                    -((std::int64_t) slot_bytes * (std::int64_t) rows),
                    (std::int64_t) prev_costs[rows],
                    rows
                };
                while (hull.size() >= head + 2u
                       && sliced_bucket_dp_line_is_redundant(hull[hull.size() - 2u], hull[hull.size() - 1u], line)) {
                    hull.pop_back();
                }
                hull.push_back(line);
            }
        }
        prev_costs.swap(curr_costs);
    }

    if (prev_costs[row_count] == inf_cost) return 0;

    {
        std::vector<std::uint32_t> raw_offsets((std::size_t) max_segments + 1u, 0u);
        std::vector<std::uint32_t> raw_widths(max_segments, 0u);
        std::uint32_t rows = row_count;
        raw_offsets[(std::size_t) max_segments] = row_count;
        for (std::uint32_t segments = max_segments; segments != 0u; --segments) {
            const std::uint32_t row_begin = cut_rows[segments][rows];
            raw_offsets[(std::size_t) segments - 1u] = row_begin;
            raw_widths[(std::size_t) segments - 1u] = sorted_widths[(std::size_t) rows - 1u];
            rows = row_begin;
        }
        layout->segment_row_offsets.reserve((std::size_t) max_segments + 1u);
        layout->segment_widths.reserve(max_segments);
        layout->segment_row_offsets.push_back(0u);
        for (std::uint32_t segment = 0u; segment < max_segments; ++segment) {
            const std::uint32_t row_end = raw_offsets[(std::size_t) segment + 1u];
            const std::uint32_t seg_width = raw_widths[segment];
            if (!layout->segment_widths.empty() && layout->segment_widths.back() == seg_width) {
                layout->segment_row_offsets.back() = row_end;
            } else {
                layout->segment_widths.push_back(seg_width);
                layout->segment_row_offsets.push_back(row_end);
            }
        }
    }
    return 1;
}

inline int allocate_bucketed_sliced_execution_partition(bucketed_sliced_ell_partition *out,
                                                        const sparse::sliced_ell *part,
                                                        const sliced_execution_bucket_layout &layout) {
    std::uint32_t segment = 0u;
    std::uint32_t row_cursor = 0u;
    if (out == 0 || part == 0) return 0;
    init(out);
    out->rows = part->rows;
    out->cols = part->cols;
    out->nnz = part->nnz;
    out->segment_count = (std::uint32_t) layout.segment_widths.size();
    out->canonical_slice_count = part->slice_count;
    out->segments = out->segment_count != 0u ? (sparse::sliced_ell *) std::calloc((std::size_t) out->segment_count, sizeof(sparse::sliced_ell)) : 0;
    out->segment_row_offsets = (std::uint32_t *) std::calloc((std::size_t) out->segment_count + 1u, sizeof(std::uint32_t));
    out->exec_to_canonical_rows = out->rows != 0u ? (std::uint32_t *) std::calloc((std::size_t) out->rows, sizeof(std::uint32_t)) : 0;
    out->canonical_to_exec_rows = out->rows != 0u ? (std::uint32_t *) std::calloc((std::size_t) out->rows, sizeof(std::uint32_t)) : 0;
    out->canonical_slice_row_offsets =
        (std::uint32_t *) std::calloc((std::size_t) out->canonical_slice_count + 1u, sizeof(std::uint32_t));
    out->canonical_slice_widths = out->canonical_slice_count != 0u
        ? (std::uint32_t *) std::calloc((std::size_t) out->canonical_slice_count, sizeof(std::uint32_t))
        : 0;
    if ((out->segment_count != 0u && (out->segments == 0 || out->segment_row_offsets == 0))
        || (out->rows != 0u && (out->exec_to_canonical_rows == 0 || out->canonical_to_exec_rows == 0))
        || (((std::size_t) out->canonical_slice_count + 1u) != 0u && out->canonical_slice_row_offsets == 0)
        || (out->canonical_slice_count != 0u && out->canonical_slice_widths == 0)) {
        clear(out);
        return 0;
    }
    if (out->canonical_slice_row_offsets != 0) {
        std::memcpy(out->canonical_slice_row_offsets,
                    part->slice_row_offsets,
                    ((std::size_t) out->canonical_slice_count + 1u) * sizeof(std::uint32_t));
    }
    if (out->canonical_slice_widths != 0) {
        std::memcpy(out->canonical_slice_widths,
                    part->slice_widths,
                    (std::size_t) out->canonical_slice_count * sizeof(std::uint32_t));
    }
    for (segment = 0u; segment < out->segment_count; ++segment) {
        const std::uint32_t row_begin = layout.segment_row_offsets[segment];
        const std::uint32_t row_end = layout.segment_row_offsets[segment + 1u];
        const std::uint32_t seg_rows = row_end - row_begin;
        const std::uint32_t seg_width = layout.segment_widths[segment];
        std::uint32_t seg_nnz = 0u;
        const std::uint32_t slice_row_offsets[2] = { 0u, seg_rows };
        const std::uint32_t slice_widths[1] = { seg_width };
        for (std::uint32_t pos = row_begin; pos < row_end; ++pos) {
            seg_nnz += layout.row_widths[layout.row_order[pos]];
        }
        out->segment_row_offsets[segment] = row_cursor;
        row_cursor += seg_rows;
        sparse::init(out->segments + segment, seg_rows, part->cols, seg_nnz);
        if (!sparse::allocate(out->segments + segment, seg_rows != 0u ? 1u : 0u, slice_row_offsets, slice_widths)) {
            clear(out);
            return 0;
        }
    }
    out->segment_row_offsets[out->segment_count] = row_cursor;
    return row_cursor == out->rows;
}

inline std::uint64_t sliced_execution_segment_bytes(const sparse::sliced_ell *part) {
    return part != 0
        ? (std::uint64_t) packed_bytes((const sparse::sliced_ell *) 0,
                                       part->rows,
                                       part->cols,
                                       part->nnz,
                                       ::cellshard::partition_aux(part),
                                       sizeof(real::storage_t))
        : 0u;
}

inline int fill_bucketed_sliced_execution_partition(bucketed_sliced_ell_partition *out,
                                                    const sparse::sliced_ell *part,
                                                    const sliced_execution_bucket_layout &layout) {
    if (out == 0 || part == 0) return 0;
    for (std::uint32_t segment = 0u; segment < out->segment_count; ++segment) {
        const std::uint32_t row_begin = layout.segment_row_offsets[segment];
        const std::uint32_t row_end = layout.segment_row_offsets[segment + 1u];
        sparse::sliced_ell *dst = out->segments + segment;
        const std::uint32_t seg_width = dst->slice_count != 0u ? dst->slice_widths[0] : 0u;
        for (std::uint32_t pos = row_begin; pos < row_end; ++pos) {
            const std::uint32_t canonical_row = layout.row_order[pos];
            const std::uint32_t exec_row = out->segment_row_offsets[segment] + (pos - row_begin);
            const std::uint32_t dst_row = pos - row_begin;
            const std::uint32_t src_slice = sparse::find_slice(part, canonical_row);
            const std::uint32_t src_row_begin = src_slice < part->slice_count ? part->slice_row_offsets[src_slice] : 0u;
            const std::uint32_t src_width = src_slice < part->slice_count ? part->slice_widths[src_slice] : 0u;
            const std::size_t src_base = sparse::slice_slot_base(part, src_slice)
                + (std::size_t) (canonical_row - src_row_begin) * (std::size_t) src_width;
            const std::size_t dst_base = (std::size_t) dst_row * (std::size_t) seg_width;
            std::size_t dst_slot = 0u;
            for (std::uint32_t src_slot = 0u; src_slot < src_width; ++src_slot) {
                const types::idx_t col = part->col_idx[src_base + src_slot];
                if (col == sparse::sliced_ell_invalid_col) continue;
                dst->col_idx[dst_base + dst_slot] = col;
                dst->val[dst_base + dst_slot] = part->val[src_base + src_slot];
                ++dst_slot;
            }
            out->exec_to_canonical_rows[exec_row] = canonical_row;
            out->canonical_to_exec_rows[canonical_row] = exec_row;
        }
    }
    return 1;
}

inline int build_bucketed_sliced_execution_partition(bucketed_sliced_ell_partition *out,
                                                     const sparse::sliced_ell *part,
                                                     std::uint32_t requested_bucket_count,
                                                     std::uint64_t *bucketed_bytes_out) {
    sliced_execution_bucket_layout layout;
    std::uint64_t candidate_bytes = 0u;
    std::uint64_t original_bytes = 0u;
    if (out == 0 || part == 0) return 0;
    if (!build_sliced_execution_bucket_layout(part, requested_bucket_count, &layout)) return 0;
    if (!allocate_bucketed_sliced_execution_partition(out, part, layout)) return 0;
    if (!fill_bucketed_sliced_execution_partition(out, part, layout)) {
        clear(out);
        return 0;
    }
    for (std::uint32_t segment = 0u; segment < out->segment_count; ++segment) {
        candidate_bytes += sliced_execution_segment_bytes(out->segments + segment);
    }
    original_bytes = sliced_execution_segment_bytes(part);
    if (out->segment_count > 1u && candidate_bytes >= original_bytes) {
        clear(out);
        if (!build_sliced_execution_bucket_layout(part, 1u, &layout)) return 0;
        if (!allocate_bucketed_sliced_execution_partition(out, part, layout)) return 0;
        if (!fill_bucketed_sliced_execution_partition(out, part, layout)) {
            clear(out);
            return 0;
        }
        candidate_bytes = original_bytes;
    }
    if (bucketed_bytes_out != 0) *bucketed_bytes_out = candidate_bytes;
    return 1;
}

inline int choose_bucket_count_for_sliced_part(const sparse::sliced_ell *part,
                                               std::uint32_t *bucket_count_out,
                                               std::uint64_t *bucketed_bytes_out) {
    const std::uint32_t rows = part != 0 ? part->rows : 0u;
    const std::uint32_t max_buckets = std::min<std::uint32_t>(8u, rows);
    bucketed_sliced_ell_partition trial;
    std::uint32_t best_buckets = 1u;
    std::uint64_t best_bytes = std::numeric_limits<std::uint64_t>::max();
    if (part == 0 || bucket_count_out == 0 || bucketed_bytes_out == 0) return 0;
    init(&trial);
    for (std::uint32_t buckets = 1u; buckets <= std::max<std::uint32_t>(1u, max_buckets); ++buckets) {
        std::uint64_t bytes = 0u;
        clear(&trial);
        init(&trial);
        if (!build_bucketed_sliced_execution_partition(&trial, part, buckets, &bytes)) {
            clear(&trial);
            return 0;
        }
        if (bytes < best_bytes || (bytes == best_bytes && buckets < best_buckets)) {
            best_bytes = bytes;
            best_buckets = buckets;
        }
    }
    clear(&trial);
    *bucket_count_out = best_buckets;
    *bucketed_bytes_out = best_bytes;
    return 1;
}

inline int build_optimized_partition_from_canonical_part(bucketed_blocked_ell_partition *out,
                                                         const sparse::blocked_ell *canonical_part,
                                                         const std::uint32_t *canonical_to_exec_cols,
                                                         const std::uint32_t *exec_to_canonical_cols,
                                                         std::uint32_t bucket_count,
                                                         std::uint64_t *bucketed_bytes_out) {
    sparse::coo canonical_coo;
    sparse::blocked_ell permuted_part;
    int ok = 0;
    sparse::init(&canonical_coo);
    sparse::init(&permuted_part);
    if (out == 0 || canonical_part == 0) return 0;
    if (!blocked_ell_to_canonical_coo(canonical_part, &canonical_coo)) goto done;
    if (!::cellshard::convert::blocked_ell_from_coo(&canonical_coo,
                                                    canonical_part->cols,
                                                    canonical_to_exec_cols,
                                                    canonical_part->block_size,
                                                    &permuted_part)) {
        goto done;
    }
    if (!build_bucketed_execution_partition(out, &permuted_part, bucket_count, bucketed_bytes_out)) goto done;
    if (!assign_partition_col_maps(out, exec_to_canonical_cols, canonical_to_exec_cols, canonical_part->cols)) {
        clear(out);
        goto done;
    }
    ok = 1;

done:
    sparse::clear(&permuted_part);
    sparse::clear(&canonical_coo);
    return ok;
}

inline int reconstruct_canonical_blocked_ell_part(const bucketed_blocked_ell_partition *optimized_part,
                                                  const std::uint32_t *exec_to_canonical_cols,
                                                  std::uint32_t block_size,
                                                  sparse::blocked_ell *out) {
    sparse::coo canonical_coo;
    int ok = 0;
    sparse::init(&canonical_coo);
    if (optimized_part == 0 || out == 0 || block_size == 0u) return 0;
    if (!bucketed_partition_to_canonical_coo(optimized_part, exec_to_canonical_cols, &canonical_coo)) goto done;
    if (!::cellshard::convert::blocked_ell_from_coo(&canonical_coo,
                                                    optimized_part->cols,
                                                    nullptr,
                                                    block_size,
                                                    out)) {
        goto done;
    }
    ok = 1;

done:
    sparse::clear(&canonical_coo);
    return ok;
}

inline void move_bucketed_blocked_ell_shard(bucketed_blocked_ell_shard *dst,
                                            bucketed_blocked_ell_shard *src) {
    if (dst == 0 || src == 0 || dst == src) return;
    clear(dst);
    *dst = *src;
    init(src);
}

inline int finalize_shard_block_size(const std::uint32_t *partition_block_sizes,
                                     std::uint32_t partition_count,
                                     std::uint32_t *shard_block_size_out) {
    std::uint32_t shard_block_size = 0u;
    if (shard_block_size_out == 0) return 0;
    if (partition_count != 0u && partition_block_sizes != 0) {
        shard_block_size = partition_block_sizes[0];
        for (std::uint32_t partition = 1u; partition < partition_count; ++partition) {
            if (partition_block_sizes[partition] != shard_block_size) {
                shard_block_size = 0u;
                break;
            }
        }
    }
    *shard_block_size_out = shard_block_size;
    return 1;
}

inline int build_heuristic_bucketed_optimized_shard_from_parts(sparse::blocked_ell *const *parts,
                                                               std::uint32_t partition_count,
                                                               std::uint32_t cols,
                                                               bucketed_blocked_ell_shard *out,
                                                               std::uint32_t *partition_block_sizes,
                                                               std::uint32_t *partition_bucket_counts,
                                                               float *partition_fill_ratios,
                                                               std::uint64_t *partition_execution_bytes,
                                                               std::uint64_t *partition_blocked_ell_bytes,
                                                               std::uint64_t *partition_bucketed_blocked_ell_bytes,
                                                               std::uint32_t *shard_block_size_out,
                                                               std::uint32_t *shard_bucketed_segment_count_out,
                                                               float *shard_fill_ratio_out,
                                                               std::uint64_t *shard_execution_bytes_out,
                                                               std::uint64_t *shard_bucketed_blocked_ell_bytes_out) {
    std::uint32_t *exec_to_canonical_cols = 0;
    std::uint32_t *canonical_to_exec_cols = 0;
    std::uint32_t local_rows = 0u;
    std::uint64_t shard_bucketed_bytes = 0u;
    std::uint64_t shard_blocked_bytes = 0u;
    std::uint64_t fill_weight = 0u;
    double fill_weighted_sum = 0.0;
    std::uint32_t shard_segment_count = 0u;
    sparse::coo canonical_coo;
    sparse::blocked_ell permuted;

    if (out == 0) return 0;
    clear(out);
    init(out);
    sparse::init(&canonical_coo);
    sparse::init(&permuted);
    if (!build_shard_column_maps(parts,
                                 partition_count,
                                 cols,
                                 &exec_to_canonical_cols,
                                 &canonical_to_exec_cols)) {
        goto done;
    }

    out->rows = 0u;
    out->cols = cols;
    out->nnz = 0u;
    out->partition_count = partition_count;
    out->partitions = partition_count != 0u
        ? (bucketed_blocked_ell_partition *) std::calloc((std::size_t) partition_count, sizeof(bucketed_blocked_ell_partition))
        : 0;
    out->partition_row_offsets = (std::uint32_t *) std::calloc((std::size_t) partition_count + 1u, sizeof(std::uint32_t));
    out->exec_to_canonical_cols = exec_to_canonical_cols;
    out->canonical_to_exec_cols = canonical_to_exec_cols;
    exec_to_canonical_cols = 0;
    canonical_to_exec_cols = 0;
    if ((partition_count != 0u && (out->partitions == 0 || out->partition_row_offsets == 0))
        || (cols != 0u && (out->exec_to_canonical_cols == 0 || out->canonical_to_exec_cols == 0))) {
        goto done;
    }

    for (std::uint32_t partition = 0u; partition < partition_count; ++partition) {
        const sparse::blocked_ell *canonical_part = parts != 0 ? parts[partition] : 0;
        std::uint32_t bucket_count = 1u;
        std::uint64_t bucketed_bytes = 0u;
        const std::uint64_t blocked_bytes = canonical_part != 0
            ? packed_bytes((const sparse::blocked_ell *) 0,
                           canonical_part->rows,
                           canonical_part->cols,
                           canonical_part->nnz,
                           partition_aux(canonical_part),
                           sizeof(real::storage_t))
            : 0u;
        bucketed_blocked_ell_partition *bucketed = out->partitions + partition;
        init(bucketed);
        out->partition_row_offsets[partition] = local_rows;
        if (canonical_part == 0) goto done;
        if (!blocked_ell_to_canonical_coo(canonical_part, &canonical_coo)
            || !convert::blocked_ell_from_coo(&canonical_coo,
                                              cols,
                                              out->canonical_to_exec_cols,
                                              canonical_part->block_size,
                                              &permuted)
            || !choose_bucket_count_for_blocked_part_host_exact(&permuted, &bucket_count)
            || !build_bucketed_execution_partition(bucketed, &permuted, bucket_count, &bucketed_bytes)
            || !assign_partition_col_maps(bucketed,
                                          out->exec_to_canonical_cols,
                                          out->canonical_to_exec_cols,
                                          cols)) {
            goto done;
        }
        local_rows += bucketed->rows;
        out->nnz += bucketed->nnz;
        shard_bucketed_bytes += bucketed_bytes;
        shard_blocked_bytes += blocked_bytes;
        fill_weight += canonical_part->rows;
        fill_weighted_sum += (double) canonical_part->rows * blocked_ell_value_fill_ratio(canonical_part);
        shard_segment_count += std::max<std::uint32_t>(1u, bucketed->segment_count);
        if (partition_block_sizes != 0) partition_block_sizes[partition] = canonical_part->block_size;
        if (partition_bucket_counts != 0) partition_bucket_counts[partition] = std::max<std::uint32_t>(1u, bucket_count);
        if (partition_fill_ratios != 0) partition_fill_ratios[partition] = blocked_ell_value_fill_ratio(canonical_part);
        if (partition_execution_bytes != 0) partition_execution_bytes[partition] = bucketed_bytes != 0u ? bucketed_bytes : blocked_bytes;
        if (partition_blocked_ell_bytes != 0) partition_blocked_ell_bytes[partition] = blocked_bytes;
        if (partition_bucketed_blocked_ell_bytes != 0) partition_bucketed_blocked_ell_bytes[partition] = bucketed_bytes;
        sparse::clear(&canonical_coo);
        sparse::init(&canonical_coo);
        sparse::clear(&permuted);
        sparse::init(&permuted);
    }
    out->partition_row_offsets[(std::size_t) partition_count] = local_rows;
    out->rows = local_rows;
    if (shard_block_size_out != 0 && !finalize_shard_block_size(partition_block_sizes, partition_count, shard_block_size_out)) goto done;
    if (shard_bucketed_segment_count_out != 0) *shard_bucketed_segment_count_out = shard_segment_count;
    if (shard_fill_ratio_out != 0) *shard_fill_ratio_out = fill_weight != 0u ? (float) (fill_weighted_sum / (double) fill_weight) : 0.0f;
    if (shard_execution_bytes_out != 0) *shard_execution_bytes_out = shard_bucketed_bytes != 0u ? shard_bucketed_bytes : shard_blocked_bytes;
    if (shard_bucketed_blocked_ell_bytes_out != 0) *shard_bucketed_blocked_ell_bytes_out = shard_bucketed_bytes != 0u ? shard_bucketed_bytes : shard_blocked_bytes;
    sparse::clear(&canonical_coo);
    sparse::clear(&permuted);
    return 1;

done:
    std::free(exec_to_canonical_cols);
    std::free(canonical_to_exec_cols);
    sparse::clear(&canonical_coo);
    sparse::clear(&permuted);
    clear(out);
    init(out);
    return 0;
}

#if CELLSHARD_ENABLE_CUDA
inline int build_graph_bucketed_optimized_shard_from_parts(sparse::blocked_ell *const *parts,
                                                           std::uint32_t partition_count,
                                                           std::uint32_t cols,
                                                           bucketed_blocked_ell_shard *out,
                                                           std::uint32_t *partition_block_sizes,
                                                           std::uint32_t *partition_bucket_counts,
                                                           float *partition_fill_ratios,
                                                           std::uint64_t *partition_execution_bytes,
                                                           std::uint64_t *partition_blocked_ell_bytes,
                                                           std::uint64_t *partition_bucketed_blocked_ell_bytes,
                                                           std::uint32_t *shard_block_size_out,
                                                           std::uint32_t *shard_bucketed_segment_count_out,
                                                           float *shard_fill_ratio_out,
                                                           std::uint64_t *shard_execution_bytes_out,
                                                           std::uint64_t *shard_bucketed_blocked_ell_bytes_out) {
    static constexpr unsigned int block_size_candidates[] = { 4u, 8u, 16u, 32u };
    static constexpr std::uint32_t sampled_row_budget = 16384u;
    std::uint32_t *exec_to_canonical_cols = 0;
    std::uint32_t *canonical_to_exec_cols = 0;
    std::uint32_t local_rows = 0u;
    std::uint64_t shard_execution_bytes = 0u;
    std::uint64_t shard_blocked_bytes = 0u;
    std::uint64_t fill_weight = 0u;
    double fill_weighted_sum = 0.0;
    sparse::coo sampled_shard;
    sparse::coo canonical_coo;
    sparse::coo permuted_coo;
    sparse::blocked_ell optimized_part;
    int device = 0;

    if (out == 0) return 0;
    clear(out);
    init(out);
    sparse::init(&sampled_shard);
    sparse::init(&canonical_coo);
    sparse::init(&permuted_coo);
    sparse::init(&optimized_part);
    if (!bucket::blocked_ell_cuda_current_device(&device)) goto done;
    if (!build_sampled_shard_coo(parts, partition_count, cols, sampled_row_budget, &sampled_shard)) goto done;
    if (!build_identity_col_maps(cols, &exec_to_canonical_cols, &canonical_to_exec_cols)) goto done;
    if (sampled_shard.rows != 0u && sampled_shard.cols != 0u && sampled_shard.nnz != 0u) {
        std::vector<std::uint32_t> sample_row_rank((std::size_t) sampled_shard.rows, 0u);
        for (std::uint32_t row = 0u; row < sampled_shard.rows; ++row) sample_row_rank[(std::size_t) row] = row;
        if (!bucket::blocked_ell_bipartite_build_column_order_cuda(&sampled_shard,
                                                                   sample_row_rank.data(),
                                                                   device,
                                                                   exec_to_canonical_cols,
                                                                   canonical_to_exec_cols)) {
            goto done;
        }
    }

    out->rows = 0u;
    out->cols = cols;
    out->nnz = 0u;
    out->partition_count = partition_count;
    out->partitions = partition_count != 0u
        ? (bucketed_blocked_ell_partition *) std::calloc((std::size_t) partition_count, sizeof(bucketed_blocked_ell_partition))
        : 0;
    out->partition_row_offsets = (std::uint32_t *) std::calloc((std::size_t) partition_count + 1u, sizeof(std::uint32_t));
    out->exec_to_canonical_cols = exec_to_canonical_cols;
    out->canonical_to_exec_cols = canonical_to_exec_cols;
    exec_to_canonical_cols = 0;
    canonical_to_exec_cols = 0;
    if ((partition_count != 0u && (out->partitions == 0 || out->partition_row_offsets == 0))
        || (cols != 0u && (out->exec_to_canonical_cols == 0 || out->canonical_to_exec_cols == 0))) {
        goto done;
    }

    for (std::uint32_t partition = 0u; partition < partition_count; ++partition) {
        const sparse::blocked_ell *canonical_part = parts != 0 ? parts[partition] : 0;
        bucketed_blocked_ell_partition *bucketed = out->partitions + partition;
        convert::blocked_ell_tune_result picked = { 0u, 0.0, 0u };
        std::uint64_t blocked_bytes = 0u;
        std::uint64_t execution_bytes = 0u;
        std::uint32_t *exec_to_canonical_rows = 0;
        std::uint32_t *canonical_to_exec_rows = 0;
        init(bucketed);
        out->partition_row_offsets[partition] = local_rows;
        sparse::clear(&canonical_coo);
        sparse::clear(&permuted_coo);
        sparse::clear(&optimized_part);
        sparse::init(&canonical_coo);
        sparse::init(&permuted_coo);
        sparse::init(&optimized_part);
        if (canonical_part == 0) goto done;
        if (!blocked_ell_to_canonical_coo(canonical_part, &canonical_coo)) goto graph_row_done;
        if (!build_identity_u32_maps(canonical_coo.rows, &exec_to_canonical_rows, &canonical_to_exec_rows)) goto graph_row_done;
        if (canonical_coo.rows != 0u && canonical_coo.cols != 0u && canonical_coo.nnz != 0u) {
            if (!bucket::blocked_ell_bipartite_build_row_order_cuda(&canonical_coo,
                                                                    out->canonical_to_exec_cols,
                                                                    device,
                                                                    exec_to_canonical_rows,
                                                                    canonical_to_exec_rows)) {
                goto graph_row_done;
            }
        }
        if (!apply_coo_permutation(&canonical_coo,
                                   canonical_to_exec_rows,
                                   out->canonical_to_exec_cols,
                                   &permuted_coo)) {
            goto graph_row_done;
        }
        if (!bucket::blocked_ell_from_coo_cuda_auto_bridge(&permuted_coo,
                                                           cols,
                                                           nullptr,
                                                           block_size_candidates,
                                                           (unsigned int) (sizeof(block_size_candidates) / sizeof(block_size_candidates[0])),
                                                           &optimized_part,
                                                           device,
                                                           &picked)) {
            goto graph_row_done;
        }
        blocked_bytes = (std::uint64_t) packed_bytes((const sparse::blocked_ell *) 0,
                                                     optimized_part.rows,
                                                     optimized_part.cols,
                                                     optimized_part.nnz,
                                                     partition_aux(&optimized_part),
                                                     sizeof(real::storage_t));
        if (!build_single_segment_execution_partition(bucketed,
                                                      &optimized_part,
                                                      exec_to_canonical_rows,
                                                      canonical_to_exec_rows,
                                                      out->exec_to_canonical_cols,
                                                      out->canonical_to_exec_cols,
                                                      &execution_bytes)) {
            goto graph_row_done;
        }
        if (partition_block_sizes != 0) partition_block_sizes[partition] = picked.block_size;
        if (partition_bucket_counts != 0) partition_bucket_counts[partition] = 1u;
        if (partition_fill_ratios != 0) partition_fill_ratios[partition] = blocked_ell_value_fill_ratio(&optimized_part);
        if (partition_execution_bytes != 0) partition_execution_bytes[partition] = execution_bytes;
        if (partition_blocked_ell_bytes != 0) partition_blocked_ell_bytes[partition] = blocked_bytes;
        if (partition_bucketed_blocked_ell_bytes != 0) partition_bucketed_blocked_ell_bytes[partition] = execution_bytes;
        local_rows += bucketed->rows;
        out->nnz += bucketed->nnz;
        shard_execution_bytes += execution_bytes;
        shard_blocked_bytes += blocked_bytes;
        fill_weight += optimized_part.rows;
        fill_weighted_sum += (double) optimized_part.rows * blocked_ell_value_fill_ratio(&optimized_part);
        std::free(exec_to_canonical_rows);
        std::free(canonical_to_exec_rows);
        exec_to_canonical_rows = 0;
        canonical_to_exec_rows = 0;
        continue;

graph_row_done:
        std::free(exec_to_canonical_rows);
        std::free(canonical_to_exec_rows);
        goto done;
    }

    out->partition_row_offsets[(std::size_t) partition_count] = local_rows;
    out->rows = local_rows;
    if (shard_block_size_out != 0 && !finalize_shard_block_size(partition_block_sizes, partition_count, shard_block_size_out)) goto done;
    if (shard_bucketed_segment_count_out != 0) *shard_bucketed_segment_count_out = partition_count;
    if (shard_fill_ratio_out != 0) *shard_fill_ratio_out = fill_weight != 0u ? (float) (fill_weighted_sum / (double) fill_weight) : 0.0f;
    if (shard_execution_bytes_out != 0) *shard_execution_bytes_out = shard_execution_bytes;
    if (shard_bucketed_blocked_ell_bytes_out != 0) *shard_bucketed_blocked_ell_bytes_out = shard_execution_bytes != 0u ? shard_execution_bytes : shard_blocked_bytes;
    sparse::clear(&sampled_shard);
    sparse::clear(&canonical_coo);
    sparse::clear(&permuted_coo);
    sparse::clear(&optimized_part);
    return 1;

done:
    std::free(exec_to_canonical_cols);
    std::free(canonical_to_exec_cols);
    sparse::clear(&sampled_shard);
    sparse::clear(&canonical_coo);
    sparse::clear(&permuted_coo);
    sparse::clear(&optimized_part);
    clear(out);
    init(out);
    return 0;
}
#endif

inline int build_bucketed_optimized_shard_from_parts(sparse::blocked_ell *const *parts,
                                                     std::uint32_t partition_count,
                                                     std::uint32_t cols,
                                                     bucketed_blocked_ell_shard *out,
                                                     std::uint32_t *partition_block_sizes,
                                                     std::uint32_t *partition_bucket_counts,
                                                     float *partition_fill_ratios,
                                                     std::uint64_t *partition_execution_bytes,
                                                     std::uint64_t *partition_blocked_ell_bytes,
                                                     std::uint64_t *partition_bucketed_blocked_ell_bytes,
                                                     std::uint32_t *shard_block_size_out,
                                                     std::uint32_t *shard_bucketed_segment_count_out,
                                                     float *shard_fill_ratio_out,
                                                     std::uint64_t *shard_execution_bytes_out,
                                                     std::uint64_t *shard_bucketed_blocked_ell_bytes_out) {
#if CELLSHARD_ENABLE_CUDA
    bucketed_blocked_ell_shard graph_shard;
    std::vector<std::uint32_t> graph_partition_block_sizes((std::size_t) partition_count, 0u);
    std::vector<std::uint32_t> graph_partition_bucket_counts((std::size_t) partition_count, 0u);
    std::vector<float> graph_partition_fill_ratios((std::size_t) partition_count, 0.0f);
    std::vector<std::uint64_t> graph_partition_execution_bytes((std::size_t) partition_count, 0u);
    std::vector<std::uint64_t> graph_partition_blocked_bytes((std::size_t) partition_count, 0u);
    std::vector<std::uint64_t> graph_partition_bucketed_bytes((std::size_t) partition_count, 0u);
    std::uint32_t graph_shard_block_size = 0u;
    std::uint32_t graph_segment_count = 0u;
    float graph_fill_ratio = 0.0f;
    std::uint64_t graph_execution_bytes = 0u;
    std::uint64_t graph_bucketed_bytes = 0u;
    init(&graph_shard);
    if (build_graph_bucketed_optimized_shard_from_parts(parts,
                                                        partition_count,
                                                        cols,
                                                        &graph_shard,
                                                        graph_partition_block_sizes.data(),
                                                        graph_partition_bucket_counts.data(),
                                                        graph_partition_fill_ratios.data(),
                                                        graph_partition_execution_bytes.data(),
                                                        graph_partition_blocked_bytes.data(),
                                                        graph_partition_bucketed_bytes.data(),
                                                        &graph_shard_block_size,
                                                        &graph_segment_count,
                                                        &graph_fill_ratio,
                                                        &graph_execution_bytes,
                                                        &graph_bucketed_bytes)) {
        std::uint32_t heuristic_shard_block_size = 0u;
        std::uint32_t heuristic_segment_count = 0u;
        float heuristic_fill_ratio = 0.0f;
        std::uint64_t heuristic_execution_bytes = 0u;
        std::uint64_t heuristic_bucketed_bytes = 0u;
        if (!build_heuristic_bucketed_optimized_shard_from_parts(parts,
                                                                 partition_count,
                                                                 cols,
                                                                 out,
                                                                 partition_block_sizes,
                                                                 partition_bucket_counts,
                                                                 partition_fill_ratios,
                                                                 partition_execution_bytes,
                                                                 partition_blocked_ell_bytes,
                                                                 partition_bucketed_blocked_ell_bytes,
                                                                 &heuristic_shard_block_size,
                                                                 &heuristic_segment_count,
                                                                 &heuristic_fill_ratio,
                                                                 &heuristic_execution_bytes,
                                                                 &heuristic_bucketed_bytes)) {
            clear(&graph_shard);
            return 0;
        }
        if (graph_execution_bytes < heuristic_execution_bytes
            || (graph_execution_bytes == heuristic_execution_bytes && graph_bucketed_bytes < heuristic_bucketed_bytes)) {
            move_bucketed_blocked_ell_shard(out, &graph_shard);
            if (partition_block_sizes != 0 && !graph_partition_block_sizes.empty()) {
                std::memcpy(partition_block_sizes,
                            graph_partition_block_sizes.data(),
                            graph_partition_block_sizes.size() * sizeof(std::uint32_t));
            }
            if (partition_bucket_counts != 0 && !graph_partition_bucket_counts.empty()) {
                std::memcpy(partition_bucket_counts,
                            graph_partition_bucket_counts.data(),
                            graph_partition_bucket_counts.size() * sizeof(std::uint32_t));
            }
            if (partition_fill_ratios != 0 && !graph_partition_fill_ratios.empty()) {
                std::memcpy(partition_fill_ratios,
                            graph_partition_fill_ratios.data(),
                            graph_partition_fill_ratios.size() * sizeof(float));
            }
            if (partition_execution_bytes != 0 && !graph_partition_execution_bytes.empty()) {
                std::memcpy(partition_execution_bytes,
                            graph_partition_execution_bytes.data(),
                            graph_partition_execution_bytes.size() * sizeof(std::uint64_t));
            }
            if (partition_blocked_ell_bytes != 0 && !graph_partition_blocked_bytes.empty()) {
                std::memcpy(partition_blocked_ell_bytes,
                            graph_partition_blocked_bytes.data(),
                            graph_partition_blocked_bytes.size() * sizeof(std::uint64_t));
            }
            if (partition_bucketed_blocked_ell_bytes != 0 && !graph_partition_bucketed_bytes.empty()) {
                std::memcpy(partition_bucketed_blocked_ell_bytes,
                            graph_partition_bucketed_bytes.data(),
                            graph_partition_bucketed_bytes.size() * sizeof(std::uint64_t));
            }
            if (shard_block_size_out != 0) *shard_block_size_out = graph_shard_block_size;
            if (shard_bucketed_segment_count_out != 0) *shard_bucketed_segment_count_out = graph_segment_count;
            if (shard_fill_ratio_out != 0) *shard_fill_ratio_out = graph_fill_ratio;
            if (shard_execution_bytes_out != 0) *shard_execution_bytes_out = graph_execution_bytes;
            if (shard_bucketed_blocked_ell_bytes_out != 0) *shard_bucketed_blocked_ell_bytes_out = graph_bucketed_bytes;
        } else {
            clear(&graph_shard);
            if (shard_block_size_out != 0) *shard_block_size_out = heuristic_shard_block_size;
            if (shard_bucketed_segment_count_out != 0) *shard_bucketed_segment_count_out = heuristic_segment_count;
            if (shard_fill_ratio_out != 0) *shard_fill_ratio_out = heuristic_fill_ratio;
            if (shard_execution_bytes_out != 0) *shard_execution_bytes_out = heuristic_execution_bytes;
            if (shard_bucketed_blocked_ell_bytes_out != 0) *shard_bucketed_blocked_ell_bytes_out = heuristic_bucketed_bytes;
        }
        return 1;
    }
#endif
    return build_heuristic_bucketed_optimized_shard_from_parts(parts,
                                                               partition_count,
                                                               cols,
                                                               out,
                                                               partition_block_sizes,
                                                               partition_bucket_counts,
                                                               partition_fill_ratios,
                                                               partition_execution_bytes,
                                                               partition_blocked_ell_bytes,
                                                               partition_bucketed_blocked_ell_bytes,
                                                               shard_block_size_out,
                                                               shard_bucketed_segment_count_out,
                                                               shard_fill_ratio_out,
                                                               shard_execution_bytes_out,
                                                               shard_bucketed_blocked_ell_bytes_out);
}

inline int write_execution_partition_blob(std::FILE *fp, const bucketed_blocked_ell_partition *part) {
    std::uint32_t segment = 0u;
    if (fp == 0 || part == 0) return 0;
    if (!write_sharded_block(fp, &part->rows, sizeof(part->rows), 1u)) return 0;
    if (!write_sharded_block(fp, &part->cols, sizeof(part->cols), 1u)) return 0;
    if (!write_sharded_block(fp, &part->nnz, sizeof(part->nnz), 1u)) return 0;
    if (!write_sharded_block(fp, &part->segment_count, sizeof(part->segment_count), 1u)) return 0;
    if (!write_sharded_block(fp, part->segment_row_offsets, sizeof(std::uint32_t), (std::size_t) part->segment_count + 1u)) return 0;
    if (!write_sharded_block(fp, part->exec_to_canonical_rows, sizeof(std::uint32_t), part->rows)) return 0;
    if (!write_sharded_block(fp, part->canonical_to_exec_rows, sizeof(std::uint32_t), part->rows)) return 0;
    if (!write_sharded_block(fp, part->exec_to_canonical_cols, sizeof(std::uint32_t), part->cols)) return 0;
    if (!write_sharded_block(fp, part->canonical_to_exec_cols, sizeof(std::uint32_t), part->cols)) return 0;
    for (segment = 0u; segment < part->segment_count; ++segment) {
        if (!::cellshard::store(fp, part->segments + segment)) return 0;
    }
    return 1;
}

inline int write_optimized_partition_blob(std::FILE *fp, const bucketed_blocked_ell_partition *part) {
    std::uint32_t segment = 0u;
    if (fp == 0 || part == 0) return 0;
    if (!write_sharded_block(fp, &part->rows, sizeof(part->rows), 1u)) return 0;
    if (!write_sharded_block(fp, &part->cols, sizeof(part->cols), 1u)) return 0;
    if (!write_sharded_block(fp, &part->nnz, sizeof(part->nnz), 1u)) return 0;
    if (!write_sharded_block(fp, &part->segment_count, sizeof(part->segment_count), 1u)) return 0;
    if (!write_packed_u32_array(fp, part->segment_row_offsets, (std::size_t) part->segment_count + 1u, 0)) return 0;
    if (!write_packed_u32_array(fp, part->exec_to_canonical_rows, part->rows, 1)) return 0;
    for (segment = 0u; segment < part->segment_count; ++segment) {
        if (!::cellshard::store(fp, part->segments + segment)) return 0;
    }
    return 1;
}

inline int load_execution_partition_blob(std::FILE *fp, bucketed_blocked_ell_partition *part) {
    std::uint32_t segment = 0u;
    if (fp == 0 || part == 0) return 0;
    clear(part);
    if (!read_sharded_block(fp, &part->rows, sizeof(part->rows), 1u)) return 0;
    if (!read_sharded_block(fp, &part->cols, sizeof(part->cols), 1u)) return 0;
    if (!read_sharded_block(fp, &part->nnz, sizeof(part->nnz), 1u)) return 0;
    if (!read_sharded_block(fp, &part->segment_count, sizeof(part->segment_count), 1u)) return 0;
    part->segments = part->segment_count != 0u ? (sparse::blocked_ell *) std::calloc((std::size_t) part->segment_count, sizeof(sparse::blocked_ell)) : 0;
    part->segment_row_offsets = (std::uint32_t *) std::calloc((std::size_t) part->segment_count + 1u, sizeof(std::uint32_t));
    part->exec_to_canonical_rows = part->rows != 0u ? (std::uint32_t *) std::calloc((std::size_t) part->rows, sizeof(std::uint32_t)) : 0;
    part->canonical_to_exec_rows = part->rows != 0u ? (std::uint32_t *) std::calloc((std::size_t) part->rows, sizeof(std::uint32_t)) : 0;
    part->exec_to_canonical_cols = part->cols != 0u ? (std::uint32_t *) std::calloc((std::size_t) part->cols, sizeof(std::uint32_t)) : 0;
    part->canonical_to_exec_cols = part->cols != 0u ? (std::uint32_t *) std::calloc((std::size_t) part->cols, sizeof(std::uint32_t)) : 0;
    if ((part->segment_count != 0u && (part->segments == 0 || part->segment_row_offsets == 0))
        || (part->rows != 0u && (part->exec_to_canonical_rows == 0 || part->canonical_to_exec_rows == 0))
        || (part->cols != 0u && (part->exec_to_canonical_cols == 0 || part->canonical_to_exec_cols == 0))) {
        clear(part);
        return 0;
    }
    for (segment = 0u; segment < part->segment_count; ++segment) sparse::init(part->segments + segment);
    if (!read_sharded_block(fp, part->segment_row_offsets, sizeof(std::uint32_t), (std::size_t) part->segment_count + 1u)) {
        clear(part);
        return 0;
    }
    if (!read_sharded_block(fp, part->exec_to_canonical_rows, sizeof(std::uint32_t), part->rows)) {
        clear(part);
        return 0;
    }
    if (!read_sharded_block(fp, part->canonical_to_exec_rows, sizeof(std::uint32_t), part->rows)) {
        clear(part);
        return 0;
    }
    if (!read_sharded_block(fp, part->exec_to_canonical_cols, sizeof(std::uint32_t), part->cols)) {
        clear(part);
        return 0;
    }
    if (!read_sharded_block(fp, part->canonical_to_exec_cols, sizeof(std::uint32_t), part->cols)) {
        clear(part);
        return 0;
    }
    for (segment = 0u; segment < part->segment_count; ++segment) {
        if (!::cellshard::load(fp, part->segments + segment)) {
            clear(part);
            return 0;
        }
    }
    return 1;
}
