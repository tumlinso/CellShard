#pragma once

inline int write_sharded_block(std::FILE *fp, const void *ptr, std::size_t elem_size, std::size_t count) {
    if (count == 0u) return 1;
    return std::fwrite(ptr, elem_size, count, fp) == count;
}

inline int read_sharded_block(std::FILE *fp, void *ptr, std::size_t elem_size, std::size_t count) {
    if (count == 0u) return 1;
    return std::fread(ptr, elem_size, count, fp) == count;
}

static const unsigned char optimized_blocked_shard_magic[8] = { 'C', 'S', 'O', 'P', 'T', 'B', '2', 0 };
static const unsigned char optimized_sliced_shard_magic[8] = { 'C', 'S', 'O', 'P', 'T', 'S', '2', 0 };

enum {
    packed_u32_identity = 0u,
    packed_u32_u8 = 1u,
    packed_u32_u16 = 2u,
    packed_u32_u32 = 3u
};

inline std::uint32_t choose_packed_u32_encoding(const std::uint32_t *values,
                                                std::size_t count,
                                                int allow_identity) {
    std::uint32_t max_value = 0u;
    std::size_t i = 0u;
    int identity = allow_identity != 0;

    if (count == 0u || values == nullptr) return allow_identity ? packed_u32_identity : packed_u32_u8;
    for (i = 0u; i < count; ++i) {
        const std::uint32_t value = values[i];
        if (identity && value != (std::uint32_t) i) identity = 0;
        if (value > max_value) max_value = value;
    }
    if (identity) return packed_u32_identity;
    if (max_value <= 0xffu) return packed_u32_u8;
    if (max_value <= 0xffffu) return packed_u32_u16;
    return packed_u32_u32;
}

inline int write_packed_u32_array(std::FILE *fp,
                                  const std::uint32_t *values,
                                  std::size_t count,
                                  int allow_identity) {
    std::uint32_t encoding = choose_packed_u32_encoding(values, count, allow_identity);
    std::size_t i = 0u;

    if (fp == 0) return 0;
    if (!write_sharded_block(fp, &encoding, sizeof(encoding), 1u)) return 0;
    if (encoding == packed_u32_identity || count == 0u) return 1;
    if (encoding == packed_u32_u8) {
        for (i = 0u; i < count; ++i) {
            const std::uint8_t value = (std::uint8_t) values[i];
            if (!write_sharded_block(fp, &value, sizeof(value), 1u)) return 0;
        }
        return 1;
    }
    if (encoding == packed_u32_u16) {
        for (i = 0u; i < count; ++i) {
            const std::uint16_t value = (std::uint16_t) values[i];
            if (!write_sharded_block(fp, &value, sizeof(value), 1u)) return 0;
        }
        return 1;
    }
    return write_sharded_block(fp, values, sizeof(std::uint32_t), count);
}

inline int read_packed_u32_array(std::FILE *fp,
                                 std::uint32_t *values,
                                 std::size_t count,
                                 int allow_identity) {
    std::uint32_t encoding = packed_u32_u32;
    std::size_t i = 0u;

    if (fp == 0 || (count != 0u && values == 0)) return 0;
    if (!read_sharded_block(fp, &encoding, sizeof(encoding), 1u)) return 0;
    if (encoding == packed_u32_identity) {
        if (!allow_identity) return 0;
        for (i = 0u; i < count; ++i) values[i] = (std::uint32_t) i;
        return 1;
    }
    if (encoding == packed_u32_u8) {
        for (i = 0u; i < count; ++i) {
            std::uint8_t value = 0u;
            if (!read_sharded_block(fp, &value, sizeof(value), 1u)) return 0;
            values[i] = (std::uint32_t) value;
        }
        return 1;
    }
    if (encoding == packed_u32_u16) {
        for (i = 0u; i < count; ++i) {
            std::uint16_t value = 0u;
            if (!read_sharded_block(fp, &value, sizeof(value), 1u)) return 0;
            values[i] = (std::uint32_t) value;
        }
        return 1;
    }
    if (encoding != packed_u32_u32) return 0;
    return read_sharded_block(fp, values, sizeof(std::uint32_t), count);
}

inline int invert_u32_permutation(std::uint32_t *inverse,
                                  const std::uint32_t *forward,
                                  std::size_t count) {
    std::size_t i = 0u;
    if ((count != 0u && (inverse == 0 || forward == 0))) return 0;
    for (i = 0u; i < count; ++i) inverse[i] = std::numeric_limits<std::uint32_t>::max();
    for (i = 0u; i < count; ++i) {
        const std::uint32_t mapped = forward[i];
        if (mapped >= count || inverse[mapped] != std::numeric_limits<std::uint32_t>::max()) return 0;
        inverse[mapped] = (std::uint32_t) i;
    }
    return 1;
}

static const char dataset_magic[] = "CSH5S1";
static const char root_group[] = "/";
static const char matrix_group[] = "/matrix";
static const char datasets_group[] = "/datasets";
static const char provenance_group[] = "/provenance";
static const char codecs_group[] = "/codecs";
static const char embedded_metadata_group[] = "/embedded_metadata";
static const char observation_metadata_group[] = "/observation_metadata";
static const char feature_metadata_group[] = "/feature_metadata";
static const char user_attributes_group[] = "/dataset_attributes";
static const char browse_group[] = "/browse";
static const char preprocess_group[] = "/preprocess";
static const char execution_group[] = "/execution";
static const char runtime_service_group[] = "/runtime_service";
static const char payload_group[] = "/payload";
static const char payload_blocked_ell_group[] = "/payload/blocked_ell";
static const char payload_quantized_blocked_ell_group[] = "/payload/quantized_blocked_ell";
static const char payload_sliced_ell_group[] = "/payload/sliced_ell";
static const char payload_optimized_blocked_ell_group[] = "/payload/optimized_blocked_ell";
static const char payload_optimized_sliced_ell_group[] = "/payload/optimized_sliced_ell";
static const char payload_layout_shard_packed[] = "shard_packed";
static const char payload_layout_optimized_blocked_ell[] = "optimized_bucketed_blocked_ell";
static const char payload_layout_optimized_sliced_ell[] = "optimized_bucketed_sliced_ell";
static const unsigned char cspack_magic[8] = { 'C', 'S', 'P', 'A', 'C', 'K', '0', '1' };
static const std::uint32_t dataset_cache_schema_version = 1u;

enum {
    dataset_cache_shard_missing = 0u,
    dataset_cache_shard_queued = 1u,
    dataset_cache_shard_building = 2u,
    dataset_cache_shard_ready = 3u,
    dataset_cache_shard_failed = 4u
};

struct dataset_h5_cache_runtime {
    std::mutex state_mutex;
    std::condition_variable state_cv;
    std::deque<unsigned long> shard_queue;
    std::thread reader_thread;
    bool reader_started;
    bool stop_requested;
    std::mutex *shard_file_mutexes;

    explicit dataset_h5_cache_runtime(std::size_t shard_count)
        : reader_started(false),
          stop_requested(false),
          shard_file_mutexes(shard_count != 0u ? new std::mutex[shard_count] : nullptr) {}

    ~dataset_h5_cache_runtime() {
        delete[] shard_file_mutexes;
    }
};

inline int build_bucketed_execution_partition(bucketed_blocked_ell_partition *out,
                                              const sparse::blocked_ell *part,
                                              std::uint32_t requested_bucket_count,
                                              std::uint64_t *bucketed_bytes_out);
inline int blocked_ell_to_canonical_coo(const sparse::blocked_ell *part,
                                        sparse::coo *out);

inline float blocked_ell_value_fill_ratio(const sparse::blocked_ell *part) {
    const std::uint64_t total_slots = part != nullptr ? (std::uint64_t) part->rows * (std::uint64_t) part->ell_cols : 0u;
    return total_slots != 0u ? (float) ((double) part->nnz / (double) total_slots) : 0.0f;
}

struct dataset_h5_state {
    hid_t file;
    std::uint64_t rows;
    std::uint64_t cols;
    std::uint64_t nnz;
    std::uint64_t num_partitions;
    std::uint64_t num_shards;
    std::uint32_t num_codecs;
    std::uint32_t matrix_family;
    int blocked_ell_optimized_payload;
    std::uint64_t *partition_block_idx_offsets;
    std::uint64_t *partition_value_offsets;
    std::uint64_t *shard_block_idx_offsets;
    std::uint64_t *shard_value_offsets;
    std::uint64_t *partition_rows;
    std::uint64_t *partition_nnz;
    std::uint64_t *partition_aux;
    std::uint64_t *partition_row_offsets;
    std::uint64_t *shard_offsets;
    std::uint64_t *partition_shard_ids;
    std::uint64_t *shard_part_begin;
    std::uint64_t *shard_part_end;
    std::uint32_t *partition_codec_ids;
    dataset_codec_descriptor *codecs;
    hid_t payload_blocked_ell;
    hid_t payload_quantized_blocked_ell;
    hid_t d_blocked_ell_block_idx;
    hid_t d_blocked_ell_values;
    std::uint64_t loaded_blocked_ell_shard_id;
    std::size_t blocked_ell_block_idx_capacity;
    std::size_t blocked_ell_value_capacity;
    types::idx_t *blocked_ell_block_idx_scratch;
    real::storage_t *blocked_ell_value_scratch;
    hid_t payload_sliced_ell;
    hid_t payload_optimized_blocked_ell;
    hid_t payload_optimized_sliced_ell;
    std::uint64_t loaded_optimized_shard_id;
    bucketed_blocked_ell_shard loaded_optimized_shard;
    std::uint64_t loaded_optimized_sliced_shard_id;
    bucketed_sliced_ell_shard loaded_optimized_sliced_shard;
    std::uint32_t preferred_base_format;
    std::uint32_t *partition_execution_formats;
    std::uint32_t *partition_blocked_ell_block_sizes;
    std::uint32_t *partition_blocked_ell_bucket_counts;
    float *partition_blocked_ell_fill_ratios;
    std::uint64_t *partition_execution_bytes;
    std::uint64_t *partition_blocked_ell_bytes;
    std::uint64_t *partition_bucketed_blocked_ell_bytes;
    std::uint32_t *partition_sliced_ell_slice_counts;
    std::uint32_t *partition_sliced_ell_slice_rows;
    std::uint64_t *partition_sliced_ell_bytes;
    std::uint64_t *partition_bucketed_sliced_ell_bytes;
    std::uint32_t *shard_execution_formats;
    std::uint32_t *shard_blocked_ell_block_sizes;
    std::uint32_t *shard_bucketed_partition_counts;
    std::uint32_t *shard_bucketed_segment_counts;
    float *shard_blocked_ell_fill_ratios;
    std::uint64_t *shard_execution_bytes;
    std::uint64_t *shard_bucketed_blocked_ell_bytes;
    std::uint32_t *shard_sliced_ell_slice_counts;
    std::uint32_t *shard_sliced_ell_slice_rows;
    std::uint64_t *shard_bucketed_sliced_ell_bytes;
    std::uint32_t *shard_preferred_pair_ids;
    std::uint32_t *shard_owner_node_ids;
    std::uint32_t *shard_owner_rank_ids;
    dataset_runtime_service_view runtime_service;
    char *cache_root;
    char *cache_instance_dir;
    char *cache_manifest_path;
    char **shard_cache_paths;
    std::FILE **shard_cache_files;
    std::uint8_t *shard_cache_state;
    std::uint32_t *shard_pin_count;
    std::uint64_t *shard_cache_bytes;
    std::uint64_t *shard_access_count;
    std::uint64_t *shard_last_access_tick;
    std::uint64_t source_size_bytes;
    std::uint64_t source_mtime_ns;
    std::uint64_t cache_budget_bytes;
    std::uint64_t cache_resident_bytes;
    std::uint64_t access_clock;
    std::uint64_t last_requested_shard;
    int cache_budget_explicit;
    int predictor_enabled;
    void *cache_runtime;
};

inline void dataset_h5_state_init(dataset_h5_state *state) {
    state->file = (hid_t) -1;
    state->rows = 0u;
    state->cols = 0u;
    state->nnz = 0u;
    state->num_partitions = 0;
    state->num_shards = 0;
    state->num_codecs = 0;
    state->matrix_family = dataset_matrix_family_none;
    state->blocked_ell_optimized_payload = 0;
    state->partition_block_idx_offsets = 0;
    state->partition_value_offsets = 0;
    state->shard_block_idx_offsets = 0;
    state->shard_value_offsets = 0;
    state->partition_rows = 0;
    state->partition_nnz = 0;
    state->partition_aux = 0;
    state->partition_row_offsets = 0;
    state->shard_offsets = 0;
    state->partition_shard_ids = 0;
    state->shard_part_begin = 0;
    state->shard_part_end = 0;
    state->partition_codec_ids = 0;
    state->codecs = 0;
    state->payload_blocked_ell = (hid_t) -1;
    state->payload_quantized_blocked_ell = (hid_t) -1;
    state->d_blocked_ell_block_idx = (hid_t) -1;
    state->d_blocked_ell_values = (hid_t) -1;
    state->loaded_blocked_ell_shard_id = std::numeric_limits<std::uint64_t>::max();
    state->blocked_ell_block_idx_capacity = 0u;
    state->blocked_ell_value_capacity = 0u;
    state->blocked_ell_block_idx_scratch = 0;
    state->blocked_ell_value_scratch = 0;
    state->payload_sliced_ell = (hid_t) -1;
    state->payload_optimized_blocked_ell = (hid_t) -1;
    state->payload_optimized_sliced_ell = (hid_t) -1;
    state->loaded_optimized_shard_id = std::numeric_limits<std::uint64_t>::max();
    init(&state->loaded_optimized_shard);
    state->loaded_optimized_sliced_shard_id = std::numeric_limits<std::uint64_t>::max();
    init(&state->loaded_optimized_sliced_shard);
    state->preferred_base_format = dataset_execution_format_unknown;
    state->partition_execution_formats = 0;
    state->partition_blocked_ell_block_sizes = 0;
    state->partition_blocked_ell_bucket_counts = 0;
    state->partition_blocked_ell_fill_ratios = 0;
    state->partition_execution_bytes = 0;
    state->partition_blocked_ell_bytes = 0;
    state->partition_bucketed_blocked_ell_bytes = 0;
    state->partition_sliced_ell_slice_counts = 0;
    state->partition_sliced_ell_slice_rows = 0;
    state->partition_sliced_ell_bytes = 0;
    state->partition_bucketed_sliced_ell_bytes = 0;
    state->shard_execution_formats = 0;
    state->shard_blocked_ell_block_sizes = 0;
    state->shard_bucketed_partition_counts = 0;
    state->shard_bucketed_segment_counts = 0;
    state->shard_blocked_ell_fill_ratios = 0;
    state->shard_execution_bytes = 0;
    state->shard_bucketed_blocked_ell_bytes = 0;
    state->shard_sliced_ell_slice_counts = 0;
    state->shard_sliced_ell_slice_rows = 0;
    state->shard_bucketed_sliced_ell_bytes = 0;
    state->shard_preferred_pair_ids = 0;
    state->shard_owner_node_ids = 0;
    state->shard_owner_rank_ids = 0;
    init(&state->runtime_service);
    state->cache_root = 0;
    state->cache_instance_dir = 0;
    state->cache_manifest_path = 0;
    state->shard_cache_paths = 0;
    state->shard_cache_files = 0;
    state->shard_cache_state = 0;
    state->shard_pin_count = 0;
    state->shard_cache_bytes = 0;
    state->shard_access_count = 0;
    state->shard_last_access_tick = 0;
    state->source_size_bytes = 0u;
    state->source_mtime_ns = 0u;
    state->cache_budget_bytes = 0u;
    state->cache_resident_bytes = 0u;
    state->access_clock = 0u;
    state->last_requested_shard = std::numeric_limits<std::uint64_t>::max();
    state->cache_budget_explicit = 0;
    state->predictor_enabled = 1;
    state->cache_runtime = 0;
}

void close_dataset_h5_backend(shard_storage *s);

inline void dataset_h5_state_clear(dataset_h5_state *state) {
    unsigned long shard_i = 0ul;
    if (state != 0 && state->cache_runtime != 0) {
        dataset_h5_cache_runtime *runtime = (dataset_h5_cache_runtime *) state->cache_runtime;
        {
            std::lock_guard<std::mutex> lock(runtime->state_mutex);
            runtime->stop_requested = true;
            runtime->state_cv.notify_all();
        }
        if (runtime->reader_started && runtime->reader_thread.joinable()) runtime->reader_thread.join();
    }
    if (state != 0 && state->shard_cache_files != 0) {
        for (shard_i = 0; shard_i < (unsigned long) state->num_shards; ++shard_i) {
            if (state->shard_cache_files[shard_i] != 0) std::fclose(state->shard_cache_files[shard_i]);
        }
    }
    if (state->d_blocked_ell_values >= 0) H5Dclose(state->d_blocked_ell_values);
    if (state->d_blocked_ell_block_idx >= 0) H5Dclose(state->d_blocked_ell_block_idx);
    if (state->payload_blocked_ell >= 0) H5Gclose(state->payload_blocked_ell);
    if (state->payload_quantized_blocked_ell >= 0) H5Gclose(state->payload_quantized_blocked_ell);
    if (state->payload_sliced_ell >= 0) H5Gclose(state->payload_sliced_ell);
    if (state->payload_optimized_blocked_ell >= 0) H5Gclose(state->payload_optimized_blocked_ell);
    if (state->payload_optimized_sliced_ell >= 0) H5Gclose(state->payload_optimized_sliced_ell);
    if (state->file >= 0) H5Fclose(state->file);
    state->file = (hid_t) -1;
    std::free(state->partition_block_idx_offsets);
    std::free(state->partition_value_offsets);
    std::free(state->shard_block_idx_offsets);
    std::free(state->shard_value_offsets);
    std::free(state->partition_rows);
    std::free(state->partition_nnz);
    std::free(state->partition_aux);
    std::free(state->partition_row_offsets);
    std::free(state->shard_offsets);
    std::free(state->partition_shard_ids);
    std::free(state->shard_part_begin);
    std::free(state->shard_part_end);
    std::free(state->partition_codec_ids);
    std::free(state->codecs);
    std::free(state->blocked_ell_block_idx_scratch);
    std::free(state->blocked_ell_value_scratch);
    clear(&state->loaded_optimized_shard);
    clear(&state->loaded_optimized_sliced_shard);
    std::free(state->partition_blocked_ell_block_sizes);
    std::free(state->partition_execution_formats);
    std::free(state->partition_blocked_ell_bucket_counts);
    std::free(state->partition_blocked_ell_fill_ratios);
    std::free(state->partition_execution_bytes);
    std::free(state->partition_blocked_ell_bytes);
    std::free(state->partition_bucketed_blocked_ell_bytes);
    std::free(state->partition_sliced_ell_slice_counts);
    std::free(state->partition_sliced_ell_slice_rows);
    std::free(state->partition_sliced_ell_bytes);
    std::free(state->partition_bucketed_sliced_ell_bytes);
    std::free(state->shard_blocked_ell_block_sizes);
    std::free(state->shard_execution_formats);
    std::free(state->shard_bucketed_partition_counts);
    std::free(state->shard_bucketed_segment_counts);
    std::free(state->shard_blocked_ell_fill_ratios);
    std::free(state->shard_execution_bytes);
    std::free(state->shard_bucketed_blocked_ell_bytes);
    std::free(state->shard_sliced_ell_slice_counts);
    std::free(state->shard_sliced_ell_slice_rows);
    std::free(state->shard_bucketed_sliced_ell_bytes);
    std::free(state->shard_preferred_pair_ids);
    std::free(state->shard_owner_node_ids);
    std::free(state->shard_owner_rank_ids);
    if (state->shard_cache_paths != 0) {
        for (shard_i = 0; shard_i < (unsigned long) state->num_shards; ++shard_i) std::free(state->shard_cache_paths[shard_i]);
    }
    std::free(state->cache_root);
    std::free(state->cache_instance_dir);
    std::free(state->cache_manifest_path);
    std::free(state->shard_cache_paths);
    std::free(state->shard_cache_files);
    std::free(state->shard_cache_state);
    std::free(state->shard_pin_count);
    std::free(state->shard_cache_bytes);
    std::free(state->shard_access_count);
    std::free(state->shard_last_access_tick);
    delete (dataset_h5_cache_runtime *) state->cache_runtime;
    state->partition_block_idx_offsets = 0;
    state->partition_value_offsets = 0;
    state->shard_block_idx_offsets = 0;
    state->shard_value_offsets = 0;
    state->partition_rows = 0;
    state->partition_nnz = 0;
    state->partition_aux = 0;
    state->partition_row_offsets = 0;
    state->shard_offsets = 0;
    state->partition_shard_ids = 0;
    state->shard_part_begin = 0;
    state->shard_part_end = 0;
    state->partition_codec_ids = 0;
    state->codecs = 0;
    state->payload_blocked_ell = (hid_t) -1;
    state->payload_quantized_blocked_ell = (hid_t) -1;
    state->d_blocked_ell_block_idx = (hid_t) -1;
    state->d_blocked_ell_values = (hid_t) -1;
    state->loaded_blocked_ell_shard_id = std::numeric_limits<std::uint64_t>::max();
    state->blocked_ell_block_idx_capacity = 0u;
    state->blocked_ell_value_capacity = 0u;
    state->blocked_ell_block_idx_scratch = 0;
    state->blocked_ell_value_scratch = 0;
    state->payload_sliced_ell = (hid_t) -1;
    state->payload_optimized_blocked_ell = (hid_t) -1;
    state->payload_optimized_sliced_ell = (hid_t) -1;
    state->loaded_optimized_shard_id = std::numeric_limits<std::uint64_t>::max();
    init(&state->loaded_optimized_shard);
    state->loaded_optimized_sliced_shard_id = std::numeric_limits<std::uint64_t>::max();
    init(&state->loaded_optimized_sliced_shard);
    state->preferred_base_format = dataset_execution_format_unknown;
    state->partition_execution_formats = 0;
    state->partition_blocked_ell_block_sizes = 0;
    state->partition_blocked_ell_bucket_counts = 0;
    state->partition_blocked_ell_fill_ratios = 0;
    state->partition_execution_bytes = 0;
    state->partition_blocked_ell_bytes = 0;
    state->partition_bucketed_blocked_ell_bytes = 0;
    state->partition_sliced_ell_slice_counts = 0;
    state->partition_sliced_ell_slice_rows = 0;
    state->partition_sliced_ell_bytes = 0;
    state->partition_bucketed_sliced_ell_bytes = 0;
    state->shard_execution_formats = 0;
    state->shard_blocked_ell_block_sizes = 0;
    state->shard_bucketed_partition_counts = 0;
    state->shard_bucketed_segment_counts = 0;
    state->shard_blocked_ell_fill_ratios = 0;
    state->shard_execution_bytes = 0;
    state->shard_bucketed_blocked_ell_bytes = 0;
    state->shard_sliced_ell_slice_counts = 0;
    state->shard_sliced_ell_slice_rows = 0;
    state->shard_bucketed_sliced_ell_bytes = 0;
    state->shard_preferred_pair_ids = 0;
    state->shard_owner_node_ids = 0;
    state->shard_owner_rank_ids = 0;
    init(&state->runtime_service);
    state->cache_root = 0;
    state->cache_instance_dir = 0;
    state->cache_manifest_path = 0;
    state->shard_cache_paths = 0;
    state->shard_cache_files = 0;
    state->shard_cache_state = 0;
    state->shard_pin_count = 0;
    state->shard_cache_bytes = 0;
    state->shard_access_count = 0;
    state->shard_last_access_tick = 0;
    state->source_size_bytes = 0u;
    state->source_mtime_ns = 0u;
    state->cache_budget_bytes = 0u;
    state->cache_resident_bytes = 0u;
    state->access_clock = 0u;
    state->last_requested_shard = std::numeric_limits<std::uint64_t>::max();
    state->cache_budget_explicit = 0;
    state->predictor_enabled = 1;
    state->cache_runtime = 0;
    state->rows = 0u;
    state->cols = 0u;
    state->nnz = 0u;
    state->num_partitions = 0;
    state->num_shards = 0;
    state->num_codecs = 0;
    state->matrix_family = dataset_matrix_family_none;
    state->blocked_ell_optimized_payload = 0;
}

inline int blocked_ell_uses_execution_payload(const dataset_h5_state *state) {
    return state != nullptr && state->matrix_family == dataset_matrix_family_blocked_ell
        && state->blocked_ell_optimized_payload != 0;
}

inline hid_t create_group(hid_t parent, const char *path) {
    hid_t group = H5Gcreate2(parent, path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (group >= 0) return group;
    return H5Gopen2(parent, path, H5P_DEFAULT);
}

inline hid_t open_optional_group(hid_t parent, const char *path) {
    if (parent < 0 || path == 0) return (hid_t) -1;
    if (H5Lexists(parent, path, H5P_DEFAULT) <= 0) return (hid_t) -1;
    return H5Gopen2(parent, path, H5P_DEFAULT);
}

inline int dataset_exists(hid_t parent, const char *name) {
    return parent >= 0 && name != 0 && H5Lexists(parent, name, H5P_DEFAULT) > 0;
}

inline int write_attr_u64(hid_t obj, const char *name, std::uint64_t value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = (hid_t) -1;
    int ok = 0;

    if (space < 0) return 0;
    attr = H5Acreate2(obj, name, H5T_NATIVE_UINT64, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) goto done;
    ok = H5Awrite(attr, H5T_NATIVE_UINT64, &value) >= 0;

done:
    if (attr >= 0) H5Aclose(attr);
    H5Sclose(space);
    return ok;
}

inline int write_attr_u32(hid_t obj, const char *name, std::uint32_t value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = (hid_t) -1;
    int ok = 0;

    if (space < 0) return 0;
    attr = H5Acreate2(obj, name, H5T_NATIVE_UINT32, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) goto done;
    ok = H5Awrite(attr, H5T_NATIVE_UINT32, &value) >= 0;

done:
    if (attr >= 0) H5Aclose(attr);
    H5Sclose(space);
    return ok;
}

inline int write_attr_f32(hid_t obj, const char *name, float value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = (hid_t) -1;
    int ok = 0;

    if (space < 0) return 0;
    attr = H5Acreate2(obj, name, H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) goto done;
    ok = H5Awrite(attr, H5T_NATIVE_FLOAT, &value) >= 0;

done:
    if (attr >= 0) H5Aclose(attr);
    H5Sclose(space);
    return ok;
}

inline int write_attr_f64(hid_t obj, const char *name, double value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = (hid_t) -1;
    int ok = 0;

    if (space < 0) return 0;
    attr = H5Acreate2(obj, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) goto done;
    ok = H5Awrite(attr, H5T_NATIVE_DOUBLE, &value) >= 0;

done:
    if (attr >= 0) H5Aclose(attr);
    H5Sclose(space);
    return ok;
}

inline int write_attr_string(hid_t obj, const char *name, const char *value) {
    hid_t type = H5Tcopy(H5T_C_S1);
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = (hid_t) -1;
    int ok = 0;

    if (type < 0 || space < 0) goto done;
    if (H5Tset_size(type, std::strlen(value) + 1u) < 0) goto done;
    attr = H5Acreate2(obj, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) goto done;
    ok = H5Awrite(attr, type, value) >= 0;

done:
    if (attr >= 0) H5Aclose(attr);
    if (space >= 0) H5Sclose(space);
    if (type >= 0) H5Tclose(type);
    return ok;
}

inline int read_attr_u64(hid_t obj, const char *name, std::uint64_t *value) {
    hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
    int ok = 0;
    if (attr < 0) return 0;
    ok = H5Aread(attr, H5T_NATIVE_UINT64, value) >= 0;
    H5Aclose(attr);
    return ok;
}

inline int read_attr_u32(hid_t obj, const char *name, std::uint32_t *value) {
    hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
    int ok = 0;
    if (attr < 0) return 0;
    ok = H5Aread(attr, H5T_NATIVE_UINT32, value) >= 0;
    H5Aclose(attr);
    return ok;
}

inline int read_optional_attr_u64(hid_t obj, const char *name, std::uint64_t *value) {
    if (H5Aexists(obj, name) <= 0) return 1;
    return read_attr_u64(obj, name, value);
}

inline int read_optional_attr_u32(hid_t obj, const char *name, std::uint32_t *value) {
    if (H5Aexists(obj, name) <= 0) return 1;
    return read_attr_u32(obj, name, value);
}

inline int read_attr_string(hid_t obj, const char *name, char *dst, std::size_t cap) {
    hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
    hid_t type = (hid_t) -1;
    std::size_t size = 0;
    int ok = 0;

    if (attr < 0 || dst == 0 || cap == 0) return 0;
    type = H5Aget_type(attr);
    if (type < 0) goto done;
    size = H5Tget_size(type);
    if (size + 1u > cap) goto done;
    std::memset(dst, 0, cap);
    ok = H5Aread(attr, type, dst) >= 0;

done:
    if (type >= 0) H5Tclose(type);
    H5Aclose(attr);
    return ok;
}

inline int write_dataset_1d(hid_t parent,
                            const char *name,
                            hid_t dtype,
                            hsize_t count,
                            const void *data) {
    hid_t space = (hid_t) -1;
    hid_t dset = (hid_t) -1;
    int ok = 0;

    space = H5Screate_simple(1, &count, 0);
    if (space < 0) return 0;
    dset = H5Dcreate2(parent, name, dtype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) goto done;
    if (data != 0) ok = H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0;
    else ok = 1;

done:
    if (dset >= 0) H5Dclose(dset);
    H5Sclose(space);
    return ok;
}

inline int read_dataset_1d(hid_t parent,
                           const char *name,
                           hid_t dtype,
                           std::uint64_t expected_count,
                           void *data) {
    hid_t dset = H5Dopen2(parent, name, H5P_DEFAULT);
    hid_t space = (hid_t) -1;
    hsize_t dims[1] = {0};
    int ndims = 0;
    int ok = 0;
    if (dset < 0 || (expected_count != 0u && data == 0)) return 0;
    space = H5Dget_space(dset);
    if (space < 0) goto done;
    ndims = H5Sget_simple_extent_dims(space, dims, 0);
    if (ndims != 1 || dims[0] != (hsize_t) expected_count) goto done;
    if (expected_count == 0u) {
        ok = 1;
        goto done;
    }
    ok = H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0;

done:
    if (space >= 0) H5Sclose(space);
    H5Dclose(dset);
    return ok;
}

inline int write_text_column(hid_t group, const char *name, const dataset_text_column_view *column) {
    hid_t sub = (hid_t) -1;
    int ok = 0;

    if (column == 0) return 1;
    sub = create_group(group, name);
    if (sub < 0) return 0;
    if (!write_attr_u32(sub, "count", column->count)) goto done;
    if (!write_attr_u32(sub, "bytes", column->bytes)) goto done;
    if (!write_dataset_1d(sub, "offsets", H5T_NATIVE_UINT32, (hsize_t) column->count + 1u, column->offsets)) goto done;
    if (!write_dataset_1d(sub, "data", H5T_NATIVE_CHAR, (hsize_t) column->bytes, column->data)) goto done;
    ok = 1;

done:
    if (sub >= 0) H5Gclose(sub);
    return ok;
}

struct owned_text_column {
    std::vector<std::uint32_t> offsets;
    std::vector<char> data;

    dataset_text_column_view view() const {
        dataset_text_column_view out{};
        out.count = offsets.empty() ? 0u : (std::uint32_t) offsets.size() - 1u;
        out.bytes = (std::uint32_t) data.size();
        out.offsets = offsets.empty() ? nullptr : offsets.data();
        out.data = data.empty() ? nullptr : data.data();
        return out;
    }
};

inline int read_text_column(hid_t group, const char *name, owned_text_column *out) {
    hid_t sub = (hid_t) -1;
    std::uint32_t count = 0u;
    std::uint32_t bytes = 0u;

    if (group < 0 || name == 0 || out == 0) return 0;
    sub = H5Gopen2(group, name, H5P_DEFAULT);
    if (sub < 0) return 0;
    if (!read_attr_u32(sub, "count", &count) || !read_attr_u32(sub, "bytes", &bytes)) {
        H5Gclose(sub);
        return 0;
    }
    out->offsets.assign((std::size_t) count + 1u, 0u);
    out->data.assign((std::size_t) bytes, 0);
    if (!read_dataset_1d(sub, "offsets", H5T_NATIVE_UINT32, (std::uint64_t) count + 1u, out->offsets.data())
        || (bytes != 0u && !read_dataset_1d(sub, "data", H5T_NATIVE_CHAR, bytes, out->data.data()))) {
        H5Gclose(sub);
        return 0;
    }
    H5Gclose(sub);
    return 1;
}

inline const char *text_column_value(const owned_text_column &column, std::uint32_t idx) {
    if (idx + 1u >= column.offsets.size() || column.data.empty()) return "";
    return column.data.data() + column.offsets[idx];
}

inline void append_text_value(owned_text_column *column, const char *value) {
    const char *src = value != nullptr ? value : "";
    const std::size_t len = std::strlen(src);
    if (column->offsets.empty()) column->offsets.push_back(0u);
    column->data.insert(column->data.end(), src, src + len);
    column->data.push_back(0);
    column->offsets.push_back((std::uint32_t) column->data.size());
}

inline int ensure_magic(hid_t file) {
    char got[32];
    if (!read_attr_string(file, "cellshard_magic", got, sizeof(got))) return 0;
    return std::strcmp(got, dataset_magic) == 0;
}

inline int ensure_schema_version(hid_t file) {
    std::uint32_t got = 0u;
    if (!read_attr_u32(file, "schema_version", &got)) return 0;
    return got == dataset_h5_schema_version;
}

inline int ensure_dataset_identity(hid_t file) {
    return ensure_magic(file) && ensure_schema_version(file);
}

inline int fail_dataset_validation(const char *filename, const char *message) {
    std::fprintf(stderr,
                 "cellshard: invalid dataset metadata in %s: %s\n",
                 filename != 0 ? filename : "<memory>",
                 message != 0 ? message : "validation failed");
    return 0;
}

inline int fail_dataset_validation_u64(const char *filename,
                                       const char *field,
                                       std::uint64_t got,
                                       std::uint64_t expected) {
    std::fprintf(stderr,
                 "cellshard: invalid dataset metadata in %s: %s=%llu expected=%llu\n",
                 filename != 0 ? filename : "<memory>",
                 field != 0 ? field : "value",
                 (unsigned long long) got,
                 (unsigned long long) expected);
    return 0;
}

inline int validate_dataset_header_scalars(const char *filename,
                                           std::uint64_t rows,
                                           std::uint64_t cols,
                                           std::uint64_t nnz,
                                           std::uint64_t num_codecs) {
    if (rows > (std::uint64_t) std::numeric_limits<unsigned long>::max()) {
        return fail_dataset_validation(filename, "rows exceeds local sharded index width");
    }
    if (cols > (std::uint64_t) std::numeric_limits<unsigned long>::max()) {
        return fail_dataset_validation(filename, "cols exceeds local sharded index width");
    }
    if (nnz > (std::uint64_t) std::numeric_limits<unsigned long>::max()) {
        return fail_dataset_validation(filename, "nnz exceeds local sharded index width");
    }
    if (num_codecs > (std::uint64_t) std::numeric_limits<std::uint32_t>::max()) {
        return fail_dataset_validation(filename, "num_codecs exceeds u32 codec table width");
    }
    return 1;
}

inline int validate_dataset_layout_tables(const char *filename,
                                          std::uint64_t rows,
                                          std::uint64_t nnz,
                                          std::uint64_t num_partitions,
                                          std::uint64_t num_shards,
                                          const std::uint64_t *partition_rows,
                                          const std::uint64_t *partition_nnz,
                                          const std::uint64_t *partition_row_offsets,
                                          const std::uint64_t *shard_offsets) {
    std::uint64_t partition_id = 0u;
    std::uint64_t sum_rows = 0u;
    std::uint64_t sum_nnz = 0u;
    std::uint64_t boundary = 0u;

    if ((num_partitions != 0u) && (partition_rows == 0 || partition_nnz == 0 || partition_row_offsets == 0)) {
        return fail_dataset_validation(filename, "partition tables are missing");
    }
    if (shard_offsets == 0) return fail_dataset_validation(filename, "shard offsets are missing");
    if (num_partitions == 0u && rows != 0u) {
        return fail_dataset_validation(filename, "rows is non-zero but no partitions were recorded");
    }
    if (num_shards == 0u && rows != 0u) {
        return fail_dataset_validation(filename, "rows is non-zero but no shards were recorded");
    }
    if (partition_row_offsets[0] != 0u) {
        return fail_dataset_validation_u64(filename, "partition_row_offsets[0]", partition_row_offsets[0], 0u);
    }
    for (partition_id = 0u; partition_id < num_partitions; ++partition_id) {
        const std::uint64_t begin = partition_row_offsets[partition_id];
        const std::uint64_t end = partition_row_offsets[partition_id + 1u];
        const std::uint64_t part_rows = partition_rows[partition_id];
        if (begin > end) {
            return fail_dataset_validation(filename, "partition_row_offsets is not monotonic");
        }
        if (end - begin != part_rows) {
            return fail_dataset_validation(filename, "partition_rows does not match partition_row_offsets");
        }
        if (part_rows > rows || sum_rows > rows - part_rows) {
            return fail_dataset_validation(filename, "partition_rows exceeds top-level rows");
        }
        if (partition_nnz[partition_id] > nnz || sum_nnz > nnz - partition_nnz[partition_id]) {
            return fail_dataset_validation(filename, "partition_nnz exceeds top-level nnz");
        }
        sum_rows += part_rows;
        sum_nnz += partition_nnz[partition_id];
    }
    if (partition_row_offsets[num_partitions] != rows) {
        return fail_dataset_validation_u64(filename, "partition_row_offsets[last]", partition_row_offsets[num_partitions], rows);
    }
    if (sum_rows != rows) {
        return fail_dataset_validation_u64(filename, "sum(partition_rows)", sum_rows, rows);
    }
    if (sum_nnz != nnz) {
        return fail_dataset_validation_u64(filename, "sum(partition_nnz)", sum_nnz, nnz);
    }
    if (shard_offsets[0] != 0u) {
        return fail_dataset_validation_u64(filename, "shard_offsets[0]", shard_offsets[0], 0u);
    }
    for (partition_id = 0u; partition_id < num_shards; ++partition_id) {
        const std::uint64_t begin = shard_offsets[partition_id];
        const std::uint64_t end = shard_offsets[partition_id + 1u];
        if (begin > end || end > rows) {
            return fail_dataset_validation(filename, "shard_offsets is not monotonic or exceeds rows");
        }
        while (boundary <= num_partitions && partition_row_offsets[boundary] < begin) ++boundary;
        if (boundary > num_partitions || partition_row_offsets[boundary] != begin) {
            return fail_dataset_validation(filename, "shard begin does not align to a partition boundary");
        }
        while (boundary <= num_partitions && partition_row_offsets[boundary] < end) ++boundary;
        if (boundary > num_partitions || partition_row_offsets[boundary] != end) {
            return fail_dataset_validation(filename, "shard end does not align to a partition boundary");
        }
    }
    if (shard_offsets[num_shards] != rows) {
        return fail_dataset_validation_u64(filename, "shard_offsets[last]", shard_offsets[num_shards], rows);
    }
    return 1;
}

inline int validate_partition_codec_ids(const char *filename,
                                        std::uint64_t num_partitions,
                                        const std::uint32_t *partition_codec_ids,
                                        std::uint32_t num_codecs,
                                        const dataset_codec_descriptor *codecs) {
    std::uint64_t partition_id = 0u;
    std::uint32_t codec_index = 0u;
    if (num_partitions == 0u) return 1;
    if (partition_codec_ids == 0) return fail_dataset_validation(filename, "partition codec ids are missing");
    if (num_codecs == 0u || codecs == 0) return fail_dataset_validation(filename, "codec table is missing");
    for (partition_id = 0u; partition_id < num_partitions; ++partition_id) {
        int found = 0;
        for (codec_index = 0u; codec_index < num_codecs; ++codec_index) {
            if (codecs[codec_index].codec_id == partition_codec_ids[partition_id]) {
                found = 1;
                break;
            }
        }
        if (!found) {
            return fail_dataset_validation(filename, "partition codec id is not present in the codec table");
        }
    }
    return 1;
}

template<typename MatrixT>
inline int validate_loaded_sharded_header(const char *filename,
                                          const sharded<MatrixT> *m,
                                          unsigned long expected_rows,
                                          unsigned long expected_nnz) {
    if (m == 0) return 0;
    if (m->rows != expected_rows) {
        return fail_dataset_validation(filename, "derived partition rows disagree with top-level rows");
    }
    if (m->nnz != expected_nnz) {
        return fail_dataset_validation(filename, "derived partition nnz disagrees with top-level nnz");
    }
    return 1;
}

inline int read_hyperslab_1d(hid_t dataset,
                             hid_t dtype,
                             std::uint64_t offset,
                             std::uint64_t count,
                             void *dst) {
    hsize_t off[1];
    hsize_t dims[1];
    hid_t filespace = (hid_t) -1;
    hid_t memspace = (hid_t) -1;
    int ok = 0;

    if (count == 0) return 1;
    off[0] = (hsize_t) offset;
    dims[0] = (hsize_t) count;
    filespace = H5Dget_space(dataset);
    if (filespace < 0) return 0;
    if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, off, 0, dims, 0) < 0) goto done;
    memspace = H5Screate_simple(1, dims, 0);
    if (memspace < 0) goto done;
    ok = H5Dread(dataset, dtype, memspace, filespace, H5P_DEFAULT, dst) >= 0;

done:
    if (memspace >= 0) H5Sclose(memspace);
    if (filespace >= 0) H5Sclose(filespace);
    return ok;
}

inline int write_blob_dataset(hid_t parent,
                              const char *name,
                              const unsigned char *data,
                              std::size_t bytes) {
    return write_dataset_1d(parent,
                            name,
                            H5T_NATIVE_UCHAR,
                            (hsize_t) bytes,
                            bytes != 0u ? data : 0);
}

inline int read_blob_dataset(hid_t parent,
                             const char *name,
                             std::vector<unsigned char> *out) {
    hid_t dset = (hid_t) -1;
    hid_t space = (hid_t) -1;
    hsize_t dims[1] = {0};
    int ndims = 0;
    if (parent < 0 || name == 0 || out == 0) return 0;
    dset = H5Dopen2(parent, name, H5P_DEFAULT);
    if (dset < 0) return 0;
    space = H5Dget_space(dset);
    if (space < 0) goto done;
    ndims = H5Sget_simple_extent_dims(space, dims, 0);
    if (ndims != 1) goto done;
    out->assign((std::size_t) dims[0], 0u);
    if (!out->empty() && H5Dread(dset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, out->data()) < 0) goto done;
    H5Sclose(space);
    H5Dclose(dset);
    return 1;

done:
    if (space >= 0) H5Sclose(space);
    if (dset >= 0) H5Dclose(dset);
    return 0;
}

inline int build_optimized_shard_dataset_name(unsigned long shard_id,
                                              char *name,
                                              std::size_t cap) {
    if (name == 0 || cap == 0u) return 0;
    return std::snprintf(name, cap, "shard.%lu", shard_id) > 0;
}

inline int build_partition_blob_dataset_name(unsigned long partition_id,
                                             char *name,
                                             std::size_t cap) {
    if (name == 0 || cap == 0u) return 0;
    return std::snprintf(name, cap, "part.%lu", partition_id) > 0;
}
