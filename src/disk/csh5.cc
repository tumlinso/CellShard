#include "csh5.cuh"

#include "../convert/blocked_ell_from_compressed.cuh"
#include "../sharded/disk.cuh"
#include "../sharded/sharded_host.cuh"

#include <hdf5.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <limits>
#include <vector>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>

namespace cellshard {

namespace {

inline int write_sharded_block(std::FILE *fp, const void *ptr, std::size_t elem_size, std::size_t count) {
    if (count == 0u) return 1;
    return std::fwrite(ptr, elem_size, count, fp) == count;
}

inline int read_sharded_block(std::FILE *fp, void *ptr, std::size_t elem_size, std::size_t count) {
    if (count == 0u) return 1;
    return std::fread(ptr, elem_size, count, fp) == count;
}

static const char dataset_magic[] = "CSH5S1";
static const char root_group[] = "/";
static const char matrix_group[] = "/matrix";
static const char datasets_group[] = "/datasets";
static const char provenance_group[] = "/provenance";
static const char codecs_group[] = "/codecs";
static const char embedded_metadata_group[] = "/embedded_metadata";
static const char observation_metadata_group[] = "/observation_metadata";
static const char browse_group[] = "/browse";
static const char preprocess_group[] = "/preprocess";
static const char execution_group[] = "/execution";
static const char runtime_service_group[] = "/runtime_service";
static const char payload_group[] = "/payload";
static const char payload_standard_group[] = "/payload/standard_csr";
static const char payload_blocked_ell_group[] = "/payload/blocked_ell";
static const char payload_sliced_ell_group[] = "/payload/sliced_ell";
static const char payload_optimized_blocked_ell_group[] = "/payload/optimized_blocked_ell";
static const char payload_optimized_sliced_ell_group[] = "/payload/optimized_sliced_ell";
static const char payload_layout_shard_packed[] = "shard_packed";
static const char payload_layout_optimized_blocked_ell[] = "optimized_bucketed_blocked_ell";
static const char payload_layout_optimized_sliced_ell[] = "optimized_bucketed_sliced_ell";
static const unsigned char execution_pack_magic[8] = { 'C', 'S', 'E', 'P', 'A', 'C', 'K', '1' };
static const std::uint32_t dataset_cache_schema_version = 1u;
static const std::uint64_t shard_pack_payload_alignment = 4096u;

enum {
    dataset_matrix_family_none = 0u,
    dataset_matrix_family_compressed = 1u,
    dataset_matrix_family_blocked_ell = 2u,
    dataset_matrix_family_optimized_blocked_ell = 3u,
    dataset_matrix_family_sliced_ell = 4u
};

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

struct dataset_h5_state {
    hid_t file;
    std::uint64_t rows;
    std::uint64_t cols;
    std::uint64_t nnz;
    std::uint64_t num_partitions;
    std::uint64_t num_shards;
    std::uint32_t num_codecs;
    std::uint32_t matrix_family;
    std::uint64_t *partition_indptr_offsets;
    std::uint64_t *partition_nnz_offsets;
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
    hid_t payload_standard;
    hid_t d_standard_indptr;
    hid_t d_standard_indices;
    hid_t d_standard_values;
    hid_t payload_blocked_ell;
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
    state->partition_indptr_offsets = 0;
    state->partition_nnz_offsets = 0;
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
    state->payload_standard = (hid_t) -1;
    state->d_standard_indptr = (hid_t) -1;
    state->d_standard_indices = (hid_t) -1;
    state->d_standard_values = (hid_t) -1;
    state->payload_blocked_ell = (hid_t) -1;
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
    if (state->d_standard_values >= 0) H5Dclose(state->d_standard_values);
    if (state->d_standard_indices >= 0) H5Dclose(state->d_standard_indices);
    if (state->d_standard_indptr >= 0) H5Dclose(state->d_standard_indptr);
    if (state->payload_standard >= 0) H5Gclose(state->payload_standard);
    if (state->d_blocked_ell_values >= 0) H5Dclose(state->d_blocked_ell_values);
    if (state->d_blocked_ell_block_idx >= 0) H5Dclose(state->d_blocked_ell_block_idx);
    if (state->payload_blocked_ell >= 0) H5Gclose(state->payload_blocked_ell);
    if (state->payload_sliced_ell >= 0) H5Gclose(state->payload_sliced_ell);
    if (state->payload_optimized_blocked_ell >= 0) H5Gclose(state->payload_optimized_blocked_ell);
    if (state->payload_optimized_sliced_ell >= 0) H5Gclose(state->payload_optimized_sliced_ell);
    if (state->file >= 0) H5Fclose(state->file);
    state->file = (hid_t) -1;
    std::free(state->partition_indptr_offsets);
    std::free(state->partition_nnz_offsets);
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
    state->partition_indptr_offsets = 0;
    state->partition_nnz_offsets = 0;
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
    state->payload_standard = (hid_t) -1;
    state->d_standard_indptr = (hid_t) -1;
    state->d_standard_indices = (hid_t) -1;
    state->d_standard_values = (hid_t) -1;
    state->payload_blocked_ell = (hid_t) -1;
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
                           void *data) {
    hid_t dset = H5Dopen2(parent, name, H5P_DEFAULT);
    int ok = 0;
    if (dset < 0) return 0;
    ok = H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0;
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

inline int ensure_magic(hid_t file) {
    char got[32];
    if (!read_attr_string(file, "cellshard_magic", got, sizeof(got))) return 0;
    return std::strcmp(got, dataset_magic) == 0;
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

inline std::size_t standard_csr_part_bytes(std::uint64_t rows, std::uint64_t nnz) {
    return (std::size_t) (rows + 1u) * sizeof(types::ptr_t)
        + (std::size_t) nnz * sizeof(types::idx_t)
        + (std::size_t) nnz * sizeof(real::storage_t);
}

inline std::size_t blocked_ell_part_block_index_count(std::uint64_t rows, std::uint64_t aux) {
    const types::u32 block_size = sparse::unpack_blocked_ell_block_size((unsigned long) aux);
    const std::uint64_t ell_width = sparse::unpack_blocked_ell_ell_width((unsigned long) aux);
    return block_size == 0u ? 0u : ((rows + block_size - 1u) / block_size) * ell_width;
}

inline std::size_t blocked_ell_part_value_count(std::uint64_t rows, std::uint64_t aux) {
    return (std::size_t) rows * (std::size_t) sparse::unpack_blocked_ell_cols((unsigned long) aux);
}

inline std::size_t sliced_ell_part_total_slots(std::uint64_t aux) {
    return (std::size_t) sparse::unpack_sliced_ell_total_slots((unsigned long) aux);
}

inline std::uint64_t local_dim_limit() {
    return (std::uint64_t) std::numeric_limits<types::dim_t>::max();
}

inline std::uint64_t local_nnz_limit() {
    return (std::uint64_t) std::numeric_limits<types::nnz_t>::max();
}

inline std::uint64_t local_index_limit() {
    return (std::uint64_t) std::numeric_limits<types::idx_t>::max();
}

inline int fail_dataset_u32_limit(const char *filename,
                                 const char *scope,
                                 std::uint64_t id,
                                 const char *field,
                                 std::uint64_t value,
                                 std::uint64_t limit) {
    std::fprintf(stderr,
                 "cellshard: %s exceeds the current u32 execution limit while writing %s (%s=%llu, %s=%llu, limit=%llu)\n",
                 scope != 0 ? scope : "dataset payload",
                 filename != 0 ? filename : "<memory>",
                 scope != 0 && std::strcmp(scope, "part") == 0 ? "partition_id" : "id",
                 (unsigned long long) id,
                 field != 0 ? field : "value",
                 (unsigned long long) value,
                 (unsigned long long) limit);
    return 0;
}

inline void warn_dataset_u32_limit(const char *filename,
                                  const char *scope,
                                  std::uint64_t id,
                                  const char *field,
                                  std::uint64_t value,
                                  std::uint64_t limit) {
    std::fprintf(stderr,
                 "cellshard: warning: %s exceeds the current u32 execution limit for %s (%s=%llu, %s=%llu, limit=%llu)\n",
                 scope != 0 ? scope : "dataset payload",
                 filename != 0 ? filename : "<memory>",
                 scope != 0 && std::strcmp(scope, "shard") == 0 ? "shard_id" : "id",
                 (unsigned long long) id,
                 field != 0 ? field : "value",
                 (unsigned long long) value,
                 (unsigned long long) limit);
}

inline int reserve_blocked_ell_shard_scratch(dataset_h5_state *state,
                                             std::size_t block_idx_count,
                                             std::size_t value_count) {
    types::idx_t *block_idx = state != 0 ? state->blocked_ell_block_idx_scratch : 0;
    real::storage_t *values = state != 0 ? state->blocked_ell_value_scratch : 0;

    if (state == 0) return 0;
    if (block_idx_count > state->blocked_ell_block_idx_capacity) {
        block_idx = (types::idx_t *) std::realloc(state->blocked_ell_block_idx_scratch,
                                                  block_idx_count * sizeof(types::idx_t));
        if (block_idx == 0) return 0;
        state->blocked_ell_block_idx_scratch = block_idx;
        state->blocked_ell_block_idx_capacity = block_idx_count;
    }
    if (value_count > state->blocked_ell_value_capacity) {
        values = (real::storage_t *) std::realloc(state->blocked_ell_value_scratch,
                                                  value_count * sizeof(real::storage_t));
        if (values == 0) return 0;
        state->blocked_ell_value_scratch = values;
        state->blocked_ell_value_capacity = value_count;
    }
    return 1;
}

int open_dataset_h5_backend(shard_storage *s);
void close_dataset_h5_backend(shard_storage *s);

inline int ensure_directory_exists(const char *path) {
    struct stat st;
    if (path == 0 || *path == 0) return 0;
    if (::stat(path, &st) == 0) return S_ISDIR(st.st_mode) ? 1 : 0;
    if (::mkdir(path, 0775) == 0) return 1;
    if (errno == EEXIST) return 1;
    return 0;
}

inline int assign_owned_string(char **dst, const char *src) {
    char *copy = 0;
    std::size_t len = 0u;

    if (dst == 0) return 0;
    std::free(*dst);
    *dst = 0;
    if (src == 0) return 1;
    len = std::strlen(src);
    copy = (char *) std::malloc(len + 1u);
    if (copy == 0) return 0;
    std::memcpy(copy, src, len + 1u);
    *dst = copy;
    return 1;
}

inline int build_default_cache_root(const char *source_path, char *path, std::size_t cap) {
    const char *slash = 0;
    int written = 0;
    if (source_path == 0 || path == 0 || cap == 0u) return 0;
    slash = std::strrchr(source_path, '/');
    if (slash == 0) return std::snprintf(path, cap, ".cellshard_cache") > 0;
    written = std::snprintf(path, cap, "%.*s/.cellshard_cache", (int) (slash - source_path), source_path);
    return written > 0 && (std::size_t) written < cap;
}

inline std::uint64_t fnv1a_mix(std::uint64_t h, const void *data, std::size_t bytes) {
    const unsigned char *ptr = (const unsigned char *) data;
    std::size_t i = 0u;
    for (i = 0u; i < bytes; ++i) {
        h ^= (std::uint64_t) ptr[i];
        h *= 1099511628211ull;
    }
    return h;
}

inline std::uint64_t build_source_fingerprint_u64(const char *source_path,
                                                  std::uint64_t size_bytes,
                                                  std::uint64_t mtime_ns,
                                                  std::uint32_t matrix_family,
                                                  std::uint64_t num_partitions,
                                                  std::uint64_t num_shards) {
    const std::uint32_t schema_version = dataset_h5_schema_version;
    std::uint64_t h = 1469598103934665603ull;
    if (source_path != 0) h = fnv1a_mix(h, source_path, std::strlen(source_path));
    h = fnv1a_mix(h, &size_bytes, sizeof(size_bytes));
    h = fnv1a_mix(h, &mtime_ns, sizeof(mtime_ns));
    h = fnv1a_mix(h, &matrix_family, sizeof(matrix_family));
    h = fnv1a_mix(h, &num_partitions, sizeof(num_partitions));
    h = fnv1a_mix(h, &num_shards, sizeof(num_shards));
    h = fnv1a_mix(h, &schema_version, sizeof(schema_version));
    h = fnv1a_mix(h, dataset_magic, std::strlen(dataset_magic));
    return h;
}

inline int build_cache_instance_path(const char *cache_root,
                                     std::uint64_t fingerprint,
                                     char *path,
                                     std::size_t cap) {
    if (cache_root == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/instances/%016llx", cache_root, (unsigned long long) fingerprint) > 0;
}

inline int build_cache_instances_root_path(const char *cache_root,
                                           char *path,
                                           std::size_t cap) {
    if (cache_root == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/instances", cache_root) > 0;
}

inline int build_cache_metadata_dir_path(const char *cache_instance_dir,
                                         char *path,
                                         std::size_t cap) {
    if (cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/metadata", cache_instance_dir) > 0;
}

inline int build_cache_pack_root_path(const char *cache_instance_dir,
                                      char *path,
                                      std::size_t cap) {
    if (cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/packs", cache_instance_dir) > 0;
}

inline int build_cache_canonical_pack_dir_path(const char *cache_instance_dir,
                                               char *path,
                                               std::size_t cap) {
    if (cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/packs/canonical", cache_instance_dir) > 0;
}

inline int build_cache_execution_pack_dir_path(const char *cache_instance_dir,
                                               char *path,
                                               std::size_t cap) {
    if (cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/packs/execution", cache_instance_dir) > 0;
}

inline int build_cache_manifest_path(const char *cache_instance_dir,
                                     char *path,
                                     std::size_t cap) {
    if (cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/metadata/manifest.txt", cache_instance_dir) > 0;
}

inline int build_shard_pack_path(const dataset_h5_state *state,
                                 unsigned long shard_id,
                                 char *path,
                                 std::size_t cap) {
    if (state == 0 || state->cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/packs/canonical/shard.%lu.pack", state->cache_instance_dir, shard_id) > 0;
}

inline int build_shard_pack_temp_path(const dataset_h5_state *state,
                                      unsigned long shard_id,
                                      char *path,
                                      std::size_t cap) {
    if (state == 0 || state->cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/packs/canonical/shard.%lu.pack.tmp", state->cache_instance_dir, shard_id) > 0;
}

inline int build_execution_pack_path(const dataset_h5_state *state,
                                     unsigned long shard_id,
                                     char *path,
                                     std::size_t cap) {
    if (state == 0 || state->cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/packs/execution/shard.%lu.exec.pack", state->cache_instance_dir, shard_id) > 0;
}

inline int build_execution_pack_temp_path(const dataset_h5_state *state,
                                          unsigned long shard_id,
                                          char *path,
                                          std::size_t cap) {
    if (state == 0 || state->cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/packs/execution/shard.%lu.exec.pack.tmp", state->cache_instance_dir, shard_id) > 0;
}

inline std::uint64_t sharded_pack_payload_offset(std::uint64_t partition_count,
                                                 std::uint64_t shard_count,
                                                 std::uint64_t payload_alignment) {
    std::uint64_t offset = 8u
        + sizeof(unsigned char)
        + 7u
        + sizeof(std::uint64_t) * 7u
        + sizeof(std::uint64_t) * partition_count * 3u
        + sizeof(std::uint64_t) * (shard_count + 1u)
        + sizeof(std::uint64_t) * partition_count * 2u;
    offset = (offset + payload_alignment - 1u) & ~(payload_alignment - 1u);
    return offset;
}

template<typename MatrixT>
inline int compute_shard_pack_locators(const std::uint64_t *partition_rows,
                                       const std::uint64_t *partition_nnz,
                                       const std::uint64_t *partition_aux,
                                       std::uint64_t cols,
                                       std::uint64_t partition_count,
                                       std::uint64_t *partition_offsets,
                                       std::uint64_t *part_sizes) {
    std::uint64_t cursor = sharded_pack_payload_offset(partition_count, 1u, shard_pack_payload_alignment);
    std::uint64_t i = 0u;
    if ((partition_count != 0u) && (partition_rows == 0 || partition_nnz == 0 || partition_offsets == 0 || part_sizes == 0)) return 0;
    for (i = 0u; i < partition_count; ++i) {
        const std::size_t bytes = packed_bytes((const MatrixT *) 0,
                                               (types::dim_t) partition_rows[i],
                                               (types::dim_t) cols,
                                               (types::nnz_t) partition_nnz[i],
                                               partition_aux != 0 ? (unsigned long) partition_aux[i] : 0ul,
                                               sizeof(real::storage_t));
        partition_offsets[i] = cursor;
        part_sizes[i] = (std::uint64_t) bytes;
        cursor += (std::uint64_t) bytes;
        cursor = (cursor + shard_pack_payload_alignment - 1u) & ~(shard_pack_payload_alignment - 1u);
    }
    return 1;
}

inline dataset_h5_cache_runtime *cache_runtime(dataset_h5_state *state) {
    return state != 0 ? (dataset_h5_cache_runtime *) state->cache_runtime : 0;
}

inline int refresh_dataset_source_stat(const char *source_path,
                                      dataset_h5_state *state) {
    struct stat st;
    if (source_path == 0 || state == 0) return 0;
    if (::stat(source_path, &st) != 0) return 0;
    state->source_size_bytes = (std::uint64_t) st.st_size;
#if defined(__linux__)
    state->source_mtime_ns = (std::uint64_t) st.st_mtim.tv_sec * 1000000000ull + (std::uint64_t) st.st_mtim.tv_nsec;
#else
    state->source_mtime_ns = (std::uint64_t) st.st_mtime * 1000000000ull;
#endif
    return 1;
}

inline int ensure_cache_tracking_allocated(dataset_h5_state *state) {
    std::size_t count = 0u;
    if (state == 0) return 0;
    if (state->cache_runtime == 0) {
        state->cache_runtime = new dataset_h5_cache_runtime((std::size_t) state->num_shards);
        if (state->cache_runtime == 0) return 0;
    }
    count = (std::size_t) state->num_shards;
    if (count == 0u) return 1;
    if (state->shard_cache_paths == 0) state->shard_cache_paths = (char **) std::calloc(count, sizeof(char *));
    if (state->shard_cache_files == 0) state->shard_cache_files = (std::FILE **) std::calloc(count, sizeof(std::FILE *));
    if (state->shard_cache_state == 0) state->shard_cache_state = (std::uint8_t *) std::calloc(count, sizeof(std::uint8_t));
    if (state->shard_pin_count == 0) state->shard_pin_count = (std::uint32_t *) std::calloc(count, sizeof(std::uint32_t));
    if (state->shard_cache_bytes == 0) state->shard_cache_bytes = (std::uint64_t *) std::calloc(count, sizeof(std::uint64_t));
    if (state->shard_access_count == 0) state->shard_access_count = (std::uint64_t *) std::calloc(count, sizeof(std::uint64_t));
    if (state->shard_last_access_tick == 0) state->shard_last_access_tick = (std::uint64_t *) std::calloc(count, sizeof(std::uint64_t));
    return state->shard_cache_paths != 0
        && state->shard_cache_files != 0
        && state->shard_cache_state != 0
        && state->shard_pin_count != 0
        && state->shard_cache_bytes != 0
        && state->shard_access_count != 0
        && state->shard_last_access_tick != 0;
}

inline int ensure_execution_metadata_allocated(dataset_h5_state *state) {
    if (state == 0) return 0;
    if (state->num_partitions != 0u) {
        if (state->partition_execution_formats == 0) {
            state->partition_execution_formats = (std::uint32_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint32_t));
        }
        if (state->partition_blocked_ell_block_sizes == 0) {
            state->partition_blocked_ell_block_sizes = (std::uint32_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint32_t));
        }
        if (state->partition_blocked_ell_bucket_counts == 0) {
            state->partition_blocked_ell_bucket_counts = (std::uint32_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint32_t));
        }
        if (state->partition_blocked_ell_fill_ratios == 0) {
            state->partition_blocked_ell_fill_ratios = (float *) std::calloc((std::size_t) state->num_partitions, sizeof(float));
        }
        if (state->partition_execution_bytes == 0) {
            state->partition_execution_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
        }
        if (state->partition_blocked_ell_bytes == 0) {
            state->partition_blocked_ell_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
        }
        if (state->partition_bucketed_blocked_ell_bytes == 0) {
            state->partition_bucketed_blocked_ell_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
        }
        if (state->partition_sliced_ell_slice_counts == 0) {
            state->partition_sliced_ell_slice_counts = (std::uint32_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint32_t));
        }
        if (state->partition_sliced_ell_bytes == 0) {
            state->partition_sliced_ell_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
        }
        if (state->partition_bucketed_sliced_ell_bytes == 0) {
            state->partition_bucketed_sliced_ell_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
        }
    }
    if (state->num_shards != 0u) {
        if (state->shard_execution_formats == 0) {
            state->shard_execution_formats = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
        if (state->shard_blocked_ell_block_sizes == 0) {
            state->shard_blocked_ell_block_sizes = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
        if (state->shard_bucketed_partition_counts == 0) {
            state->shard_bucketed_partition_counts = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
        if (state->shard_bucketed_segment_counts == 0) {
            state->shard_bucketed_segment_counts = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
        if (state->shard_blocked_ell_fill_ratios == 0) {
            state->shard_blocked_ell_fill_ratios = (float *) std::calloc((std::size_t) state->num_shards, sizeof(float));
        }
        if (state->shard_execution_bytes == 0) {
            state->shard_execution_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint64_t));
        }
        if (state->shard_bucketed_blocked_ell_bytes == 0) {
            state->shard_bucketed_blocked_ell_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint64_t));
        }
        if (state->shard_sliced_ell_slice_counts == 0) {
            state->shard_sliced_ell_slice_counts = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
        if (state->shard_bucketed_sliced_ell_bytes == 0) {
            state->shard_bucketed_sliced_ell_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint64_t));
        }
        if (state->shard_preferred_pair_ids == 0) {
            state->shard_preferred_pair_ids = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
        if (state->shard_owner_node_ids == 0) {
            state->shard_owner_node_ids = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
        if (state->shard_owner_rank_ids == 0) {
            state->shard_owner_rank_ids = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
    }
    return (state->num_partitions == 0u
            || (state->partition_execution_formats != 0
                && state->partition_blocked_ell_block_sizes != 0
                && state->partition_blocked_ell_bucket_counts != 0
                && state->partition_blocked_ell_fill_ratios != 0
                && state->partition_execution_bytes != 0
                && state->partition_blocked_ell_bytes != 0
                && state->partition_bucketed_blocked_ell_bytes != 0
                && state->partition_sliced_ell_slice_counts != 0
                && state->partition_sliced_ell_bytes != 0
                && state->partition_bucketed_sliced_ell_bytes != 0))
        && (state->num_shards == 0u
            || (state->shard_execution_formats != 0
                && state->shard_blocked_ell_block_sizes != 0
                && state->shard_bucketed_partition_counts != 0
                && state->shard_bucketed_segment_counts != 0
                && state->shard_blocked_ell_fill_ratios != 0
                && state->shard_execution_bytes != 0
                && state->shard_bucketed_blocked_ell_bytes != 0
                && state->shard_sliced_ell_slice_counts != 0
                && state->shard_bucketed_sliced_ell_bytes != 0
                && state->shard_preferred_pair_ids != 0
                && state->shard_owner_node_ids != 0
                && state->shard_owner_rank_ids != 0));
}

inline void default_execution_metadata(dataset_h5_state *state) {
    std::uint64_t shard_id = 0u;
    std::uint64_t partition_id = 0u;
    const std::uint32_t default_format =
        state != 0 && state->matrix_family == dataset_matrix_family_optimized_blocked_ell
            ? dataset_execution_format_bucketed_blocked_ell
            : (state != 0 && state->matrix_family == dataset_matrix_family_sliced_ell
                   ? dataset_execution_format_sliced_ell
                   : dataset_execution_format_blocked_ell);
    if (state == 0) return;
    state->preferred_base_format = default_format;
    for (partition_id = 0u; partition_id < state->num_partitions; ++partition_id) {
        state->partition_execution_formats[partition_id] = default_format;
        state->partition_blocked_ell_block_sizes[partition_id] = 0u;
        state->partition_blocked_ell_bucket_counts[partition_id] = 1u;
        state->partition_blocked_ell_fill_ratios[partition_id] = 0.0f;
        state->partition_execution_bytes[partition_id] = 0u;
        state->partition_blocked_ell_bytes[partition_id] = 0u;
        state->partition_bucketed_blocked_ell_bytes[partition_id] = 0u;
        state->partition_sliced_ell_slice_counts[partition_id] = 0u;
        state->partition_sliced_ell_bytes[partition_id] = 0u;
        state->partition_bucketed_sliced_ell_bytes[partition_id] = 0u;
    }
    for (shard_id = 0u; shard_id < state->num_shards; ++shard_id) {
        state->shard_execution_formats[shard_id] = default_format;
        state->shard_blocked_ell_block_sizes[shard_id] = 0u;
        state->shard_bucketed_partition_counts[shard_id] =
            state->shard_part_end != 0 && state->shard_part_begin != 0
                ? (std::uint32_t) (state->shard_part_end[shard_id] - state->shard_part_begin[shard_id])
                : 0u;
        state->shard_bucketed_segment_counts[shard_id] = state->shard_bucketed_partition_counts[shard_id];
        state->shard_blocked_ell_fill_ratios[shard_id] = 0.0f;
        state->shard_execution_bytes[shard_id] = 0u;
        state->shard_bucketed_blocked_ell_bytes[shard_id] = 0u;
        state->shard_sliced_ell_slice_counts[shard_id] = 0u;
        state->shard_bucketed_sliced_ell_bytes[shard_id] = 0u;
        state->shard_preferred_pair_ids[shard_id] = 0u;
        state->shard_owner_node_ids[shard_id] = 0u;
        state->shard_owner_rank_ids[shard_id] = 0u;
    }
}

inline int load_dataset_execution_metadata(hid_t file, dataset_h5_state *state) {
    hid_t execution = (hid_t) -1;
    if (state == 0) return 0;
    if (!ensure_execution_metadata_allocated(state)) return 0;
    default_execution_metadata(state);
    execution = open_optional_group(file, execution_group);
    if (execution < 0) return 1;
    if (!read_optional_attr_u32(execution, "preferred_base_format", &state->preferred_base_format)) goto done;
    if (state->num_partitions != 0u) {
        if (dataset_exists(execution, "partition_execution_formats")
            && !read_dataset_1d(execution, "partition_execution_formats", H5T_NATIVE_UINT32, state->partition_execution_formats)) goto done;
        if (dataset_exists(execution, "partition_blocked_ell_block_sizes")
            && !read_dataset_1d(execution, "partition_blocked_ell_block_sizes", H5T_NATIVE_UINT32, state->partition_blocked_ell_block_sizes)) goto done;
        if (dataset_exists(execution, "partition_blocked_ell_bucket_counts")
            && !read_dataset_1d(execution, "partition_blocked_ell_bucket_counts", H5T_NATIVE_UINT32, state->partition_blocked_ell_bucket_counts)) goto done;
        if (dataset_exists(execution, "partition_blocked_ell_fill_ratios")
            && !read_dataset_1d(execution, "partition_blocked_ell_fill_ratios", H5T_NATIVE_FLOAT, state->partition_blocked_ell_fill_ratios)) goto done;
        if (dataset_exists(execution, "partition_execution_bytes")
            && !read_dataset_1d(execution, "partition_execution_bytes", H5T_NATIVE_UINT64, state->partition_execution_bytes)) goto done;
        if (dataset_exists(execution, "partition_blocked_ell_bytes")
            && !read_dataset_1d(execution, "partition_blocked_ell_bytes", H5T_NATIVE_UINT64, state->partition_blocked_ell_bytes)) goto done;
        if (dataset_exists(execution, "partition_bucketed_blocked_ell_bytes")
            && !read_dataset_1d(execution, "partition_bucketed_blocked_ell_bytes", H5T_NATIVE_UINT64, state->partition_bucketed_blocked_ell_bytes)) goto done;
        if (dataset_exists(execution, "partition_sliced_ell_slice_counts")
            && !read_dataset_1d(execution, "partition_sliced_ell_slice_counts", H5T_NATIVE_UINT32, state->partition_sliced_ell_slice_counts)) goto done;
        if (dataset_exists(execution, "partition_sliced_ell_bytes")
            && !read_dataset_1d(execution, "partition_sliced_ell_bytes", H5T_NATIVE_UINT64, state->partition_sliced_ell_bytes)) goto done;
        if (dataset_exists(execution, "partition_bucketed_sliced_ell_bytes")
            && !read_dataset_1d(execution, "partition_bucketed_sliced_ell_bytes", H5T_NATIVE_UINT64, state->partition_bucketed_sliced_ell_bytes)) goto done;
    }
    if (state->num_shards != 0u) {
        if (dataset_exists(execution, "shard_execution_formats")
            && !read_dataset_1d(execution, "shard_execution_formats", H5T_NATIVE_UINT32, state->shard_execution_formats)) goto done;
        if (dataset_exists(execution, "shard_blocked_ell_block_sizes")
            && !read_dataset_1d(execution, "shard_blocked_ell_block_sizes", H5T_NATIVE_UINT32, state->shard_blocked_ell_block_sizes)) goto done;
        if (dataset_exists(execution, "shard_bucketed_partition_counts")
            && !read_dataset_1d(execution, "shard_bucketed_partition_counts", H5T_NATIVE_UINT32, state->shard_bucketed_partition_counts)) goto done;
        if (dataset_exists(execution, "shard_bucketed_segment_counts")
            && !read_dataset_1d(execution, "shard_bucketed_segment_counts", H5T_NATIVE_UINT32, state->shard_bucketed_segment_counts)) goto done;
        if (dataset_exists(execution, "shard_blocked_ell_fill_ratios")
            && !read_dataset_1d(execution, "shard_blocked_ell_fill_ratios", H5T_NATIVE_FLOAT, state->shard_blocked_ell_fill_ratios)) goto done;
        if (dataset_exists(execution, "shard_execution_bytes")
            && !read_dataset_1d(execution, "shard_execution_bytes", H5T_NATIVE_UINT64, state->shard_execution_bytes)) goto done;
        if (dataset_exists(execution, "shard_bucketed_blocked_ell_bytes")
            && !read_dataset_1d(execution, "shard_bucketed_blocked_ell_bytes", H5T_NATIVE_UINT64, state->shard_bucketed_blocked_ell_bytes)) goto done;
        if (dataset_exists(execution, "shard_sliced_ell_slice_counts")
            && !read_dataset_1d(execution, "shard_sliced_ell_slice_counts", H5T_NATIVE_UINT32, state->shard_sliced_ell_slice_counts)) goto done;
        if (dataset_exists(execution, "shard_bucketed_sliced_ell_bytes")
            && !read_dataset_1d(execution, "shard_bucketed_sliced_ell_bytes", H5T_NATIVE_UINT64, state->shard_bucketed_sliced_ell_bytes)) goto done;
        if (dataset_exists(execution, "shard_preferred_pair_ids")
            && !read_dataset_1d(execution, "shard_preferred_pair_ids", H5T_NATIVE_UINT32, state->shard_preferred_pair_ids)) goto done;
        if (dataset_exists(execution, "shard_owner_node_ids")
            && !read_dataset_1d(execution, "shard_owner_node_ids", H5T_NATIVE_UINT32, state->shard_owner_node_ids)) goto done;
        if (dataset_exists(execution, "shard_owner_rank_ids")
            && !read_dataset_1d(execution, "shard_owner_rank_ids", H5T_NATIVE_UINT32, state->shard_owner_rank_ids)) goto done;
    }
    H5Gclose(execution);
    return 1;

done:
    if (execution >= 0) H5Gclose(execution);
    return 0;
}

inline void default_runtime_service_metadata(dataset_h5_state *state) {
    if (state == 0) return;
    init(&state->runtime_service);
    state->runtime_service.service_mode = dataset_runtime_service_mode_local_cache;
    state->runtime_service.live_write_mode = dataset_live_write_mode_read_only;
    state->runtime_service.prefer_pack_delivery = 1u;
    state->runtime_service.remote_pack_delivery = 0u;
    state->runtime_service.single_reader_coordinator = 0u;
    state->runtime_service.maintenance_lock_blocks_overwrite = 1u;
    state->runtime_service.canonical_generation = 1u;
    state->runtime_service.execution_plan_generation = 1u;
    state->runtime_service.pack_generation = 1u;
    state->runtime_service.service_epoch = 1u;
    state->runtime_service.active_read_generation = 1u;
    state->runtime_service.staged_write_generation = 1u;
}

inline int load_dataset_runtime_service_metadata(hid_t file, dataset_h5_state *state) {
    hid_t runtime = (hid_t) -1;
    if (state == 0) return 0;
    default_runtime_service_metadata(state);
    runtime = open_optional_group(file, runtime_service_group);
    if (runtime < 0) return 1;
    if (!read_optional_attr_u32(runtime, "service_mode", &state->runtime_service.service_mode)
        || !read_optional_attr_u32(runtime, "live_write_mode", &state->runtime_service.live_write_mode)
        || !read_optional_attr_u32(runtime, "prefer_pack_delivery", &state->runtime_service.prefer_pack_delivery)
        || !read_optional_attr_u32(runtime, "remote_pack_delivery", &state->runtime_service.remote_pack_delivery)
        || !read_optional_attr_u32(runtime, "single_reader_coordinator", &state->runtime_service.single_reader_coordinator)
        || !read_optional_attr_u32(runtime, "maintenance_lock_blocks_overwrite", &state->runtime_service.maintenance_lock_blocks_overwrite)
        || !read_optional_attr_u64(runtime, "canonical_generation", &state->runtime_service.canonical_generation)
        || !read_optional_attr_u64(runtime, "execution_plan_generation", &state->runtime_service.execution_plan_generation)
        || !read_optional_attr_u64(runtime, "pack_generation", &state->runtime_service.pack_generation)
        || !read_optional_attr_u64(runtime, "service_epoch", &state->runtime_service.service_epoch)
        || !read_optional_attr_u64(runtime, "active_read_generation", &state->runtime_service.active_read_generation)
        || !read_optional_attr_u64(runtime, "staged_write_generation", &state->runtime_service.staged_write_generation)) {
        goto done;
    }
    H5Gclose(runtime);
    return 1;

done:
    if (runtime >= 0) H5Gclose(runtime);
    return 0;
}

inline std::uint64_t estimate_shard_pack_bytes(const dataset_h5_state *state, unsigned long shard_id) {
    std::uint64_t begin = 0u;
    std::uint64_t end = 0u;
    std::uint64_t local_count = 0u;
    std::uint64_t rows = 0u;
    std::uint64_t nnz = 0u;
    std::uint64_t *local_offsets = 0;
    std::uint64_t *local_sizes = 0;
    std::uint64_t bytes = 0u;
    std::uint64_t i = 0u;

    if (state == 0 || shard_id >= state->num_shards || state->shard_part_begin == 0 || state->shard_part_end == 0) return 0u;
    begin = state->shard_part_begin[shard_id];
    end = state->shard_part_end[shard_id];
    local_count = end >= begin ? (end - begin) : 0u;
    if (local_count == 0u) return sharded_pack_payload_offset(0u, 1u, shard_pack_payload_alignment);
    local_offsets = (std::uint64_t *) std::calloc((std::size_t) local_count, sizeof(std::uint64_t));
    local_sizes = (std::uint64_t *) std::calloc((std::size_t) local_count, sizeof(std::uint64_t));
    if (local_offsets == 0 || local_sizes == 0) goto done;
    for (i = begin; i < end; ++i) {
        rows += state->partition_rows[i];
        nnz += state->partition_nnz[i];
    }
    if (state->matrix_family == dataset_matrix_family_compressed) {
        if (!compute_shard_pack_locators<sparse::compressed>(state->partition_rows + begin,
                                                             state->partition_nnz + begin,
                                                             state->partition_aux + begin,
                                                             state->cols,
                                                             local_count,
                                                             local_offsets,
                                                             local_sizes)) {
            goto done;
        }
    } else if (state->matrix_family == dataset_matrix_family_blocked_ell) {
        if (!compute_shard_pack_locators<sparse::blocked_ell>(state->partition_rows + begin,
                                                              state->partition_nnz + begin,
                                                              state->partition_aux + begin,
                                                              state->cols,
                                                              local_count,
                                                              local_offsets,
                                                              local_sizes)) {
            goto done;
        }
    } else if (state->matrix_family == dataset_matrix_family_optimized_blocked_ell) {
        if (!compute_shard_pack_locators<sparse::blocked_ell>(state->partition_rows + begin,
                                                              state->partition_nnz + begin,
                                                              state->partition_aux + begin,
                                                              state->cols,
                                                              local_count,
                                                              local_offsets,
                                                              local_sizes)) {
            goto done;
        }
    } else if (state->matrix_family == dataset_matrix_family_sliced_ell) {
        if (!compute_shard_pack_locators<sparse::sliced_ell>(state->partition_rows + begin,
                                                             state->partition_nnz + begin,
                                                             state->partition_aux + begin,
                                                             state->cols,
                                                             local_count,
                                                             local_offsets,
                                                             local_sizes)) {
            goto done;
        }
    } else {
        goto done;
    }
    bytes = sharded_pack_payload_offset(local_count, 1u, shard_pack_payload_alignment);
    if (local_count != 0u) bytes = local_offsets[local_count - 1u] + local_sizes[local_count - 1u];
    (void) rows;
    (void) nnz;

done:
    std::free(local_offsets);
    std::free(local_sizes);
    return bytes;
}

inline int write_dataset_cache_manifest(const char *source_path,
                                       const dataset_h5_state *state) {
    std::FILE *fp = 0;
    unsigned long shard_id = 0ul;
    char canonical_pack_dir[4096];
    char execution_pack_dir[4096];
    if (source_path == 0 || state == 0 || state->cache_manifest_path == 0) return 0;
    fp = std::fopen(state->cache_manifest_path, "wb");
    if (fp == 0) return 0;
    canonical_pack_dir[0] = '\0';
    execution_pack_dir[0] = '\0';
    if (state->cache_instance_dir != 0) {
        (void) build_cache_canonical_pack_dir_path(state->cache_instance_dir, canonical_pack_dir, sizeof(canonical_pack_dir));
        (void) build_cache_execution_pack_dir_path(state->cache_instance_dir, execution_pack_dir, sizeof(execution_pack_dir));
    }
    std::fprintf(fp, "cache_schema_version=%u\n", (unsigned int) dataset_cache_schema_version);
    std::fprintf(fp, "source_path=%s\n", source_path);
    std::fprintf(fp, "cache_root=%s\n", state->cache_root != 0 ? state->cache_root : "");
    std::fprintf(fp, "cache_instance_dir=%s\n", state->cache_instance_dir != 0 ? state->cache_instance_dir : "");
    std::fprintf(fp, "canonical_pack_dir=%s\n", canonical_pack_dir);
    std::fprintf(fp, "execution_pack_dir=%s\n", execution_pack_dir);
    std::fprintf(fp, "source_size_bytes=%llu\n", (unsigned long long) state->source_size_bytes);
    std::fprintf(fp, "source_mtime_ns=%llu\n", (unsigned long long) state->source_mtime_ns);
    std::fprintf(fp, "matrix_family=%u\n", (unsigned int) state->matrix_family);
    std::fprintf(fp, "service_mode=%u\n", (unsigned int) state->runtime_service.service_mode);
    std::fprintf(fp, "live_write_mode=%u\n", (unsigned int) state->runtime_service.live_write_mode);
    std::fprintf(fp, "canonical_generation=%llu\n", (unsigned long long) state->runtime_service.canonical_generation);
    std::fprintf(fp, "execution_plan_generation=%llu\n", (unsigned long long) state->runtime_service.execution_plan_generation);
    std::fprintf(fp, "pack_generation=%llu\n", (unsigned long long) state->runtime_service.pack_generation);
    std::fprintf(fp, "service_epoch=%llu\n", (unsigned long long) state->runtime_service.service_epoch);
    std::fprintf(fp, "num_partitions=%llu\n", (unsigned long long) state->num_partitions);
    std::fprintf(fp, "num_shards=%llu\n", (unsigned long long) state->num_shards);
    for (shard_id = 0ul; shard_id < (unsigned long) state->num_shards; ++shard_id) {
        std::fprintf(fp,
                     "shard.%lu=%llu,%llu,%llu\n",
                     shard_id,
                     (unsigned long long) state->shard_part_begin[shard_id],
                     (unsigned long long) state->shard_part_end[shard_id],
                     (unsigned long long) estimate_shard_pack_bytes(state, shard_id));
    }
    std::fclose(fp);
    return 1;
}

inline int ensure_dataset_cache_layout(shard_storage *s) {
    dataset_h5_state *state = 0;
    char path[4096];
    char instances_root[4096];
    char metadata_dir[4096];
    char pack_root_dir[4096];
    char canonical_pack_dir[4096];
    char execution_pack_dir[4096];
    std::uint64_t fingerprint = 0u;
    unsigned long shard_id = 0ul;
    struct statvfs vfs;

    if (s == 0 || s->backend != shard_storage_backend_dataset_h5 || s->backend_state == 0 || s->source_path == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
    if (!refresh_dataset_source_stat(s->source_path, state)) return 0;
    if (state->cache_root == 0) {
        if (!build_default_cache_root(s->source_path, path, sizeof(path))) return 0;
        if (!assign_owned_string(&state->cache_root, path)) return 0;
    }
    if (!ensure_directory_exists(state->cache_root)) return 0;
    if (!build_cache_instances_root_path(state->cache_root, instances_root, sizeof(instances_root))) return 0;
    if (!ensure_directory_exists(instances_root)) return 0;
    fingerprint = build_source_fingerprint_u64(s->source_path,
                                               state->source_size_bytes,
                                               state->source_mtime_ns,
                                               state->matrix_family,
                                               state->num_partitions,
                                               state->num_shards);
    if (!build_cache_instance_path(state->cache_root, fingerprint, path, sizeof(path))) return 0;
    if (state->cache_instance_dir == 0 || std::strcmp(state->cache_instance_dir, path) != 0) {
        if (!assign_owned_string(&state->cache_instance_dir, path)) return 0;
    }
    if (!ensure_directory_exists(state->cache_instance_dir)) return 0;
    if (!build_cache_metadata_dir_path(state->cache_instance_dir, metadata_dir, sizeof(metadata_dir))) return 0;
    if (!ensure_directory_exists(metadata_dir)) return 0;
    if (!build_cache_pack_root_path(state->cache_instance_dir, pack_root_dir, sizeof(pack_root_dir))) return 0;
    if (!ensure_directory_exists(pack_root_dir)) return 0;
    if (!build_cache_canonical_pack_dir_path(state->cache_instance_dir, canonical_pack_dir, sizeof(canonical_pack_dir))) return 0;
    if (!ensure_directory_exists(canonical_pack_dir)) return 0;
    if (!build_cache_execution_pack_dir_path(state->cache_instance_dir, execution_pack_dir, sizeof(execution_pack_dir))) return 0;
    if (!ensure_directory_exists(execution_pack_dir)) return 0;
    if (!build_cache_manifest_path(state->cache_instance_dir, path, sizeof(path))) return 0;
    if (!assign_owned_string(&state->cache_manifest_path, path)) return 0;
    if (!ensure_cache_tracking_allocated(state)) return 0;
    for (shard_id = 0ul; shard_id < (unsigned long) state->num_shards; ++shard_id) {
        if (state->shard_cache_paths[shard_id] == 0) {
            if (!build_shard_pack_path(state, shard_id, path, sizeof(path))) return 0;
            if (!assign_owned_string(state->shard_cache_paths + shard_id, path)) return 0;
        }
        if (::access(state->shard_cache_paths[shard_id], R_OK) == 0) {
            if (state->shard_cache_state[shard_id] != dataset_cache_shard_ready) {
                state->shard_cache_state[shard_id] = dataset_cache_shard_ready;
                state->shard_cache_bytes[shard_id] = estimate_shard_pack_bytes(state, shard_id);
                state->cache_resident_bytes += state->shard_cache_bytes[shard_id];
            }
        }
    }
    if (!write_dataset_cache_manifest(s->source_path, state)) return 0;
    if (!state->cache_budget_explicit) {
        std::uint64_t free_half = 0u;
        std::uint64_t estimated_total = 0u;
        if (::statvfs(state->cache_root, &vfs) == 0) {
            free_half = ((std::uint64_t) vfs.f_bavail * (std::uint64_t) vfs.f_frsize) / 2u;
        }
        for (shard_id = 0ul; shard_id < (unsigned long) state->num_shards; ++shard_id) {
            estimated_total += estimate_shard_pack_bytes(state, shard_id);
        }
        state->cache_budget_bytes = free_half == 0u ? estimated_total : (estimated_total < free_half ? estimated_total : free_half);
    }
    return 1;
}

inline int build_shard_partition_spans(dataset_h5_state *state) {
    std::uint64_t shard_id = 0u;
    std::uint64_t partition_id = 0u;
    if (state == 0) return 0;
    if (state->num_shards == 0u) return 1;
    if (state->shard_part_begin == 0) state->shard_part_begin = (std::uint64_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint64_t));
    if (state->shard_part_end == 0) state->shard_part_end = (std::uint64_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint64_t));
    if (state->partition_shard_ids == 0 && state->num_partitions != 0u) state->partition_shard_ids = (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
    if (state->shard_part_begin == 0 || state->shard_part_end == 0 || (state->num_partitions != 0u && state->partition_shard_ids == 0)) return 0;
    for (shard_id = 0u; shard_id < state->num_shards; ++shard_id) {
        const std::uint64_t row_begin = state->shard_offsets[shard_id];
        const std::uint64_t row_end = state->shard_offsets[shard_id + 1u];
        while (partition_id < state->num_partitions && state->partition_row_offsets[partition_id] < row_begin) ++partition_id;
        state->shard_part_begin[shard_id] = partition_id;
        while (partition_id < state->num_partitions && state->partition_row_offsets[partition_id + 1u] <= row_end) {
            state->partition_shard_ids[partition_id] = shard_id;
            ++partition_id;
        }
        state->shard_part_end[shard_id] = partition_id;
    }
    return 1;
}

template<typename MatrixT>
inline int write_shard_pack_file(const char *filename,
                                 std::uint64_t cols,
                                 const std::uint64_t *partition_rows,
                                 const std::uint64_t *partition_nnz,
                                 const std::uint64_t *partition_aux,
                                 std::uint64_t partition_count,
                                 MatrixT *const *parts) {
    static const unsigned char magic[8] = { 'C', 'S', 'P', 'A', 'C', 'K', '0', '1' };
    std::uint64_t *partition_offsets = 0;
    std::uint64_t *part_sizes = 0;
    std::uint64_t *shard_offsets = 0;
    std::FILE *fp = 0;
    std::uint64_t rows = 0u;
    std::uint64_t nnz = 0u;
    std::uint64_t payload_offset = 0u;
    std::uint64_t i = 0u;
    int ok = 0;

    if (filename == 0 || ((partition_count != 0u) && (partition_rows == 0 || partition_nnz == 0 || parts == 0))) return 0;
    if (partition_count != 0u) {
        partition_offsets = (std::uint64_t *) std::calloc((std::size_t) partition_count, sizeof(std::uint64_t));
        part_sizes = (std::uint64_t *) std::calloc((std::size_t) partition_count, sizeof(std::uint64_t));
        if (partition_offsets == 0 || part_sizes == 0) goto done;
    }
    shard_offsets = (std::uint64_t *) std::calloc(2u, sizeof(std::uint64_t));
    if (shard_offsets == 0) goto done;
    if (!compute_shard_pack_locators<MatrixT>(partition_rows, partition_nnz, partition_aux, cols, partition_count, partition_offsets, part_sizes)) goto done;
    payload_offset = sharded_pack_payload_offset(partition_count, 1u, shard_pack_payload_alignment);
    for (i = 0u; i < partition_count; ++i) {
        rows += partition_rows[i];
        nnz += partition_nnz[i];
    }
    shard_offsets[0] = 0u;
    shard_offsets[1] = rows;

    fp = std::fopen(filename, "wb");
    if (fp == 0) goto done;
    std::setvbuf(fp, 0, _IOFBF, (std::size_t) 8u << 20u);
    if (!write_sharded_block(fp, magic, sizeof(magic), 1u)) goto done;
    {
        const unsigned char format = (unsigned char) disk_format_code<MatrixT>::value;
        const unsigned char reserved[7] = { 0, 0, 0, 0, 0, 0, 0 };
        const std::uint64_t num_partitions = partition_count;
        const std::uint64_t num_shards = 1u;
        if (!write_sharded_block(fp, &format, sizeof(format), 1u)) goto done;
        if (!write_sharded_block(fp, reserved, sizeof(reserved), 1u)) goto done;
        if (!write_sharded_block(fp, &rows, sizeof(rows), 1u)) goto done;
        if (!write_sharded_block(fp, &cols, sizeof(cols), 1u)) goto done;
        if (!write_sharded_block(fp, &nnz, sizeof(nnz), 1u)) goto done;
        if (!write_sharded_block(fp, &num_partitions, sizeof(num_partitions), 1u)) goto done;
        if (!write_sharded_block(fp, &num_shards, sizeof(num_shards), 1u)) goto done;
        if (!write_sharded_block(fp, &shard_pack_payload_alignment, sizeof(shard_pack_payload_alignment), 1u)) goto done;
        if (!write_sharded_block(fp, &payload_offset, sizeof(payload_offset), 1u)) goto done;
        if (!write_sharded_block(fp, partition_rows, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
        if (!write_sharded_block(fp, partition_nnz, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
        if (partition_aux != 0) {
            if (!write_sharded_block(fp, partition_aux, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
        } else {
            const std::uint64_t zero = 0u;
            for (i = 0u; i < partition_count; ++i) {
                if (!write_sharded_block(fp, &zero, sizeof(zero), 1u)) goto done;
            }
        }
        if (!write_sharded_block(fp, shard_offsets, sizeof(std::uint64_t), 2u)) goto done;
        if (!write_sharded_block(fp, partition_offsets, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
        if (!write_sharded_block(fp, part_sizes, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
    }
    if (std::fflush(fp) != 0) goto done;
    for (i = 0u; i < partition_count; ++i) {
        if (parts[i] == 0) goto done;
        if (std::fseek(fp, (long) partition_offsets[i], SEEK_SET) != 0) goto done;
        if (!::cellshard::store(fp, parts[i])) goto done;
    }
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    std::free(partition_offsets);
    std::free(part_sizes);
    std::free(shard_offsets);
    return ok;
}

inline int load_dataset_h5_state(hid_t file, dataset_h5_state *state) {
    hid_t payload = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    int ok = 0;
    std::uint64_t num_codecs = 0;

    payload = H5Gopen2(file, payload_standard_group, H5P_DEFAULT);
    codecs = H5Gopen2(file, codecs_group, H5P_DEFAULT);
    if (payload < 0 || codecs < 0) goto done;
    if (!read_attr_u64(file, "num_partitions", &state->num_partitions)) goto done;
    if (!read_attr_u64(file, "num_codecs", &num_codecs)) goto done;
    state->num_codecs = (std::uint32_t) num_codecs;
    if (state->num_partitions != 0) {
        state->partition_indptr_offsets = (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
        state->partition_nnz_offsets = (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
        state->partition_codec_ids = (std::uint32_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint32_t));
        if (state->partition_indptr_offsets == 0 || state->partition_nnz_offsets == 0 || state->partition_codec_ids == 0) goto done;
    }
    if (state->num_codecs != 0) {
        state->codecs = (dataset_codec_descriptor *) std::calloc((std::size_t) state->num_codecs, sizeof(dataset_codec_descriptor));
        if (state->codecs == 0) goto done;
    }
    if (!read_dataset_1d(payload, "partition_indptr_offsets", H5T_NATIVE_UINT64, state->partition_indptr_offsets)) goto done;
    if (!read_dataset_1d(payload, "partition_nnz_offsets", H5T_NATIVE_UINT64, state->partition_nnz_offsets)) goto done;
    if (!read_dataset_1d(H5Gopen2(file, matrix_group, H5P_DEFAULT), "partition_codec_ids", H5T_NATIVE_UINT32, state->partition_codec_ids)) goto done;
    if (state->num_codecs != 0) {
        if (!read_dataset_1d(codecs, "codec_id", H5T_NATIVE_UINT32, &state->codecs[0].codec_id)) goto done;
    }
    ok = 1;

done:
    if (payload >= 0) H5Gclose(payload);
    if (codecs >= 0) H5Gclose(codecs);
    return ok;
}

inline int load_codec_table(hid_t codecs, dataset_codec_descriptor *descs, std::uint32_t count) {
    std::uint32_t i = 0;
    std::uint32_t *codec_ids = 0;
    std::uint32_t *families = 0;
    std::uint32_t *value_codes = 0;
    std::uint32_t *scale_value_codes = 0;
    std::uint32_t *bits = 0;
    std::uint32_t *flags = 0;
    int ok = 0;

    if (count == 0) return 1;
    codec_ids = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    families = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    value_codes = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    scale_value_codes = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    bits = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    flags = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    if (codec_ids == 0 || families == 0 || value_codes == 0 || scale_value_codes == 0 || bits == 0 || flags == 0) goto done;
    if (!read_dataset_1d(codecs, "codec_id", H5T_NATIVE_UINT32, codec_ids)) goto done;
    if (!read_dataset_1d(codecs, "family", H5T_NATIVE_UINT32, families)) goto done;
    if (!read_dataset_1d(codecs, "value_code", H5T_NATIVE_UINT32, value_codes)) goto done;
    if (!read_dataset_1d(codecs, "scale_value_code", H5T_NATIVE_UINT32, scale_value_codes)) goto done;
    if (!read_dataset_1d(codecs, "bits", H5T_NATIVE_UINT32, bits)) goto done;
    if (!read_dataset_1d(codecs, "flags", H5T_NATIVE_UINT32, flags)) goto done;
    for (i = 0; i < count; ++i) {
        descs[i].codec_id = codec_ids[i];
        descs[i].family = families[i];
        descs[i].value_code = value_codes[i];
        descs[i].scale_value_code = scale_value_codes[i];
        descs[i].bits = bits[i];
        descs[i].flags = flags[i];
    }
    ok = 1;

done:
    std::free(codec_ids);
    std::free(families);
    std::free(value_codes);
    std::free(scale_value_codes);
    std::free(bits);
    std::free(flags);
    return ok;
}

inline const dataset_codec_descriptor *find_codec(const dataset_h5_state *state, std::uint32_t codec_id) {
    std::uint32_t i = 0;
    if (state == 0) return 0;
    for (i = 0; i < state->num_codecs; ++i) {
        if (state->codecs[i].codec_id == codec_id) return state->codecs + i;
    }
    return 0;
}

inline int ensure_standard_payload_open(dataset_h5_state *state) {
    if (state == 0 || state->file < 0) return 0;
    if (state->payload_standard >= 0
        && state->d_standard_indptr >= 0
        && state->d_standard_indices >= 0
        && state->d_standard_values >= 0) {
        return 1;
    }
    if (state->payload_standard < 0) {
        state->payload_standard = H5Gopen2(state->file, payload_standard_group, H5P_DEFAULT);
        if (state->payload_standard < 0) return 0;
    }
    if (state->d_standard_indptr < 0) {
        state->d_standard_indptr = H5Dopen2(state->payload_standard, "indptr", H5P_DEFAULT);
        if (state->d_standard_indptr < 0) return 0;
    }
    if (state->d_standard_indices < 0) {
        state->d_standard_indices = H5Dopen2(state->payload_standard, "indices", H5P_DEFAULT);
        if (state->d_standard_indices < 0) return 0;
    }
    if (state->d_standard_values < 0) {
        state->d_standard_values = H5Dopen2(state->payload_standard, "values", H5P_DEFAULT);
        if (state->d_standard_values < 0) return 0;
    }
    return 1;
}

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

inline int ensure_optimized_blocked_ell_payload_open(dataset_h5_state *state) {
    if (state == 0 || state->file < 0) return 0;
    if (state->payload_optimized_blocked_ell >= 0) return 1;
    state->payload_optimized_blocked_ell = H5Gopen2(state->file, payload_optimized_blocked_ell_group, H5P_DEFAULT);
    return state->payload_optimized_blocked_ell >= 0;
}

inline int ensure_optimized_sliced_ell_payload_open(dataset_h5_state *state) {
    if (state == 0 || state->file < 0) return 0;
    if (state->payload_optimized_sliced_ell >= 0) return 1;
    state->payload_optimized_sliced_ell = H5Gopen2(state->file, payload_optimized_sliced_ell_group, H5P_DEFAULT);
    return state->payload_optimized_sliced_ell >= 0;
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
    if (!deserialize_optimized_shard(blob.data(), blob.size(), &state->loaded_optimized_shard)) return 0;
    state->loaded_optimized_shard_id = shard_id;
    return 1;
}

inline int load_sliced_ell_partition_payload(dataset_h5_state *state,
                                             unsigned long partition_id,
                                             sparse::sliced_ell *part) {
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
    sparse::clear(part);
    sparse::init(part);
    ok = ::cellshard::load(fp, part);
    std::fclose(fp);
    return ok;
}

inline int load_optimized_sliced_ell_shard_payload(dataset_h5_state *state,
                                                   std::uint64_t shard_id) {
    std::vector<unsigned char> blob;
    char dataset_name[64];
    if (state == 0 || shard_id >= state->num_shards) return 0;
    if (state->loaded_optimized_sliced_shard_id == shard_id) return 1;
    if (!ensure_optimized_sliced_ell_payload_open(state)) return 0;
    if (!build_optimized_shard_dataset_name((unsigned long) shard_id, dataset_name, sizeof(dataset_name))) return 0;
    if (!read_blob_dataset(state->payload_optimized_sliced_ell, dataset_name, &blob)) return 0;
    clear(&state->loaded_optimized_sliced_shard);
    init(&state->loaded_optimized_sliced_shard);
    if (!deserialize_optimized_sliced_shard(blob.data(), blob.size(), &state->loaded_optimized_sliced_shard)) return 0;
    state->loaded_optimized_sliced_shard_id = shard_id;
    return 1;
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
    dst->segments = dst->segment_count != 0u ? (sparse::sliced_ell *) std::calloc((std::size_t) dst->segment_count, sizeof(sparse::sliced_ell)) : 0;
    dst->segment_row_offsets = (std::uint32_t *) std::calloc((std::size_t) dst->segment_count + 1u, sizeof(std::uint32_t));
    if ((dst->segment_count != 0u && (dst->segments == 0 || dst->segment_row_offsets == 0))
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
    std::size_t emitted = 0u;
    if (part == 0 || out == 0) return 0;
    sparse::clear(out);
    sparse::init(out, part->rows, part->cols, part->nnz);
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
                if (emitted >= out->nnz) {
                    sparse::clear(out);
                    return 0;
                }
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

inline int bucketed_partition_to_canonical_coo(const bucketed_blocked_ell_partition *part,
                                               const std::uint32_t *exec_to_canonical_cols,
                                               sparse::coo *out) {
    std::size_t emitted = 0u;
    if (part == 0 || out == 0) return 0;
    sparse::clear(out);
    sparse::init(out, part->rows, part->cols, part->nnz);
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
                    if (emitted >= out->nnz) {
                        sparse::clear(out);
                        return 0;
                    }
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
                     layout->row_block_order.end(),
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
                    out->exec_to_canonical_rows[exec_row] = src_rb * block_size + row_in_block;
                    out->canonical_to_exec_rows[src_rb * block_size + row_in_block] = exec_row;
                    ++exec_row;
                }
                ++dst_slot;
            }
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

inline int build_sliced_execution_bucket_layout(const sparse::sliced_ell *part,
                                                std::uint32_t requested_bucket_count,
                                                sliced_execution_bucket_layout *layout) {
    const std::uint32_t row_count = part != 0 ? part->rows : 0u;
    std::uint32_t bucket_count = 0u;
    std::uint32_t bucket = 0u;
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
    bucket_count = std::max<std::uint32_t>(1u, std::min<std::uint32_t>(requested_bucket_count, row_count));
    std::stable_sort(layout->row_order.begin(),
                     layout->row_order.end(),
                     [&](std::uint32_t lhs, std::uint32_t rhs) {
                         const std::uint32_t lhs_width = layout->row_widths[lhs];
                         const std::uint32_t rhs_width = layout->row_widths[rhs];
                         if (lhs_width != rhs_width) return lhs_width < rhs_width;
                         return lhs < rhs;
                     });
    layout->segment_row_offsets.reserve((std::size_t) bucket_count + 1u);
    layout->segment_widths.reserve(bucket_count);
    layout->segment_row_offsets.push_back(0u);
    for (bucket = 0u; bucket < bucket_count; ++bucket) {
        const std::uint32_t row_begin = (bucket * row_count) / bucket_count;
        const std::uint32_t row_end = ((bucket + 1u) * row_count) / bucket_count;
        std::uint32_t seg_width = 0u;
        for (std::uint32_t pos = row_begin; pos < row_end; ++pos) {
            seg_width = std::max(seg_width, layout->row_widths[layout->row_order[pos]]);
        }
        layout->segment_widths.push_back(seg_width);
        layout->segment_row_offsets.push_back(row_end);
    }
    if (layout->segment_widths.size() > 1u) {
        bool all_same = true;
        for (std::size_t i = 1; i < layout->segment_widths.size(); ++i) {
            if (layout->segment_widths[i] != layout->segment_widths[0]) {
                all_same = false;
                break;
            }
        }
        if (all_same) {
            layout->segment_row_offsets.assign(2u, 0u);
            layout->segment_row_offsets[1] = row_count;
            layout->segment_widths.assign(1u, layout->segment_widths[0]);
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
    out->segments = out->segment_count != 0u ? (sparse::sliced_ell *) std::calloc((std::size_t) out->segment_count, sizeof(sparse::sliced_ell)) : 0;
    out->segment_row_offsets = (std::uint32_t *) std::calloc((std::size_t) out->segment_count + 1u, sizeof(std::uint32_t));
    out->exec_to_canonical_rows = out->rows != 0u ? (std::uint32_t *) std::calloc((std::size_t) out->rows, sizeof(std::uint32_t)) : 0;
    out->canonical_to_exec_rows = out->rows != 0u ? (std::uint32_t *) std::calloc((std::size_t) out->rows, sizeof(std::uint32_t)) : 0;
    if ((out->segment_count != 0u && (out->segments == 0 || out->segment_row_offsets == 0))
        || (out->rows != 0u && (out->exec_to_canonical_rows == 0 || out->canonical_to_exec_rows == 0))) {
        clear(out);
        return 0;
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
    return write_execution_partition_blob(fp, part);
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

inline int load_optimized_partition_blob(std::FILE *fp, bucketed_blocked_ell_partition *part) {
    return load_execution_partition_blob(fp, part);
}

inline int serialize_optimized_shard(const bucketed_blocked_ell_shard *shard,
                                     unsigned char **data_out,
                                     std::size_t *bytes_out) {
    char *buffer = 0;
    std::size_t bytes = 0u;
    std::FILE *fp = 0;
    std::uint32_t partition = 0u;
    if (shard == 0 || data_out == 0 || bytes_out == 0) return 0;
    *data_out = 0;
    *bytes_out = 0u;
    fp = open_memstream(&buffer, &bytes);
    if (fp == 0) return 0;
    if (!write_sharded_block(fp, &shard->rows, sizeof(shard->rows), 1u)
        || !write_sharded_block(fp, &shard->cols, sizeof(shard->cols), 1u)
        || !write_sharded_block(fp, &shard->nnz, sizeof(shard->nnz), 1u)
        || !write_sharded_block(fp, &shard->partition_count, sizeof(shard->partition_count), 1u)
        || !write_sharded_block(fp, shard->partition_row_offsets, sizeof(std::uint32_t), (std::size_t) shard->partition_count + 1u)
        || !write_sharded_block(fp, shard->exec_to_canonical_cols, sizeof(std::uint32_t), shard->cols)
        || !write_sharded_block(fp, shard->canonical_to_exec_cols, sizeof(std::uint32_t), shard->cols)) {
        std::fclose(fp);
        std::free(buffer);
        return 0;
    }
    for (partition = 0u; partition < shard->partition_count; ++partition) {
        if (!write_optimized_partition_blob(fp, shard->partitions + partition)) {
            std::fclose(fp);
            std::free(buffer);
            return 0;
        }
    }
    if (std::fclose(fp) != 0) {
        std::free(buffer);
        return 0;
    }
    *data_out = (unsigned char *) buffer;
    *bytes_out = bytes;
    return 1;
}

inline int deserialize_optimized_shard(const unsigned char *data,
                                       std::size_t bytes,
                                       bucketed_blocked_ell_shard *shard) {
    std::FILE *fp = 0;
    std::uint32_t partition = 0u;
    if (data == 0 || shard == 0) return 0;
    clear(shard);
    init(shard);
    fp = fmemopen((void *) data, bytes, "rb");
    if (fp == 0) return 0;
    if (!read_sharded_block(fp, &shard->rows, sizeof(shard->rows), 1u)
        || !read_sharded_block(fp, &shard->cols, sizeof(shard->cols), 1u)
        || !read_sharded_block(fp, &shard->nnz, sizeof(shard->nnz), 1u)
        || !read_sharded_block(fp, &shard->partition_count, sizeof(shard->partition_count), 1u)) {
        std::fclose(fp);
        clear(shard);
        return 0;
    }
    shard->partition_row_offsets = (std::uint32_t *) std::calloc((std::size_t) shard->partition_count + 1u, sizeof(std::uint32_t));
    shard->partitions = shard->partition_count != 0u
        ? (bucketed_blocked_ell_partition *) std::calloc((std::size_t) shard->partition_count, sizeof(bucketed_blocked_ell_partition))
        : 0;
    shard->exec_to_canonical_cols = shard->cols != 0u ? (std::uint32_t *) std::calloc((std::size_t) shard->cols, sizeof(std::uint32_t)) : 0;
    shard->canonical_to_exec_cols = shard->cols != 0u ? (std::uint32_t *) std::calloc((std::size_t) shard->cols, sizeof(std::uint32_t)) : 0;
    if ((shard->partition_count != 0u && (shard->partition_row_offsets == 0 || shard->partitions == 0))
        || (shard->cols != 0u && (shard->exec_to_canonical_cols == 0 || shard->canonical_to_exec_cols == 0))) {
        std::fclose(fp);
        clear(shard);
        return 0;
    }
    for (partition = 0u; partition < shard->partition_count; ++partition) init(shard->partitions + partition);
    if (!read_sharded_block(fp, shard->partition_row_offsets, sizeof(std::uint32_t), (std::size_t) shard->partition_count + 1u)
        || !read_sharded_block(fp, shard->exec_to_canonical_cols, sizeof(std::uint32_t), shard->cols)
        || !read_sharded_block(fp, shard->canonical_to_exec_cols, sizeof(std::uint32_t), shard->cols)) {
        std::fclose(fp);
        clear(shard);
        return 0;
    }
    for (partition = 0u; partition < shard->partition_count; ++partition) {
        if (!load_optimized_partition_blob(fp, shard->partitions + partition)
            || !assign_partition_col_maps(shard->partitions + partition,
                                          shard->exec_to_canonical_cols,
                                          shard->canonical_to_exec_cols,
                                          shard->cols)) {
            std::fclose(fp);
            clear(shard);
            return 0;
        }
    }
    std::fclose(fp);
    return 1;
}

inline int write_sliced_execution_partition_blob(std::FILE *fp, const bucketed_sliced_ell_partition *part) {
    std::uint32_t segment = 0u;
    if (fp == 0 || part == 0) return 0;
    if (!write_sharded_block(fp, &part->rows, sizeof(part->rows), 1u)) return 0;
    if (!write_sharded_block(fp, &part->cols, sizeof(part->cols), 1u)) return 0;
    if (!write_sharded_block(fp, &part->nnz, sizeof(part->nnz), 1u)) return 0;
    if (!write_sharded_block(fp, &part->segment_count, sizeof(part->segment_count), 1u)) return 0;
    if (!write_sharded_block(fp, part->segment_row_offsets, sizeof(std::uint32_t), (std::size_t) part->segment_count + 1u)) return 0;
    if (!write_sharded_block(fp, part->exec_to_canonical_rows, sizeof(std::uint32_t), part->rows)) return 0;
    if (!write_sharded_block(fp, part->canonical_to_exec_rows, sizeof(std::uint32_t), part->rows)) return 0;
    for (segment = 0u; segment < part->segment_count; ++segment) {
        if (!::cellshard::store(fp, part->segments + segment)) return 0;
    }
    return 1;
}

inline int load_sliced_execution_partition_blob(std::FILE *fp, bucketed_sliced_ell_partition *part) {
    std::uint32_t segment = 0u;
    if (fp == 0 || part == 0) return 0;
    clear(part);
    if (!read_sharded_block(fp, &part->rows, sizeof(part->rows), 1u)) return 0;
    if (!read_sharded_block(fp, &part->cols, sizeof(part->cols), 1u)) return 0;
    if (!read_sharded_block(fp, &part->nnz, sizeof(part->nnz), 1u)) return 0;
    if (!read_sharded_block(fp, &part->segment_count, sizeof(part->segment_count), 1u)) return 0;
    part->segments = part->segment_count != 0u ? (sparse::sliced_ell *) std::calloc((std::size_t) part->segment_count, sizeof(sparse::sliced_ell)) : 0;
    part->segment_row_offsets = (std::uint32_t *) std::calloc((std::size_t) part->segment_count + 1u, sizeof(std::uint32_t));
    part->exec_to_canonical_rows = part->rows != 0u ? (std::uint32_t *) std::calloc((std::size_t) part->rows, sizeof(std::uint32_t)) : 0;
    part->canonical_to_exec_rows = part->rows != 0u ? (std::uint32_t *) std::calloc((std::size_t) part->rows, sizeof(std::uint32_t)) : 0;
    if ((part->segment_count != 0u && (part->segments == 0 || part->segment_row_offsets == 0))
        || (part->rows != 0u && (part->exec_to_canonical_rows == 0 || part->canonical_to_exec_rows == 0))) {
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
    for (segment = 0u; segment < part->segment_count; ++segment) {
        if (!::cellshard::load(fp, part->segments + segment)) {
            clear(part);
            return 0;
        }
    }
    return 1;
}

inline int serialize_optimized_sliced_shard(const bucketed_sliced_ell_shard *shard,
                                            unsigned char **data_out,
                                            std::size_t *bytes_out) {
    char *buffer = 0;
    std::size_t bytes = 0u;
    std::FILE *fp = 0;
    std::uint32_t partition = 0u;
    if (shard == 0 || data_out == 0 || bytes_out == 0) return 0;
    *data_out = 0;
    *bytes_out = 0u;
    fp = open_memstream(&buffer, &bytes);
    if (fp == 0) return 0;
    if (!write_sharded_block(fp, &shard->rows, sizeof(shard->rows), 1u)
        || !write_sharded_block(fp, &shard->cols, sizeof(shard->cols), 1u)
        || !write_sharded_block(fp, &shard->nnz, sizeof(shard->nnz), 1u)
        || !write_sharded_block(fp, &shard->partition_count, sizeof(shard->partition_count), 1u)
        || !write_sharded_block(fp, shard->partition_row_offsets, sizeof(std::uint32_t), (std::size_t) shard->partition_count + 1u)) {
        std::fclose(fp);
        std::free(buffer);
        return 0;
    }
    for (partition = 0u; partition < shard->partition_count; ++partition) {
        if (!write_sliced_execution_partition_blob(fp, shard->partitions + partition)) {
            std::fclose(fp);
            std::free(buffer);
            return 0;
        }
    }
    if (std::fclose(fp) != 0) {
        std::free(buffer);
        return 0;
    }
    *data_out = (unsigned char *) buffer;
    *bytes_out = bytes;
    return 1;
}

inline int deserialize_optimized_sliced_shard(const unsigned char *data,
                                              std::size_t bytes,
                                              bucketed_sliced_ell_shard *shard) {
    std::FILE *fp = 0;
    std::uint32_t partition = 0u;
    if (data == 0 || shard == 0) return 0;
    clear(shard);
    init(shard);
    fp = fmemopen((void *) data, bytes, "rb");
    if (fp == 0) return 0;
    if (!read_sharded_block(fp, &shard->rows, sizeof(shard->rows), 1u)
        || !read_sharded_block(fp, &shard->cols, sizeof(shard->cols), 1u)
        || !read_sharded_block(fp, &shard->nnz, sizeof(shard->nnz), 1u)
        || !read_sharded_block(fp, &shard->partition_count, sizeof(shard->partition_count), 1u)) {
        std::fclose(fp);
        clear(shard);
        return 0;
    }
    shard->partition_row_offsets = (std::uint32_t *) std::calloc((std::size_t) shard->partition_count + 1u, sizeof(std::uint32_t));
    shard->partitions = shard->partition_count != 0u
        ? (bucketed_sliced_ell_partition *) std::calloc((std::size_t) shard->partition_count, sizeof(bucketed_sliced_ell_partition))
        : 0;
    if (shard->partition_count != 0u && (shard->partition_row_offsets == 0 || shard->partitions == 0)) {
        std::fclose(fp);
        clear(shard);
        return 0;
    }
    for (partition = 0u; partition < shard->partition_count; ++partition) init(shard->partitions + partition);
    if (!read_sharded_block(fp, shard->partition_row_offsets, sizeof(std::uint32_t), (std::size_t) shard->partition_count + 1u)) {
        std::fclose(fp);
        clear(shard);
        return 0;
    }
    for (partition = 0u; partition < shard->partition_count; ++partition) {
        if (!load_sliced_execution_partition_blob(fp, shard->partitions + partition)) {
            std::fclose(fp);
            clear(shard);
            return 0;
        }
    }
    std::fclose(fp);
    return 1;
}

inline void close_cached_shard_file(dataset_h5_state *state, unsigned long shard_id) {
    dataset_h5_cache_runtime *runtime = cache_runtime(state);
    if (state == 0 || runtime == 0 || shard_id >= state->num_shards || state->shard_cache_files == 0) return;
    std::lock_guard<std::mutex> file_lock(runtime->shard_file_mutexes[shard_id]);
    if (state->shard_cache_files[shard_id] != 0) {
        std::fclose(state->shard_cache_files[shard_id]);
        state->shard_cache_files[shard_id] = 0;
    }
}

inline int ensure_cached_shard_file_open(dataset_h5_state *state, unsigned long shard_id) {
    dataset_h5_cache_runtime *runtime = cache_runtime(state);
    if (state == 0 || runtime == 0 || shard_id >= state->num_shards || state->shard_cache_paths == 0) return 0;
    std::lock_guard<std::mutex> file_lock(runtime->shard_file_mutexes[shard_id]);
    if (state->shard_cache_files[shard_id] != 0) return 1;
    if (state->shard_cache_paths[shard_id] == 0) return 0;
    state->shard_cache_files[shard_id] = std::fopen(state->shard_cache_paths[shard_id], "rb");
    if (state->shard_cache_files[shard_id] == 0) return 0;
    std::setvbuf(state->shard_cache_files[shard_id], 0, _IOFBF, (std::size_t) 8u << 20u);
    return 1;
}

template<typename MatrixT>
inline void compute_cached_part_locator(const dataset_h5_state *state,
                                        unsigned long partition_id,
                                        std::uint64_t *offset,
                                        std::uint64_t *bytes) {
    const std::uint64_t shard_id = state != 0 && state->partition_shard_ids != 0 ? state->partition_shard_ids[partition_id] : 0u;
    const std::uint64_t begin = state != 0 && state->shard_part_begin != 0 ? state->shard_part_begin[shard_id] : 0u;
    const std::uint64_t end = state != 0 && state->shard_part_end != 0 ? state->shard_part_end[shard_id] : 0u;
    std::uint64_t cursor = sharded_pack_payload_offset(end - begin, 1u, shard_pack_payload_alignment);
    std::uint64_t i = 0u;

    if (offset != 0) *offset = cursor;
    if (bytes != 0) *bytes = 0u;
    if (state == 0 || partition_id >= state->num_partitions || begin > partition_id || end <= partition_id) return;
    for (i = begin; i < partition_id; ++i) {
        const std::size_t partition_bytes = packed_bytes((const MatrixT *) 0,
                                                    (types::dim_t) state->partition_rows[i],
                                                    (types::dim_t) state->cols,
                                                    (types::nnz_t) state->partition_nnz[i],
                                                    (unsigned long) state->partition_aux[i],
                                                    sizeof(real::storage_t));
        cursor += (std::uint64_t) partition_bytes;
        cursor = (cursor + shard_pack_payload_alignment - 1u) & ~(shard_pack_payload_alignment - 1u);
    }
    if (offset != 0) *offset = cursor;
    if (bytes != 0) {
        *bytes = (std::uint64_t) packed_bytes((const MatrixT *) 0,
                                              (types::dim_t) state->partition_rows[partition_id],
                                              (types::dim_t) state->cols,
                                              (types::nnz_t) state->partition_nnz[partition_id],
                                              (unsigned long) state->partition_aux[partition_id],
                                              sizeof(real::storage_t));
    }
}

inline int load_compressed_part_from_cached_pack(sharded<sparse::compressed> *m,
                                                 dataset_h5_state *state,
                                                 unsigned long partition_id) {
    const unsigned long shard_id = state != 0 && state->partition_shard_ids != 0 ? (unsigned long) state->partition_shard_ids[partition_id] : 0ul;
    dataset_h5_cache_runtime *runtime = cache_runtime(state);
    sparse::compressed *part = 0;
    std::uint64_t offset = 0u;
    int ok = 0;

    if (m == 0 || state == 0 || runtime == 0 || partition_id >= m->num_partitions) return 0;
    if (!ensure_cached_shard_file_open(state, shard_id)) return 0;
    compute_cached_part_locator<sparse::compressed>(state, partition_id, &offset, 0);
    std::lock_guard<std::mutex> file_lock(runtime->shard_file_mutexes[shard_id]);
    if (state->shard_cache_files[shard_id] == 0) return 0;
    part = new sparse::compressed;
    sparse::init(part);
    if (fseeko(state->shard_cache_files[shard_id], (off_t) offset, SEEK_SET) != 0) goto done;
    if (!::cellshard::load(state->shard_cache_files[shard_id], part)) goto done;
    if (part->rows != m->partition_rows[partition_id]) goto done;
    if (part->cols != m->cols) goto done;
    if (part->nnz != m->partition_nnz[partition_id]) goto done;
    if ((unsigned long) part->axis != m->partition_aux[partition_id]) goto done;
    if (m->parts[partition_id] != 0) destroy(m->parts[partition_id]);
    m->parts[partition_id] = part;
    part = 0;
    ok = 1;

done:
    if (part != 0) {
        sparse::clear(part);
        delete part;
    }
    return ok;
}

inline int load_blocked_ell_part_from_cached_pack(sharded<sparse::blocked_ell> *m,
                                                  dataset_h5_state *state,
                                                  unsigned long partition_id) {
    const unsigned long shard_id = state != 0 && state->partition_shard_ids != 0 ? (unsigned long) state->partition_shard_ids[partition_id] : 0ul;
    dataset_h5_cache_runtime *runtime = cache_runtime(state);
    sparse::blocked_ell *part = 0;
    std::uint64_t offset = 0u;
    int ok = 0;

    if (m == 0 || state == 0 || runtime == 0 || partition_id >= m->num_partitions) return 0;
    if (!ensure_cached_shard_file_open(state, shard_id)) return 0;
    compute_cached_part_locator<sparse::blocked_ell>(state, partition_id, &offset, 0);
    std::lock_guard<std::mutex> file_lock(runtime->shard_file_mutexes[shard_id]);
    if (state->shard_cache_files[shard_id] == 0) return 0;
    part = new sparse::blocked_ell;
    sparse::init(part);
    if (fseeko(state->shard_cache_files[shard_id], (off_t) offset, SEEK_SET) != 0) goto done;
    if (!::cellshard::load(state->shard_cache_files[shard_id], part)) goto done;
    if (part->rows != m->partition_rows[partition_id]) goto done;
    if (part->cols != m->cols) goto done;
    if (part->nnz != m->partition_nnz[partition_id]) goto done;
    if (::cellshard::partition_aux(part) != m->partition_aux[partition_id]) goto done;
    if (m->parts[partition_id] != 0) destroy(m->parts[partition_id]);
    m->parts[partition_id] = part;
    part = 0;
    ok = 1;

done:
    if (part != 0) {
        sparse::clear(part);
        delete part;
    }
    return ok;
}

inline int load_sliced_ell_part_from_cached_pack(sharded<sparse::sliced_ell> *m,
                                                 dataset_h5_state *state,
                                                 unsigned long partition_id) {
    const unsigned long shard_id = state != 0 && state->partition_shard_ids != 0 ? (unsigned long) state->partition_shard_ids[partition_id] : 0ul;
    dataset_h5_cache_runtime *runtime = cache_runtime(state);
    sparse::sliced_ell *part = 0;
    std::uint64_t offset = 0u;
    int ok = 0;

    if (m == 0 || state == 0 || runtime == 0 || partition_id >= m->num_partitions) return 0;
    if (!ensure_cached_shard_file_open(state, shard_id)) return 0;
    compute_cached_part_locator<sparse::sliced_ell>(state, partition_id, &offset, 0);
    std::lock_guard<std::mutex> file_lock(runtime->shard_file_mutexes[shard_id]);
    if (state->shard_cache_files[shard_id] == 0) return 0;
    part = new sparse::sliced_ell;
    sparse::init(part);
    if (fseeko(state->shard_cache_files[shard_id], (off_t) offset, SEEK_SET) != 0) goto done;
    if (!::cellshard::load(state->shard_cache_files[shard_id], part)) goto done;
    if (part->rows != m->partition_rows[partition_id]) goto done;
    if (part->cols != m->cols) goto done;
    if (part->nnz != m->partition_nnz[partition_id]) goto done;
    if (::cellshard::partition_aux(part) != m->partition_aux[partition_id]) goto done;
    if (m->parts[partition_id] != 0) destroy(m->parts[partition_id]);
    m->parts[partition_id] = part;
    part = 0;
    ok = 1;

done:
    if (part != 0) {
        sparse::clear(part);
        delete part;
    }
    return ok;
}

inline int load_execution_partition_from_pack(const dataset_h5_state *state,
                                              unsigned long shard_id,
                                              unsigned long partition_id,
                                              bucketed_blocked_ell_partition *out) {
    char path[4096];
    std::FILE *fp = 0;
    unsigned char magic[8];
    std::uint64_t file_shard_id = 0u;
    std::uint64_t partition_count = 0u;
    std::uint64_t *partition_offsets = 0;
    std::uint64_t local_partition_id = 0u;
    int ok = 0;

    if (state == 0 || out == 0 || shard_id >= state->num_shards || partition_id >= state->num_partitions) return 0;
    if (!build_execution_pack_path(state, shard_id, path, sizeof(path))) return 0;
    fp = std::fopen(path, "rb");
    if (fp == 0) return 0;
    if (!read_sharded_block(fp, magic, sizeof(magic), 1u)) goto done;
    if (std::memcmp(magic, execution_pack_magic, sizeof(magic)) != 0) goto done;
    if (!read_sharded_block(fp, &file_shard_id, sizeof(file_shard_id), 1u)) goto done;
    if (!read_sharded_block(fp, &partition_count, sizeof(partition_count), 1u)) goto done;
    if (file_shard_id != shard_id) goto done;
    local_partition_id = partition_id - state->shard_part_begin[shard_id];
    if (local_partition_id >= partition_count) goto done;
    partition_offsets = (std::uint64_t *) std::calloc((std::size_t) partition_count, sizeof(std::uint64_t));
    if (partition_count != 0u && partition_offsets == 0) goto done;
    if (!read_sharded_block(fp, partition_offsets, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
    if (fseeko(fp, (off_t) partition_offsets[local_partition_id], SEEK_SET) != 0) goto done;
    if (!load_execution_partition_blob(fp, out)) goto done;
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    std::free(partition_offsets);
    return ok;
}

inline int load_sliced_execution_partition_from_pack(const dataset_h5_state *state,
                                                     unsigned long shard_id,
                                                     unsigned long partition_id,
                                                     bucketed_sliced_ell_partition *out) {
    char path[4096];
    std::FILE *fp = 0;
    unsigned char magic[8];
    std::uint64_t file_shard_id = 0u;
    std::uint64_t partition_count = 0u;
    std::uint64_t *partition_offsets = 0;
    std::uint64_t local_partition_id = 0u;
    int ok = 0;

    if (state == 0 || out == 0 || shard_id >= state->num_shards || partition_id >= state->num_partitions) return 0;
    if (!build_execution_pack_path(state, shard_id, path, sizeof(path))) return 0;
    fp = std::fopen(path, "rb");
    if (fp == 0) return 0;
    if (!read_sharded_block(fp, magic, sizeof(magic), 1u)) goto done;
    if (std::memcmp(magic, execution_pack_magic, sizeof(magic)) != 0) goto done;
    if (!read_sharded_block(fp, &file_shard_id, sizeof(file_shard_id), 1u)) goto done;
    if (!read_sharded_block(fp, &partition_count, sizeof(partition_count), 1u)) goto done;
    if (file_shard_id != shard_id) goto done;
    local_partition_id = partition_id - state->shard_part_begin[shard_id];
    if (local_partition_id >= partition_count) goto done;
    partition_offsets = (std::uint64_t *) std::calloc((std::size_t) partition_count, sizeof(std::uint64_t));
    if (partition_count != 0u && partition_offsets == 0) goto done;
    if (!read_sharded_block(fp, partition_offsets, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
    if (fseeko(fp, (off_t) partition_offsets[local_partition_id], SEEK_SET) != 0) goto done;
    if (!load_sliced_execution_partition_blob(fp, out)) goto done;
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    std::free(partition_offsets);
    return ok;
}

inline int materialize_blocked_ell_execution_pack(shard_storage *s, dataset_h5_state *state, unsigned long shard_id) {
    const std::uint64_t begin = state != 0 && state->shard_part_begin != 0 ? state->shard_part_begin[shard_id] : 0u;
    const std::uint64_t end = state != 0 && state->shard_part_end != 0 ? state->shard_part_end[shard_id] : 0u;
    const std::uint64_t partition_count = end >= begin ? (end - begin) : 0u;
    sparse::blocked_ell **parts = 0;
    bucketed_blocked_ell_partition *exec_parts = 0;
    const bucketed_blocked_ell_shard *optimized_shard = 0;
    std::uint64_t *partition_offsets = 0;
    char tmp_path[4096];
    char final_path[4096];
    std::FILE *fp = 0;
    std::uint64_t local = 0u;
    std::uint64_t pack_bytes = 0u;
    int ok = 0;

    if (s == 0 || state == 0 || shard_id >= state->num_shards) return 0;
    if (!open_dataset_h5_backend(s)) return 0;
    if (partition_count != 0u) {
        partition_offsets = (std::uint64_t *) std::calloc((std::size_t) partition_count, sizeof(std::uint64_t));
        if (partition_offsets == 0) goto done;
    }
    if (state->matrix_family == dataset_matrix_family_optimized_blocked_ell) {
        if (!load_optimized_blocked_ell_shard_payload(state, shard_id)) goto done;
        optimized_shard = &state->loaded_optimized_shard;
        if (optimized_shard->partition_count != partition_count) goto done;
    } else {
        if (!load_blocked_ell_shard_payload(state, shard_id)) goto done;
        if (partition_count != 0u) {
            parts = (sparse::blocked_ell **) std::calloc((std::size_t) partition_count, sizeof(sparse::blocked_ell *));
            exec_parts = (bucketed_blocked_ell_partition *) std::calloc((std::size_t) partition_count, sizeof(bucketed_blocked_ell_partition));
            if (parts == 0 || exec_parts == 0) goto done;
            for (local = 0u; local < partition_count; ++local) init(exec_parts + local);
        }
        if (!prepare_blocked_ell_parts_from_state(state, (unsigned long) begin, (unsigned long) end, parts)) goto done;
        if (!fill_blocked_ell_parts_from_loaded_shard(state, shard_id, (unsigned long) begin, (unsigned long) end, parts)) goto done;
        for (local = 0u; local < partition_count; ++local) {
            const std::uint64_t partition_id = begin + local;
            const std::uint32_t requested_bucket_count =
                state->partition_execution_formats != 0
                && state->partition_execution_formats[partition_id] == dataset_execution_format_bucketed_blocked_ell
                && state->partition_blocked_ell_bucket_counts != 0
                    ? std::max<std::uint32_t>(1u, state->partition_blocked_ell_bucket_counts[partition_id])
                    : 1u;
            std::uint64_t bucketed_bytes = 0u;
            if (!build_bucketed_execution_partition(exec_parts + local, parts[local], requested_bucket_count, &bucketed_bytes)) goto done;
            if (state->partition_bucketed_blocked_ell_bytes != 0) state->partition_bucketed_blocked_ell_bytes[partition_id] = bucketed_bytes;
        }
    }
    if (!build_execution_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) goto done;
    if (!build_execution_pack_path(state, shard_id, final_path, sizeof(final_path))) goto done;
    fp = std::fopen(tmp_path, "wb");
    if (fp == 0) goto done;
    std::setvbuf(fp, 0, _IOFBF, (std::size_t) 8u << 20u);
    if (!write_sharded_block(fp, execution_pack_magic, sizeof(execution_pack_magic), 1u)) goto done;
    if (!write_sharded_block(fp, &shard_id, sizeof(shard_id), 1u)) goto done;
    if (!write_sharded_block(fp, &partition_count, sizeof(partition_count), 1u)) goto done;
    for (local = 0u; local < partition_count; ++local) {
        const std::uint64_t zero = 0u;
        if (!write_sharded_block(fp, &zero, sizeof(zero), 1u)) goto done;
    }
    for (local = 0u; local < partition_count; ++local) {
        partition_offsets[local] = (std::uint64_t) ftello(fp);
        if (optimized_shard != 0) {
            if (!write_execution_partition_blob(fp, optimized_shard->partitions + local)) goto done;
        } else {
            if (!write_execution_partition_blob(fp, exec_parts + local)) goto done;
        }
    }
    if (fseeko(fp, (off_t) (sizeof(execution_pack_magic) + sizeof(std::uint64_t) * 2u), SEEK_SET) != 0) goto done;
    if (!write_sharded_block(fp, partition_offsets, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
    if (std::fflush(fp) != 0) goto done;
    std::fclose(fp);
    fp = 0;
    if (::rename(tmp_path, final_path) != 0) {
        std::remove(tmp_path);
        goto done;
    }
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    if (!ok && build_execution_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) std::remove(tmp_path);
    if (exec_parts != 0) {
        for (local = 0u; local < partition_count; ++local) clear(exec_parts + local);
    }
    clear_blocked_ell_parts(parts, (unsigned long) partition_count);
    std::free(partition_offsets);
    std::free(exec_parts);
    std::free(parts);
    return ok;
}

inline int materialize_sliced_ell_execution_pack(shard_storage *s, dataset_h5_state *state, unsigned long shard_id) {
    const std::uint64_t begin = state != 0 && state->shard_part_begin != 0 ? state->shard_part_begin[shard_id] : 0u;
    const std::uint64_t end = state != 0 && state->shard_part_end != 0 ? state->shard_part_end[shard_id] : 0u;
    const std::uint64_t partition_count = end >= begin ? (end - begin) : 0u;
    bucketed_sliced_ell_partition *exec_parts = 0;
    const bucketed_sliced_ell_shard *optimized_shard = 0;
    std::uint64_t *partition_offsets = 0;
    char tmp_path[4096];
    char final_path[4096];
    std::FILE *fp = 0;
    std::uint64_t local = 0u;
    std::uint64_t pack_bytes = 0u;
    int ok = 0;

    if (s == 0 || state == 0 || shard_id >= state->num_shards) return 0;
    if (!open_dataset_h5_backend(s)) return 0;
    if (partition_count != 0u) {
        partition_offsets = (std::uint64_t *) std::calloc((std::size_t) partition_count, sizeof(std::uint64_t));
        exec_parts = (bucketed_sliced_ell_partition *) std::calloc((std::size_t) partition_count, sizeof(bucketed_sliced_ell_partition));
        if (partition_offsets == 0 || exec_parts == 0) goto done;
        for (local = 0u; local < partition_count; ++local) init(exec_parts + local);
    }
    if (!ensure_execution_metadata_allocated(state)) goto done;
    if (!ensure_optimized_sliced_ell_payload_open(state) || !load_optimized_sliced_ell_shard_payload(state, shard_id)) {
        std::uint32_t shard_segment_total = 0u;
        for (local = 0u; local < partition_count; ++local) {
            const std::uint64_t partition_id = begin + local;
            const std::uint32_t requested_bucket_count =
                state->partition_execution_formats != 0
                && state->partition_execution_formats[partition_id] == dataset_execution_format_bucketed_sliced_ell
                && state->partition_blocked_ell_bucket_counts != 0
                    ? std::max<std::uint32_t>(1u, state->partition_blocked_ell_bucket_counts[partition_id])
                    : 1u;
            std::uint64_t bucketed_bytes = 0u;
            sparse::sliced_ell part;
            sparse::init(&part);
            if (!load_sliced_ell_partition_payload(state, (unsigned long) partition_id, &part)) {
                sparse::clear(&part);
                goto done;
            }
            if (!build_bucketed_sliced_execution_partition(exec_parts + local, &part, requested_bucket_count, &bucketed_bytes)) {
                sparse::clear(&part);
                goto done;
            }
            state->partition_sliced_ell_slice_counts[partition_id] = part.slice_count;
            state->partition_sliced_ell_bytes[partition_id] = sliced_execution_segment_bytes(&part);
            state->partition_bucketed_sliced_ell_bytes[partition_id] = bucketed_bytes;
            state->partition_execution_bytes[partition_id] = bucketed_bytes;
            state->partition_execution_formats[partition_id] = requested_bucket_count > 1u
                ? dataset_execution_format_bucketed_sliced_ell
                : dataset_execution_format_sliced_ell;
            shard_segment_total += exec_parts[local].segment_count;
            sparse::clear(&part);
        }
        state->shard_execution_formats[shard_id] = dataset_execution_format_bucketed_sliced_ell;
        state->shard_bucketed_partition_counts[shard_id] = (std::uint32_t) partition_count;
        state->shard_bucketed_segment_counts[shard_id] = shard_segment_total;
        state->shard_sliced_ell_slice_counts[shard_id] = shard_segment_total;
    } else {
        optimized_shard = &state->loaded_optimized_sliced_shard;
        if (optimized_shard->partition_count != partition_count) goto done;
        state->shard_execution_formats[shard_id] = dataset_execution_format_bucketed_sliced_ell;
        state->shard_bucketed_partition_counts[shard_id] = optimized_shard->partition_count;
        state->shard_bucketed_segment_counts[shard_id] = 0u;
        state->shard_sliced_ell_slice_counts[shard_id] = 0u;
        for (local = 0u; local < partition_count; ++local) {
            state->shard_bucketed_segment_counts[shard_id] += optimized_shard->partitions[local].segment_count;
            state->shard_sliced_ell_slice_counts[shard_id] += optimized_shard->partitions[local].segment_count;
        }
    }
    if (!build_execution_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) goto done;
    if (!build_execution_pack_path(state, shard_id, final_path, sizeof(final_path))) goto done;
    fp = std::fopen(tmp_path, "wb");
    if (fp == 0) goto done;
    std::setvbuf(fp, 0, _IOFBF, (std::size_t) 8u << 20u);
    if (!write_sharded_block(fp, execution_pack_magic, sizeof(execution_pack_magic), 1u)) goto done;
    if (!write_sharded_block(fp, &shard_id, sizeof(shard_id), 1u)) goto done;
    if (!write_sharded_block(fp, &partition_count, sizeof(partition_count), 1u)) goto done;
    for (local = 0u; local < partition_count; ++local) {
        const std::uint64_t zero = 0u;
        if (!write_sharded_block(fp, &zero, sizeof(zero), 1u)) goto done;
    }
    for (local = 0u; local < partition_count; ++local) {
        partition_offsets[local] = (std::uint64_t) ftello(fp);
        if (optimized_shard != 0) {
            if (!write_sliced_execution_partition_blob(fp, optimized_shard->partitions + local)) goto done;
        } else {
            if (!write_sliced_execution_partition_blob(fp, exec_parts + local)) goto done;
        }
    }
    pack_bytes = (std::uint64_t) ftello(fp);
    if (fseeko(fp, (off_t) (sizeof(execution_pack_magic) + sizeof(std::uint64_t) * 2u), SEEK_SET) != 0) goto done;
    if (!write_sharded_block(fp, partition_offsets, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
    if (std::fflush(fp) != 0) goto done;
    std::fclose(fp);
    fp = 0;
    if (::rename(tmp_path, final_path) != 0) {
        std::remove(tmp_path);
        goto done;
    }
    state->shard_bucketed_sliced_ell_bytes[shard_id] = pack_bytes;
    state->shard_execution_bytes[shard_id] = pack_bytes;
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    if (!ok && build_execution_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) std::remove(tmp_path);
    if (exec_parts != 0) {
        for (local = 0u; local < partition_count; ++local) clear(exec_parts + local);
    }
    std::free(partition_offsets);
    std::free(exec_parts);
    return ok;
}

inline int ensure_execution_pack_ready(shard_storage *s, dataset_h5_state *state, unsigned long shard_id) {
    char path[4096];
    if (s == 0 || state == 0 || shard_id >= state->num_shards) return 0;
    if (!ensure_dataset_cache_layout(s)) return 0;
    if (!build_execution_pack_path(state, shard_id, path, sizeof(path))) return 0;
    if (::access(path, R_OK) == 0) return 1;
    if (state->matrix_family == dataset_matrix_family_sliced_ell) return materialize_sliced_ell_execution_pack(s, state, shard_id);
    return materialize_blocked_ell_execution_pack(s, state, shard_id);
}

inline int materialize_compressed_shard_pack(shard_storage *s, dataset_h5_state *state, unsigned long shard_id) {
    const std::uint64_t begin = state != 0 && state->shard_part_begin != 0 ? state->shard_part_begin[shard_id] : 0u;
    const std::uint64_t end = state != 0 && state->shard_part_end != 0 ? state->shard_part_end[shard_id] : 0u;
    const std::uint64_t partition_count = end >= begin ? (end - begin) : 0u;
    sparse::compressed **parts = 0;
    char tmp_path[4096];
    char final_path[4096];
    std::uint64_t local = 0u;
    int ok = 0;

    if (s == 0 || state == 0 || shard_id >= state->num_shards) return 0;
    if (!open_dataset_h5_backend(s) || !ensure_standard_payload_open(state)) return 0;
    if (partition_count != 0u) {
        parts = (sparse::compressed **) std::calloc((std::size_t) partition_count, sizeof(sparse::compressed *));
        if (parts == 0) return 0;
    }
    for (local = 0u; local < partition_count; ++local) {
        const std::uint64_t partition_id = begin + local;
        const dataset_codec_descriptor *codec = find_codec(state, state->partition_codec_ids[partition_id]);
        sparse::compressed *part = new sparse::compressed;
        sparse::init(part,
                     (types::dim_t) state->partition_rows[partition_id],
                     (types::dim_t) state->cols,
                     (types::nnz_t) state->partition_nnz[partition_id],
                     (types::u32) state->partition_aux[partition_id]);
        if (codec == 0 || codec->family != dataset_codec_family_standard_csr) {
            sparse::clear(part);
            delete part;
            goto done;
        }
        if (!sparse::allocate(part)) {
            sparse::clear(part);
            delete part;
            goto done;
        }
        if (!read_hyperslab_1d(state->d_standard_indptr,
                               H5T_NATIVE_UINT32,
                               state->partition_indptr_offsets[partition_id],
                               state->partition_rows[partition_id] + 1u,
                               part->majorPtr)) {
            sparse::clear(part);
            delete part;
            goto done;
        }
        if (!read_hyperslab_1d(state->d_standard_indices,
                               H5T_NATIVE_UINT32,
                               state->partition_nnz_offsets[partition_id],
                               state->partition_nnz[partition_id],
                               part->minorIdx)) {
            sparse::clear(part);
            delete part;
            goto done;
        }
        if (!read_hyperslab_1d(state->d_standard_values,
                               H5T_NATIVE_UINT16,
                               state->partition_nnz_offsets[partition_id],
                               state->partition_nnz[partition_id],
                               part->val)) {
            sparse::clear(part);
            delete part;
            goto done;
        }
        parts[local] = part;
    }
    if (!build_shard_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) goto done;
    if (!build_shard_pack_path(state, shard_id, final_path, sizeof(final_path))) goto done;
    if (!write_shard_pack_file<sparse::compressed>(tmp_path,
                                                   state->cols,
                                                   state->partition_rows + begin,
                                                   state->partition_nnz + begin,
                                                   state->partition_aux + begin,
                                                   partition_count,
                                                   parts)) {
        goto done;
    }
    if (::rename(tmp_path, final_path) != 0) {
        std::remove(tmp_path);
        goto done;
    }
    ok = 1;

done:
    if (!ok && build_shard_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) std::remove(tmp_path);
    if (parts != 0) {
        for (local = 0u; local < partition_count; ++local) {
            if (parts[local] != 0) {
                sparse::clear(parts[local]);
                delete parts[local];
            }
        }
    }
    std::free(parts);
    return ok;
}

inline int materialize_blocked_ell_shard_pack(shard_storage *s, dataset_h5_state *state, unsigned long shard_id) {
    const std::uint64_t begin = state != 0 && state->shard_part_begin != 0 ? state->shard_part_begin[shard_id] : 0u;
    const std::uint64_t end = state != 0 && state->shard_part_end != 0 ? state->shard_part_end[shard_id] : 0u;
    const std::uint64_t partition_count = end >= begin ? (end - begin) : 0u;
    sparse::blocked_ell **parts = 0;
    char tmp_path[4096];
    char final_path[4096];
    int ok = 0;

    if (s == 0 || state == 0 || shard_id >= state->num_shards) return 0;
    if (!open_dataset_h5_backend(s)) return 0;
    if (partition_count != 0u) {
        parts = (sparse::blocked_ell **) std::calloc((std::size_t) partition_count, sizeof(sparse::blocked_ell *));
        if (parts == 0) return 0;
    }
    if (state->matrix_family == dataset_matrix_family_optimized_blocked_ell) {
        if (!load_optimized_blocked_ell_shard_payload(state, shard_id)) goto done;
        if (state->loaded_optimized_shard.partition_count != partition_count) goto done;
        for (std::uint64_t local = 0u; local < partition_count; ++local) {
            const std::uint64_t partition_id = begin + local;
            const std::uint32_t block_size =
                sparse::unpack_blocked_ell_block_size((unsigned long) state->partition_aux[partition_id]);
            sparse::blocked_ell *part = new sparse::blocked_ell;
            sparse::init(part);
            if (!reconstruct_canonical_blocked_ell_part(state->loaded_optimized_shard.partitions + local,
                                                        state->loaded_optimized_shard.exec_to_canonical_cols,
                                                        block_size,
                                                        part)) {
                sparse::clear(part);
                delete part;
                goto done;
            }
            parts[local] = part;
        }
    } else {
        if (!load_blocked_ell_shard_payload(state, shard_id)) return 0;
        if (!prepare_blocked_ell_parts_from_state(state, (unsigned long) begin, (unsigned long) end, parts)) goto done;
        if (!fill_blocked_ell_parts_from_loaded_shard(state, shard_id, (unsigned long) begin, (unsigned long) end, parts)) goto done;
    }
    if (!build_shard_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) goto done;
    if (!build_shard_pack_path(state, shard_id, final_path, sizeof(final_path))) goto done;
    if (!write_shard_pack_file<sparse::blocked_ell>(tmp_path,
                                                    state->cols,
                                                    state->partition_rows + begin,
                                                    state->partition_nnz + begin,
                                                    state->partition_aux + begin,
                                                    partition_count,
                                                    parts)) {
        goto done;
    }
    if (::rename(tmp_path, final_path) != 0) {
        std::remove(tmp_path);
        goto done;
    }
    ok = 1;

done:
    if (!ok && build_shard_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) std::remove(tmp_path);
    clear_blocked_ell_parts(parts, (unsigned long) partition_count);
    std::free(parts);
    return ok;
}

inline int materialize_sliced_ell_shard_pack(shard_storage *s, dataset_h5_state *state, unsigned long shard_id) {
    const std::uint64_t begin = state != 0 && state->shard_part_begin != 0 ? state->shard_part_begin[shard_id] : 0u;
    const std::uint64_t end = state != 0 && state->shard_part_end != 0 ? state->shard_part_end[shard_id] : 0u;
    const std::uint64_t partition_count = end >= begin ? (end - begin) : 0u;
    sparse::sliced_ell **parts = 0;
    char tmp_path[4096];
    char final_path[4096];
    std::uint64_t local = 0u;
    int ok = 0;

    if (s == 0 || state == 0 || shard_id >= state->num_shards) return 0;
    if (!open_dataset_h5_backend(s)) return 0;
    if (partition_count != 0u) {
        parts = (sparse::sliced_ell **) std::calloc((std::size_t) partition_count, sizeof(sparse::sliced_ell *));
        if (parts == 0) return 0;
    }
    for (local = 0u; local < partition_count; ++local) {
        const std::uint64_t partition_id = begin + local;
        sparse::sliced_ell *part = new sparse::sliced_ell;
        sparse::init(part);
        if (!load_sliced_ell_partition_payload(state, (unsigned long) partition_id, part)) {
            sparse::clear(part);
            delete part;
            goto done;
        }
        parts[local] = part;
    }
    if (!build_shard_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) goto done;
    if (!build_shard_pack_path(state, shard_id, final_path, sizeof(final_path))) goto done;
    if (!write_shard_pack_file<sparse::sliced_ell>(tmp_path,
                                                   state->cols,
                                                   state->partition_rows + begin,
                                                   state->partition_nnz + begin,
                                                   state->partition_aux + begin,
                                                   partition_count,
                                                   parts)) {
        goto done;
    }
    if (::rename(tmp_path, final_path) != 0) {
        std::remove(tmp_path);
        goto done;
    }
    ok = 1;

done:
    if (!ok && build_shard_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) std::remove(tmp_path);
    if (parts != 0) {
        for (local = 0u; local < partition_count; ++local) {
            if (parts[local] != 0) {
                sparse::clear(parts[local]);
                delete parts[local];
            }
        }
    }
    std::free(parts);
    return ok;
}

inline int materialize_shard_pack(shard_storage *s, dataset_h5_state *state, unsigned long shard_id) {
    if (state == 0) return 0;
    if (state->matrix_family == dataset_matrix_family_compressed) return materialize_compressed_shard_pack(s, state, shard_id);
    if (state->matrix_family == dataset_matrix_family_blocked_ell) return materialize_blocked_ell_shard_pack(s, state, shard_id);
    if (state->matrix_family == dataset_matrix_family_optimized_blocked_ell) return materialize_blocked_ell_shard_pack(s, state, shard_id);
    if (state->matrix_family == dataset_matrix_family_sliced_ell) return materialize_sliced_ell_shard_pack(s, state, shard_id);
    return 0;
}

inline void touch_shard_locked(dataset_h5_state *state, unsigned long shard_id) {
    if (state == 0 || shard_id >= state->num_shards) return;
    state->shard_access_count[shard_id] += 1u;
    state->shard_last_access_tick[shard_id] = ++state->access_clock;
    state->last_requested_shard = shard_id;
}

inline std::uint64_t shard_eviction_score(const dataset_h5_state *state,
                                          unsigned long shard_id,
                                          std::uint64_t now_tick) {
    const std::uint64_t age = now_tick >= state->shard_last_access_tick[shard_id] ? (now_tick - state->shard_last_access_tick[shard_id]) : 0u;
    const std::uint64_t accesses = state->shard_access_count[shard_id];
    return (age + 1u) * (state->shard_cache_bytes[shard_id] + 1u) / (accesses + 1u);
}

inline void evict_cached_shard_locked(dataset_h5_state *state, unsigned long shard_id) {
    if (state == 0 || shard_id >= state->num_shards) return;
    close_cached_shard_file(state, shard_id);
    if (state->shard_cache_paths != 0 && state->shard_cache_paths[shard_id] != 0) ::unlink(state->shard_cache_paths[shard_id]);
    if (state->shard_cache_state[shard_id] == dataset_cache_shard_ready) {
        if (state->cache_resident_bytes >= state->shard_cache_bytes[shard_id]) state->cache_resident_bytes -= state->shard_cache_bytes[shard_id];
        else state->cache_resident_bytes = 0u;
    }
    state->shard_cache_state[shard_id] = dataset_cache_shard_missing;
    state->shard_cache_bytes[shard_id] = 0u;
    state->shard_access_count[shard_id] = 0u;
    state->shard_last_access_tick[shard_id] = 0u;
    state->shard_pin_count[shard_id] = 0u;
}

inline void maybe_evict_cached_shards_locked(dataset_h5_state *state, unsigned long keep_shard_id) {
    while (state != 0
           && state->predictor_enabled
           && state->cache_budget_bytes != 0u
           && state->cache_resident_bytes > state->cache_budget_bytes) {
        unsigned long victim = (unsigned long) state->num_shards;
        std::uint64_t best_score = 0u;
        unsigned long shard_id = 0ul;
        for (shard_id = 0ul; shard_id < (unsigned long) state->num_shards; ++shard_id) {
            const std::uint64_t score = shard_eviction_score(state, shard_id, state->access_clock + 1u);
            if (shard_id == keep_shard_id) continue;
            if (state->shard_cache_state[shard_id] != dataset_cache_shard_ready) continue;
            if (state->shard_pin_count[shard_id] != 0u) continue;
            if (victim >= state->num_shards || score > best_score) {
                victim = shard_id;
                best_score = score;
            }
        }
        if (victim >= state->num_shards) break;
        evict_cached_shard_locked(state, victim);
    }
}

inline void reader_materialize_loop(shard_storage *s) {
    dataset_h5_state *state = s != 0 ? (dataset_h5_state *) s->backend_state : 0;
    dataset_h5_cache_runtime *runtime = cache_runtime(state);
    if (state == 0 || runtime == 0) return;
    for (;;) {
        unsigned long shard_id = 0ul;
        {
            std::unique_lock<std::mutex> lock(runtime->state_mutex);
            runtime->state_cv.wait(lock, [&]() { return runtime->stop_requested || !runtime->shard_queue.empty(); });
            if (runtime->stop_requested) break;
            shard_id = runtime->shard_queue.front();
            runtime->shard_queue.pop_front();
            state->shard_cache_state[shard_id] = dataset_cache_shard_building;
        }

        const int ok = materialize_shard_pack(s, state, shard_id);

        {
            std::lock_guard<std::mutex> lock(runtime->state_mutex);
            if (ok) {
                if (state->shard_cache_state[shard_id] != dataset_cache_shard_ready) {
                    state->shard_cache_bytes[shard_id] = estimate_shard_pack_bytes(state, shard_id);
                    state->cache_resident_bytes += state->shard_cache_bytes[shard_id];
                }
                state->shard_cache_state[shard_id] = dataset_cache_shard_ready;
                touch_shard_locked(state, shard_id);
                maybe_evict_cached_shards_locked(state, shard_id);
            } else {
                state->shard_cache_state[shard_id] = dataset_cache_shard_failed;
            }
            runtime->state_cv.notify_all();
        }
    }
}

inline int ensure_cache_reader_started(shard_storage *s) {
    dataset_h5_state *state = 0;
    dataset_h5_cache_runtime *runtime = 0;
    if (s == 0 || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
    runtime = cache_runtime(state);
    if (runtime == 0) return 0;
    if (runtime->reader_started) return 1;
    runtime->stop_requested = false;
    runtime->reader_thread = std::thread(reader_materialize_loop, s);
    runtime->reader_started = true;
    return 1;
}

inline int ensure_cached_shard_ready(shard_storage *s, unsigned long shard_id) {
    dataset_h5_state *state = 0;
    dataset_h5_cache_runtime *runtime = 0;
    if (s == 0 || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
    if (!ensure_dataset_cache_layout(s)) return 0;
    runtime = cache_runtime(state);
    if (runtime == 0 || shard_id >= state->num_shards) return 0;
    if (!ensure_cache_reader_started(s)) return 0;

    {
        std::unique_lock<std::mutex> lock(runtime->state_mutex);
        if (state->shard_cache_state[shard_id] == dataset_cache_shard_ready) {
            touch_shard_locked(state, shard_id);
            return 1;
        }
        if (state->shard_cache_state[shard_id] == dataset_cache_shard_missing
            || state->shard_cache_state[shard_id] == dataset_cache_shard_failed) {
            state->shard_cache_state[shard_id] = dataset_cache_shard_queued;
            runtime->shard_queue.push_back(shard_id);
            runtime->state_cv.notify_all();
        }
        runtime->state_cv.wait(lock, [&]() {
            return state->shard_cache_state[shard_id] == dataset_cache_shard_ready
                || state->shard_cache_state[shard_id] == dataset_cache_shard_failed;
        });
        if (state->shard_cache_state[shard_id] != dataset_cache_shard_ready) return 0;
        touch_shard_locked(state, shard_id);
    }
    return 1;
}

int open_dataset_h5_backend(shard_storage *s) {
    dataset_h5_state *state = 0;
    if (s == 0 || s->source_path == 0 || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
    if (state->file >= 0) return 1;
    state->file = H5Fopen(s->source_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    return state->file >= 0;
}

void close_dataset_h5_backend(shard_storage *s) {
    dataset_h5_state *state = 0;
    if (s == 0 || s->backend_state == 0) return;
    state = (dataset_h5_state *) s->backend_state;
    dataset_h5_state_clear(state);
    std::free(state);
    s->backend_state = 0;
    s->open_backend = 0;
    s->close_backend = 0;
    s->backend = shard_storage_backend_none;
}

} // namespace

int create_dataset_compressed_h5(const char *filename,
                                const dataset_layout_view *layout,
                                const dataset_dataset_table_view *datasets,
                                const dataset_provenance_view *provenance) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    hid_t dsets = (hid_t) -1;
    hid_t prov = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    hid_t payload_root = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    std::uint64_t total_indptr = 0;
    std::uint64_t total_nnz = 0;
    std::uint64_t *partition_indptr_offsets = 0;
    std::uint64_t *partition_nnz_offsets = 0;
    std::uint64_t *partition_aux = 0;
    std::uint32_t i = 0;
    int ok = 0;
    const std::uint64_t dim_limit = local_dim_limit();
    const std::uint64_t nnz_limit = local_nnz_limit();
    unsigned long shard_part_begin = 0ul;

    if (filename == 0 || layout == 0) return 0;
    if (layout->partition_rows == 0 || layout->partition_nnz == 0 || layout->partition_axes == 0 || layout->partition_row_offsets == 0 || layout->partition_dataset_ids == 0 || layout->partition_codec_ids == 0 || layout->shard_offsets == 0) return 0;
    if (layout->cols > local_index_limit()) {
        std::fprintf(stderr,
                     "cellshard: dataset column count exceeds the current u32 execution limit while writing %s (cols=%llu, limit=%llu)\n",
                     filename,
                     (unsigned long long) layout->cols,
                     (unsigned long long) local_index_limit());
        return 0;
    }

    partition_indptr_offsets = (std::uint64_t *) std::calloc((std::size_t) layout->num_partitions, sizeof(std::uint64_t));
    partition_nnz_offsets = (std::uint64_t *) std::calloc((std::size_t) layout->num_partitions, sizeof(std::uint64_t));
    partition_aux = (std::uint64_t *) std::calloc((std::size_t) layout->num_partitions, sizeof(std::uint64_t));
    if ((layout->num_partitions != 0) && (partition_indptr_offsets == 0 || partition_nnz_offsets == 0 || partition_aux == 0)) goto done;

    for (i = 0; i < layout->num_partitions; ++i) {
        if (layout->partition_rows[i] > dim_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "rows", layout->partition_rows[i], dim_limit);
            goto done;
        }
        if (layout->partition_nnz[i] > nnz_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "nnz", layout->partition_nnz[i], nnz_limit);
            goto done;
        }
        partition_indptr_offsets[i] = total_indptr;
        partition_nnz_offsets[i] = total_nnz;
        total_indptr += layout->partition_rows[i] + 1u;
        total_nnz += layout->partition_nnz[i];
        partition_aux[i] = layout->partition_aux != 0 ? layout->partition_aux[i] : (std::uint64_t) layout->partition_axes[i];
    }

    for (std::uint32_t shard_i = 0; shard_i < layout->num_shards; ++shard_i) {
        const std::uint64_t row_begin = layout->shard_offsets[shard_i];
        const std::uint64_t row_end = layout->shard_offsets[shard_i + 1u];
        std::uint64_t shard_nnz = 0u;
        unsigned long part_end = shard_part_begin;
        while (shard_part_begin < layout->num_partitions && layout->partition_row_offsets[shard_part_begin] < row_begin) ++shard_part_begin;
        part_end = shard_part_begin;
        while (part_end < layout->num_partitions && layout->partition_row_offsets[part_end + 1u] <= row_end) {
            shard_nnz += layout->partition_nnz[part_end];
            ++part_end;
        }
        if (row_end - row_begin > dim_limit) warn_dataset_u32_limit(filename, "shard", shard_i, "rows", row_end - row_begin, dim_limit);
        if (shard_nnz > nnz_limit) warn_dataset_u32_limit(filename, "shard", shard_i, "nnz", shard_nnz, nnz_limit);
        shard_part_begin = part_end;
    }

    file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) goto done;
    if (!write_attr_string(file, "cellshard_magic", dataset_magic)) goto done;
    if (!write_attr_u32(file, "schema_version", dataset_h5_schema_version)) goto done;
    if (!write_attr_string(file, "matrix_format", "compressed")) goto done;
    if (!write_attr_u64(file, "rows", layout->rows)) goto done;
    if (!write_attr_u64(file, "cols", layout->cols)) goto done;
    if (!write_attr_u64(file, "nnz", layout->nnz)) goto done;
    if (!write_attr_u64(file, "num_partitions", layout->num_partitions)) goto done;
    if (!write_attr_u64(file, "num_shards", layout->num_shards)) goto done;
    if (!write_attr_u64(file, "num_codecs", layout->num_codecs)) goto done;
    if (!write_attr_u64(file, "num_datasets", datasets != 0 ? datasets->count : 0u)) goto done;

    matrix = create_group(file, matrix_group);
    dsets = create_group(file, datasets_group);
    prov = create_group(file, provenance_group);
    codecs = create_group(file, codecs_group);
    payload_root = create_group(file, payload_group);
    payload = payload_root >= 0 ? create_group(payload_root, "standard_csr") : (hid_t) -1;
    if (matrix < 0 || dsets < 0 || prov < 0 || codecs < 0 || payload_root < 0 || payload < 0) goto done;

    if (!write_dataset_1d(matrix, "partition_rows", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, layout->partition_rows)) goto done;
    if (!write_dataset_1d(matrix, "partition_nnz", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, layout->partition_nnz)) goto done;
    if (!write_dataset_1d(matrix, "partition_axes", H5T_NATIVE_UINT32, (hsize_t) layout->num_partitions, layout->partition_axes)) goto done;
    if (!write_dataset_1d(matrix, "partition_aux", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, partition_aux)) goto done;
    if (!write_dataset_1d(matrix, "partition_row_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions + 1u, layout->partition_row_offsets)) goto done;
    if (!write_dataset_1d(matrix, "partition_dataset_ids", H5T_NATIVE_UINT32, (hsize_t) layout->num_partitions, layout->partition_dataset_ids)) goto done;
    if (!write_dataset_1d(matrix, "partition_codec_ids", H5T_NATIVE_UINT32, (hsize_t) layout->num_partitions, layout->partition_codec_ids)) goto done;
    if (!write_dataset_1d(matrix, "shard_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_shards + 1u, layout->shard_offsets)) goto done;

    if (datasets != 0) {
        if (!write_text_column(dsets, "dataset_ids", &datasets->dataset_ids)) goto done;
        if (!write_text_column(dsets, "matrix_paths", &datasets->matrix_paths)) goto done;
        if (!write_text_column(dsets, "feature_paths", &datasets->feature_paths)) goto done;
        if (!write_text_column(dsets, "barcode_paths", &datasets->barcode_paths)) goto done;
        if (!write_text_column(dsets, "metadata_paths", &datasets->metadata_paths)) goto done;
        if (!write_dataset_1d(dsets, "formats", H5T_NATIVE_UINT32, (hsize_t) datasets->count, datasets->formats)) goto done;
        if (!write_dataset_1d(dsets, "row_begin", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->row_begin)) goto done;
        if (!write_dataset_1d(dsets, "row_end", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->row_end)) goto done;
        if (!write_dataset_1d(dsets, "rows", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->rows)) goto done;
        if (!write_dataset_1d(dsets, "cols", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->cols)) goto done;
        if (!write_dataset_1d(dsets, "nnz", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->nnz)) goto done;
    }

    if (provenance != 0) {
        if (!write_text_column(prov, "global_barcodes", &provenance->global_barcodes)) goto done;
        if (!write_dataset_1d(prov, "cell_dataset_ids", H5T_NATIVE_UINT32, (hsize_t) layout->rows, provenance->cell_dataset_ids)) goto done;
        if (!write_dataset_1d(prov, "cell_local_indices", H5T_NATIVE_UINT64, (hsize_t) layout->rows, provenance->cell_local_indices)) goto done;
        if (!write_text_column(prov, "feature_ids", &provenance->feature_ids)) goto done;
        if (!write_text_column(prov, "feature_names", &provenance->feature_names)) goto done;
        if (!write_text_column(prov, "feature_types", &provenance->feature_types)) goto done;
        if (!write_dataset_1d(prov, "feature_dataset_ids", H5T_NATIVE_UINT32, (hsize_t) layout->cols, provenance->feature_dataset_ids)) goto done;
        if (!write_dataset_1d(prov, "feature_local_indices", H5T_NATIVE_UINT64, (hsize_t) layout->cols, provenance->feature_local_indices)) goto done;
        if (datasets != 0) {
            if (!write_dataset_1d(prov, "dataset_feature_offsets", H5T_NATIVE_UINT64, (hsize_t) datasets->count + 1u, provenance->dataset_feature_offsets)) goto done;
            if (!write_dataset_1d(prov, "dataset_feature_to_global", H5T_NATIVE_UINT32, (hsize_t) provenance->dataset_feature_offsets[datasets->count], provenance->dataset_feature_to_global)) goto done;
        }
    }

    if (layout->num_codecs != 0) {
        std::uint32_t *codec_id = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *family = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *value_code = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *scale_value_code = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *bits = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *flags = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        if (codec_id == 0 || family == 0 || value_code == 0 || scale_value_code == 0 || bits == 0 || flags == 0) {
            std::free(codec_id);
            std::free(family);
            std::free(value_code);
            std::free(scale_value_code);
            std::free(bits);
            std::free(flags);
            goto done;
        }
        for (i = 0; i < layout->num_codecs; ++i) {
            codec_id[i] = layout->codecs[i].codec_id;
            family[i] = layout->codecs[i].family;
            value_code[i] = layout->codecs[i].value_code;
            scale_value_code[i] = layout->codecs[i].scale_value_code;
            bits[i] = layout->codecs[i].bits;
            flags[i] = layout->codecs[i].flags;
        }
        if (!write_dataset_1d(codecs, "codec_id", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, codec_id)) goto done;
        if (!write_dataset_1d(codecs, "family", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, family)) goto done;
        if (!write_dataset_1d(codecs, "value_code", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, value_code)) goto done;
        if (!write_dataset_1d(codecs, "scale_value_code", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, scale_value_code)) goto done;
        if (!write_dataset_1d(codecs, "bits", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, bits)) goto done;
        if (!write_dataset_1d(codecs, "flags", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, flags)) goto done;
        std::free(codec_id);
        std::free(family);
        std::free(value_code);
        std::free(scale_value_code);
        std::free(bits);
        std::free(flags);
    }

    if (!write_dataset_1d(payload, "partition_indptr_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, partition_indptr_offsets)) goto done;
    if (!write_dataset_1d(payload, "partition_nnz_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, partition_nnz_offsets)) goto done;
    if (!write_dataset_1d(payload, "indptr", H5T_NATIVE_UINT32, (hsize_t) total_indptr, 0)) goto done;
    if (!write_dataset_1d(payload, "indices", H5T_NATIVE_UINT32, (hsize_t) total_nnz, 0)) goto done;
    if (!write_dataset_1d(payload, "values", H5T_NATIVE_UINT16, (hsize_t) total_nnz, 0)) goto done;

    ok = 1;

done:
    std::free(partition_indptr_offsets);
    std::free(partition_nnz_offsets);
    std::free(partition_aux);
    if (payload >= 0) H5Gclose(payload);
    if (payload_root >= 0) H5Gclose(payload_root);
    if (codecs >= 0) H5Gclose(codecs);
    if (prov >= 0) H5Gclose(prov);
    if (dsets >= 0) H5Gclose(dsets);
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int create_dataset_blocked_ell_h5(const char *filename,
                                 const dataset_layout_view *layout,
                                 const dataset_dataset_table_view *datasets,
                                 const dataset_provenance_view *provenance) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    hid_t dsets = (hid_t) -1;
    hid_t prov = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    hid_t payload_root = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    std::uint64_t total_block_idx = 0;
    std::uint64_t total_values = 0;
    std::uint64_t *partition_aux = 0;
    std::uint32_t *partition_axes = 0;
    std::uint64_t *partition_block_idx_offsets = 0;
    std::uint64_t *partition_value_offsets = 0;
    std::uint64_t *shard_block_idx_offsets = 0;
    std::uint64_t *shard_value_offsets = 0;
    std::uint32_t i = 0;
    std::uint32_t shard_i = 0;
    int ok = 0;
    const std::uint64_t dim_limit = local_dim_limit();
    const std::uint64_t nnz_limit = local_nnz_limit();
    const std::uint64_t idx_limit = local_index_limit();

    if (filename == 0 || layout == 0) return 0;
    if (layout->partition_rows == 0 || layout->partition_nnz == 0 || layout->partition_aux == 0 || layout->partition_row_offsets == 0 || layout->partition_dataset_ids == 0 || layout->partition_codec_ids == 0 || layout->shard_offsets == 0) return 0;
    if (layout->cols > idx_limit) {
        std::fprintf(stderr,
                     "cellshard: dataset column count exceeds the current u32 execution limit while writing %s (cols=%llu, limit=%llu)\n",
                     filename,
                     (unsigned long long) layout->cols,
                     (unsigned long long) idx_limit);
        return 0;
    }

    partition_aux = (std::uint64_t *) std::calloc((std::size_t) layout->num_partitions, sizeof(std::uint64_t));
    partition_axes = (std::uint32_t *) std::calloc((std::size_t) layout->num_partitions, sizeof(std::uint32_t));
    partition_block_idx_offsets = (std::uint64_t *) std::calloc((std::size_t) layout->num_partitions, sizeof(std::uint64_t));
    partition_value_offsets = (std::uint64_t *) std::calloc((std::size_t) layout->num_partitions, sizeof(std::uint64_t));
    shard_block_idx_offsets = (std::uint64_t *) std::calloc((std::size_t) layout->num_shards + 1u, sizeof(std::uint64_t));
    shard_value_offsets = (std::uint64_t *) std::calloc((std::size_t) layout->num_shards + 1u, sizeof(std::uint64_t));
    if ((layout->num_partitions != 0) && (partition_aux == 0 || partition_axes == 0 || partition_block_idx_offsets == 0 || partition_value_offsets == 0)) goto done;
    if (layout->num_shards != 0 && (shard_block_idx_offsets == 0 || shard_value_offsets == 0)) goto done;

    for (i = 0; i < layout->num_partitions; ++i) {
        const std::uint64_t part_block_idx = (std::uint64_t) blocked_ell_part_block_index_count(layout->partition_rows[i], layout->partition_aux[i]);
        const std::uint64_t part_values = (std::uint64_t) blocked_ell_part_value_count(layout->partition_rows[i], layout->partition_aux[i]);
        if (layout->partition_rows[i] > dim_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "rows", layout->partition_rows[i], dim_limit);
            goto done;
        }
        if (layout->partition_nnz[i] > nnz_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "nnz", layout->partition_nnz[i], nnz_limit);
            goto done;
        }
        if (part_block_idx > idx_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "block_col_idx_count", part_block_idx, idx_limit);
            goto done;
        }
        if (part_values > nnz_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "value_count", part_values, nnz_limit);
            goto done;
        }
        partition_aux[i] = layout->partition_aux[i];
        partition_axes[i] = layout->partition_axes != 0 ? layout->partition_axes[i] : 0u;
        partition_block_idx_offsets[i] = total_block_idx;
        partition_value_offsets[i] = total_values;
        total_block_idx += part_block_idx;
        total_values += part_values;
    }

    file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) goto done;
    if (!write_attr_string(file, "cellshard_magic", dataset_magic)) goto done;
    if (!write_attr_u32(file, "schema_version", dataset_h5_schema_version)) goto done;
    if (!write_attr_string(file, "matrix_format", "blocked_ell")) goto done;
    if (!write_attr_string(file, "payload_layout", payload_layout_shard_packed)) {
        goto done;
    }
    if (!write_attr_u64(file, "rows", layout->rows)) goto done;
    if (!write_attr_u64(file, "cols", layout->cols)) goto done;
    if (!write_attr_u64(file, "nnz", layout->nnz)) goto done;
    if (!write_attr_u64(file, "num_partitions", layout->num_partitions)) goto done;
    if (!write_attr_u64(file, "num_shards", layout->num_shards)) goto done;
    if (!write_attr_u64(file, "num_codecs", layout->num_codecs)) goto done;
    if (!write_attr_u64(file, "num_datasets", datasets != 0 ? datasets->count : 0u)) goto done;

    matrix = create_group(file, matrix_group);
    dsets = create_group(file, datasets_group);
    prov = create_group(file, provenance_group);
    codecs = create_group(file, codecs_group);
    payload_root = create_group(file, payload_group);
    payload = payload_root >= 0 ? create_group(payload_root, "blocked_ell") : (hid_t) -1;
    if (matrix < 0 || dsets < 0 || prov < 0 || codecs < 0 || payload_root < 0 || payload < 0) goto done;

    if (!write_dataset_1d(matrix, "partition_rows", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, layout->partition_rows)) goto done;
    if (!write_dataset_1d(matrix, "partition_nnz", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, layout->partition_nnz)) goto done;
    if (!write_dataset_1d(matrix, "partition_axes", H5T_NATIVE_UINT32, (hsize_t) layout->num_partitions, partition_axes)) goto done;
    if (!write_dataset_1d(matrix, "partition_aux", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, partition_aux)) goto done;
    if (!write_dataset_1d(matrix, "partition_row_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions + 1u, layout->partition_row_offsets)) goto done;
    if (!write_dataset_1d(matrix, "partition_dataset_ids", H5T_NATIVE_UINT32, (hsize_t) layout->num_partitions, layout->partition_dataset_ids)) goto done;
    if (!write_dataset_1d(matrix, "partition_codec_ids", H5T_NATIVE_UINT32, (hsize_t) layout->num_partitions, layout->partition_codec_ids)) goto done;
    if (!write_dataset_1d(matrix, "shard_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_shards + 1u, layout->shard_offsets)) goto done;

    if (datasets != 0) {
        if (!write_text_column(dsets, "dataset_ids", &datasets->dataset_ids)) goto done;
        if (!write_text_column(dsets, "matrix_paths", &datasets->matrix_paths)) goto done;
        if (!write_text_column(dsets, "feature_paths", &datasets->feature_paths)) goto done;
        if (!write_text_column(dsets, "barcode_paths", &datasets->barcode_paths)) goto done;
        if (!write_text_column(dsets, "metadata_paths", &datasets->metadata_paths)) goto done;
        if (!write_dataset_1d(dsets, "formats", H5T_NATIVE_UINT32, (hsize_t) datasets->count, datasets->formats)) goto done;
        if (!write_dataset_1d(dsets, "row_begin", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->row_begin)) goto done;
        if (!write_dataset_1d(dsets, "row_end", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->row_end)) goto done;
        if (!write_dataset_1d(dsets, "rows", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->rows)) goto done;
        if (!write_dataset_1d(dsets, "cols", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->cols)) goto done;
        if (!write_dataset_1d(dsets, "nnz", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->nnz)) goto done;
    }

    if (provenance != 0) {
        if (!write_text_column(prov, "global_barcodes", &provenance->global_barcodes)) goto done;
        if (!write_dataset_1d(prov, "cell_dataset_ids", H5T_NATIVE_UINT32, (hsize_t) layout->rows, provenance->cell_dataset_ids)) goto done;
        if (!write_dataset_1d(prov, "cell_local_indices", H5T_NATIVE_UINT64, (hsize_t) layout->rows, provenance->cell_local_indices)) goto done;
        if (!write_text_column(prov, "feature_ids", &provenance->feature_ids)) goto done;
        if (!write_text_column(prov, "feature_names", &provenance->feature_names)) goto done;
        if (!write_text_column(prov, "feature_types", &provenance->feature_types)) goto done;
        if (!write_dataset_1d(prov, "feature_dataset_ids", H5T_NATIVE_UINT32, (hsize_t) layout->cols, provenance->feature_dataset_ids)) goto done;
        if (!write_dataset_1d(prov, "feature_local_indices", H5T_NATIVE_UINT64, (hsize_t) layout->cols, provenance->feature_local_indices)) goto done;
        if (datasets != 0) {
            if (!write_dataset_1d(prov, "dataset_feature_offsets", H5T_NATIVE_UINT64, (hsize_t) datasets->count + 1u, provenance->dataset_feature_offsets)) goto done;
            if (!write_dataset_1d(prov, "dataset_feature_to_global", H5T_NATIVE_UINT32, (hsize_t) provenance->dataset_feature_offsets[datasets->count], provenance->dataset_feature_to_global)) goto done;
        }
    }

    if (layout->num_codecs != 0) {
        std::uint32_t *codec_id = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *family = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *value_code = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *scale_value_code = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *bits = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *flags = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        if (codec_id == 0 || family == 0 || value_code == 0 || scale_value_code == 0 || bits == 0 || flags == 0) {
            std::free(codec_id);
            std::free(family);
            std::free(value_code);
            std::free(scale_value_code);
            std::free(bits);
            std::free(flags);
            goto done;
        }
        for (i = 0; i < layout->num_codecs; ++i) {
            codec_id[i] = layout->codecs[i].codec_id;
            family[i] = layout->codecs[i].family;
            value_code[i] = layout->codecs[i].value_code;
            scale_value_code[i] = layout->codecs[i].scale_value_code;
            bits[i] = layout->codecs[i].bits;
            flags[i] = layout->codecs[i].flags;
        }
        if (!write_dataset_1d(codecs, "codec_id", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, codec_id)) goto done;
        if (!write_dataset_1d(codecs, "family", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, family)) goto done;
        if (!write_dataset_1d(codecs, "value_code", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, value_code)) goto done;
        if (!write_dataset_1d(codecs, "scale_value_code", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, scale_value_code)) goto done;
        if (!write_dataset_1d(codecs, "bits", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, bits)) goto done;
        if (!write_dataset_1d(codecs, "flags", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, flags)) goto done;
        std::free(codec_id);
        std::free(family);
        std::free(value_code);
        std::free(scale_value_code);
        std::free(bits);
        std::free(flags);
    }

    if (layout->num_shards != 0) {
        unsigned long part_begin = 0;
        for (shard_i = 0; shard_i < layout->num_shards; ++shard_i) {
            const std::uint64_t row_begin = layout->shard_offsets[shard_i];
            const std::uint64_t row_end = layout->shard_offsets[shard_i + 1u];
            unsigned long part_end = part_begin;
            std::uint64_t shard_nnz = 0u;
            while (part_begin < layout->num_partitions && layout->partition_row_offsets[part_begin] < row_begin) ++part_begin;
            part_end = part_begin;
            while (part_end < layout->num_partitions && layout->partition_row_offsets[part_end + 1u] <= row_end) {
                shard_nnz += layout->partition_nnz[part_end];
                ++part_end;
            }
            shard_block_idx_offsets[shard_i] = part_begin < layout->num_partitions ? partition_block_idx_offsets[part_begin] : total_block_idx;
            shard_value_offsets[shard_i] = part_begin < layout->num_partitions ? partition_value_offsets[part_begin] : total_values;
            if (part_end == layout->num_partitions) {
                shard_block_idx_offsets[shard_i + 1u] = total_block_idx;
                shard_value_offsets[shard_i + 1u] = total_values;
            } else {
                shard_block_idx_offsets[shard_i + 1u] = partition_block_idx_offsets[part_end];
                shard_value_offsets[shard_i + 1u] = partition_value_offsets[part_end];
            }
            if (row_end - row_begin > dim_limit) warn_dataset_u32_limit(filename, "shard", shard_i, "rows", row_end - row_begin, dim_limit);
            if (shard_nnz > nnz_limit) warn_dataset_u32_limit(filename, "shard", shard_i, "nnz", shard_nnz, nnz_limit);
            if (shard_block_idx_offsets[shard_i + 1u] - shard_block_idx_offsets[shard_i] > idx_limit) {
                warn_dataset_u32_limit(filename,
                                      "shard",
                                      shard_i,
                                      "block_col_idx_count",
                                      shard_block_idx_offsets[shard_i + 1u] - shard_block_idx_offsets[shard_i],
                                      idx_limit);
            }
            if (shard_value_offsets[shard_i + 1u] - shard_value_offsets[shard_i] > nnz_limit) {
                warn_dataset_u32_limit(filename,
                                      "shard",
                                      shard_i,
                                      "value_count",
                                      shard_value_offsets[shard_i + 1u] - shard_value_offsets[shard_i],
                                      nnz_limit);
            }
            part_begin = part_end;
        }
    }

    if (!write_dataset_1d(payload, "partition_block_idx_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, partition_block_idx_offsets)) goto done;
    if (!write_dataset_1d(payload, "partition_value_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, partition_value_offsets)) goto done;
    if (!write_dataset_1d(payload, "shard_block_idx_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_shards + 1u, shard_block_idx_offsets)) goto done;
    if (!write_dataset_1d(payload, "shard_value_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_shards + 1u, shard_value_offsets)) goto done;
    if (!write_dataset_1d(payload, "block_col_idx", H5T_NATIVE_UINT32, (hsize_t) total_block_idx, 0)) goto done;
    if (!write_dataset_1d(payload, "values", H5T_NATIVE_UINT16, (hsize_t) total_values, 0)) goto done;

    ok = 1;

done:
    std::free(partition_aux);
    std::free(partition_axes);
    std::free(partition_block_idx_offsets);
    std::free(partition_value_offsets);
    std::free(shard_block_idx_offsets);
    std::free(shard_value_offsets);
    if (payload >= 0) H5Gclose(payload);
    if (payload_root >= 0) H5Gclose(payload_root);
    if (codecs >= 0) H5Gclose(codecs);
    if (prov >= 0) H5Gclose(prov);
    if (dsets >= 0) H5Gclose(dsets);
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int create_dataset_sliced_ell_h5(const char *filename,
                                 const dataset_layout_view *layout,
                                 const dataset_dataset_table_view *datasets,
                                 const dataset_provenance_view *provenance) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    hid_t dsets = (hid_t) -1;
    hid_t prov = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    hid_t payload_root = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    std::uint64_t *partition_aux = 0;
    std::uint32_t *partition_axes = 0;
    std::uint32_t i = 0;
    int ok = 0;
    const std::uint64_t dim_limit = local_dim_limit();
    const std::uint64_t nnz_limit = local_nnz_limit();
    const std::uint64_t idx_limit = local_index_limit();

    if (filename == 0 || layout == 0) return 0;
    if (layout->partition_rows == 0 || layout->partition_nnz == 0 || layout->partition_aux == 0 || layout->partition_row_offsets == 0 || layout->partition_dataset_ids == 0 || layout->partition_codec_ids == 0 || layout->shard_offsets == 0) return 0;
    if (layout->cols > idx_limit) {
        std::fprintf(stderr,
                     "cellshard: dataset column count exceeds the current u32 execution limit while writing %s (cols=%llu, limit=%llu)\n",
                     filename,
                     (unsigned long long) layout->cols,
                     (unsigned long long) idx_limit);
        return 0;
    }

    partition_aux = (std::uint64_t *) std::calloc((std::size_t) layout->num_partitions, sizeof(std::uint64_t));
    partition_axes = (std::uint32_t *) std::calloc((std::size_t) layout->num_partitions, sizeof(std::uint32_t));
    if (layout->num_partitions != 0u && (partition_aux == 0 || partition_axes == 0)) goto done;
    for (i = 0; i < layout->num_partitions; ++i) {
        const std::uint64_t total_slots = sparse::unpack_sliced_ell_total_slots((unsigned long) layout->partition_aux[i]);
        if (layout->partition_rows[i] > dim_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "rows", layout->partition_rows[i], dim_limit);
            goto done;
        }
        if (layout->partition_nnz[i] > nnz_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "nnz", layout->partition_nnz[i], nnz_limit);
            goto done;
        }
        if (total_slots > idx_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "slot_count", total_slots, idx_limit);
            goto done;
        }
        partition_aux[i] = layout->partition_aux[i];
        partition_axes[i] = layout->partition_axes != 0 ? layout->partition_axes[i] : 0u;
    }

    file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) goto done;
    if (!write_attr_string(file, "cellshard_magic", dataset_magic)) goto done;
    if (!write_attr_u32(file, "schema_version", dataset_h5_schema_version)) goto done;
    if (!write_attr_string(file, "matrix_format", "sliced_ell")) goto done;
    if (!write_attr_string(file, "payload_layout", payload_layout_shard_packed)) goto done;
    if (!write_attr_u64(file, "rows", layout->rows)) goto done;
    if (!write_attr_u64(file, "cols", layout->cols)) goto done;
    if (!write_attr_u64(file, "nnz", layout->nnz)) goto done;
    if (!write_attr_u64(file, "num_partitions", layout->num_partitions)) goto done;
    if (!write_attr_u64(file, "num_shards", layout->num_shards)) goto done;
    if (!write_attr_u64(file, "num_codecs", layout->num_codecs)) goto done;
    if (!write_attr_u64(file, "num_datasets", datasets != 0 ? datasets->count : 0u)) goto done;

    matrix = create_group(file, matrix_group);
    dsets = create_group(file, datasets_group);
    prov = create_group(file, provenance_group);
    codecs = create_group(file, codecs_group);
    payload_root = create_group(file, payload_group);
    payload = payload_root >= 0 ? create_group(payload_root, "sliced_ell") : (hid_t) -1;
    if (matrix < 0 || dsets < 0 || prov < 0 || codecs < 0 || payload_root < 0 || payload < 0) goto done;

    if (!write_dataset_1d(matrix, "partition_rows", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, layout->partition_rows)) goto done;
    if (!write_dataset_1d(matrix, "partition_nnz", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, layout->partition_nnz)) goto done;
    if (!write_dataset_1d(matrix, "partition_axes", H5T_NATIVE_UINT32, (hsize_t) layout->num_partitions, partition_axes)) goto done;
    if (!write_dataset_1d(matrix, "partition_aux", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, partition_aux)) goto done;
    if (!write_dataset_1d(matrix, "partition_row_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions + 1u, layout->partition_row_offsets)) goto done;
    if (!write_dataset_1d(matrix, "partition_dataset_ids", H5T_NATIVE_UINT32, (hsize_t) layout->num_partitions, layout->partition_dataset_ids)) goto done;
    if (!write_dataset_1d(matrix, "partition_codec_ids", H5T_NATIVE_UINT32, (hsize_t) layout->num_partitions, layout->partition_codec_ids)) goto done;
    if (!write_dataset_1d(matrix, "shard_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_shards + 1u, layout->shard_offsets)) goto done;

    if (datasets != 0) {
        if (!write_text_column(dsets, "dataset_ids", &datasets->dataset_ids)) goto done;
        if (!write_text_column(dsets, "matrix_paths", &datasets->matrix_paths)) goto done;
        if (!write_text_column(dsets, "feature_paths", &datasets->feature_paths)) goto done;
        if (!write_text_column(dsets, "barcode_paths", &datasets->barcode_paths)) goto done;
        if (!write_text_column(dsets, "metadata_paths", &datasets->metadata_paths)) goto done;
        if (!write_dataset_1d(dsets, "formats", H5T_NATIVE_UINT32, (hsize_t) datasets->count, datasets->formats)) goto done;
        if (!write_dataset_1d(dsets, "row_begin", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->row_begin)) goto done;
        if (!write_dataset_1d(dsets, "row_end", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->row_end)) goto done;
        if (!write_dataset_1d(dsets, "rows", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->rows)) goto done;
        if (!write_dataset_1d(dsets, "cols", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->cols)) goto done;
        if (!write_dataset_1d(dsets, "nnz", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->nnz)) goto done;
    }

    if (provenance != 0) {
        if (!write_text_column(prov, "global_barcodes", &provenance->global_barcodes)) goto done;
        if (!write_dataset_1d(prov, "cell_dataset_ids", H5T_NATIVE_UINT32, (hsize_t) layout->rows, provenance->cell_dataset_ids)) goto done;
        if (!write_dataset_1d(prov, "cell_local_indices", H5T_NATIVE_UINT64, (hsize_t) layout->rows, provenance->cell_local_indices)) goto done;
        if (!write_text_column(prov, "feature_ids", &provenance->feature_ids)) goto done;
        if (!write_text_column(prov, "feature_names", &provenance->feature_names)) goto done;
        if (!write_text_column(prov, "feature_types", &provenance->feature_types)) goto done;
        if (!write_dataset_1d(prov, "feature_dataset_ids", H5T_NATIVE_UINT32, (hsize_t) layout->cols, provenance->feature_dataset_ids)) goto done;
        if (!write_dataset_1d(prov, "feature_local_indices", H5T_NATIVE_UINT64, (hsize_t) layout->cols, provenance->feature_local_indices)) goto done;
        if (datasets != 0) {
            if (!write_dataset_1d(prov, "dataset_feature_offsets", H5T_NATIVE_UINT64, (hsize_t) datasets->count + 1u, provenance->dataset_feature_offsets)) goto done;
            if (!write_dataset_1d(prov, "dataset_feature_to_global", H5T_NATIVE_UINT32, (hsize_t) provenance->dataset_feature_offsets[datasets->count], provenance->dataset_feature_to_global)) goto done;
        }
    }

    if (layout->num_codecs != 0) {
        std::uint32_t *codec_id = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *family = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *value_code = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *scale_value_code = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *bits = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *flags = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        if (codec_id == 0 || family == 0 || value_code == 0 || scale_value_code == 0 || bits == 0 || flags == 0) {
            std::free(codec_id);
            std::free(family);
            std::free(value_code);
            std::free(scale_value_code);
            std::free(bits);
            std::free(flags);
            goto done;
        }
        for (i = 0; i < layout->num_codecs; ++i) {
            codec_id[i] = layout->codecs[i].codec_id;
            family[i] = layout->codecs[i].family;
            value_code[i] = layout->codecs[i].value_code;
            scale_value_code[i] = layout->codecs[i].scale_value_code;
            bits[i] = layout->codecs[i].bits;
            flags[i] = layout->codecs[i].flags;
        }
        if (!write_dataset_1d(codecs, "codec_id", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, codec_id)) goto done;
        if (!write_dataset_1d(codecs, "family", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, family)) goto done;
        if (!write_dataset_1d(codecs, "value_code", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, value_code)) goto done;
        if (!write_dataset_1d(codecs, "scale_value_code", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, scale_value_code)) goto done;
        if (!write_dataset_1d(codecs, "bits", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, bits)) goto done;
        if (!write_dataset_1d(codecs, "flags", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, flags)) goto done;
        std::free(codec_id);
        std::free(family);
        std::free(value_code);
        std::free(scale_value_code);
        std::free(bits);
        std::free(flags);
    }

    ok = 1;

done:
    std::free(partition_aux);
    std::free(partition_axes);
    if (payload >= 0) H5Gclose(payload);
    if (payload_root >= 0) H5Gclose(payload_root);
    if (codecs >= 0) H5Gclose(codecs);
    if (prov >= 0) H5Gclose(prov);
    if (dsets >= 0) H5Gclose(dsets);
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_dataset_embedded_metadata_h5(const char *filename,
                                       const dataset_embedded_metadata_view *metadata) {
    hid_t file = (hid_t) -1;
    hid_t root = (hid_t) -1;
    int ok = 0;

    if (filename == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    root = create_group(file, embedded_metadata_group);
    if (root < 0) goto done;

    if (!write_attr_u32(root, "count", metadata != 0 ? metadata->count : 0u)) goto done;
    if (metadata == 0 || metadata->count == 0u) {
        ok = 1;
        goto done;
    }

    if (!write_dataset_1d(root, "dataset_indices", H5T_NATIVE_UINT32, (hsize_t) metadata->count, metadata->dataset_indices)) goto done;
    if (!write_dataset_1d(root, "global_row_begin", H5T_NATIVE_UINT64, (hsize_t) metadata->count, metadata->global_row_begin)) goto done;
    if (!write_dataset_1d(root, "global_row_end", H5T_NATIVE_UINT64, (hsize_t) metadata->count, metadata->global_row_end)) goto done;

    for (std::uint32_t i = 0; i < metadata->count; ++i) {
        char name[64];
        hid_t table = (hid_t) -1;
        const dataset_metadata_table_view *view = metadata->tables + i;
        if (std::snprintf(name, sizeof(name), "table_%u", i) <= 0) goto done;
        table = create_group(root, name);
        if (table < 0) goto done;
        if (!write_attr_u32(table, "rows", view->rows) || !write_attr_u32(table, "cols", view->cols)) {
            H5Gclose(table);
            goto done;
        }
        if (!write_text_column(table, "column_names", &view->column_names)
            || !write_text_column(table, "field_values", &view->field_values)
            || !write_dataset_1d(table, "row_offsets", H5T_NATIVE_UINT32, (hsize_t) view->rows + 1u, view->row_offsets)) {
            H5Gclose(table);
            goto done;
        }
        H5Gclose(table);
    }

    ok = 1;

done:
    if (root >= 0) H5Gclose(root);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_dataset_observation_metadata_h5(const char *filename,
                                          const dataset_observation_metadata_view *metadata) {
    hid_t file = (hid_t) -1;
    hid_t root = (hid_t) -1;
    int ok = 0;

    if (filename == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    root = create_group(file, observation_metadata_group);
    if (root < 0) goto done;

    if (!write_attr_u64(root, "rows", metadata != 0 ? metadata->rows : 0u)) goto done;
    if (!write_attr_u32(root, "cols", metadata != 0 ? metadata->cols : 0u)) goto done;
    if (metadata == 0 || metadata->cols == 0u) {
        ok = 1;
        goto done;
    }

    for (std::uint32_t i = 0; i < metadata->cols; ++i) {
        char name[64];
        hid_t column = (hid_t) -1;
        const dataset_observation_metadata_column_view *view = metadata->columns + i;
        const hsize_t rows = (hsize_t) metadata->rows;

        if (view == 0 || view->name == 0) goto done;
        if (std::snprintf(name, sizeof(name), "column_%u", i) <= 0) goto done;
        column = create_group(root, name);
        if (column < 0) goto done;
        if (!write_attr_string(column, "name", view->name)
            || !write_attr_u32(column, "type", view->type)) {
            H5Gclose(column);
            goto done;
        }

        if (view->type == dataset_observation_metadata_type_text) {
            if (view->text_values.count != metadata->rows
                || !write_text_column(column, "values", &view->text_values)) {
                H5Gclose(column);
                goto done;
            }
        } else if (view->type == dataset_observation_metadata_type_float32) {
            if ((rows != 0u && view->float32_values == 0)
                || !write_dataset_1d(column, "values", H5T_NATIVE_FLOAT, rows, view->float32_values)) {
                H5Gclose(column);
                goto done;
            }
        } else if (view->type == dataset_observation_metadata_type_uint8) {
            if ((rows != 0u && view->uint8_values == 0)
                || !write_dataset_1d(column, "values", H5T_NATIVE_UINT8, rows, view->uint8_values)) {
                H5Gclose(column);
                goto done;
            }
        } else {
            H5Gclose(column);
            goto done;
        }

        H5Gclose(column);
    }

    ok = 1;

done:
    if (root >= 0) H5Gclose(root);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_dataset_browse_cache_h5(const char *filename,
                                  const dataset_browse_cache_view *browse) {
    hid_t file = (hid_t) -1;
    hid_t root = (hid_t) -1;
    int ok = 0;
    const hsize_t selected = (hsize_t) (browse != 0 ? browse->selected_feature_count : 0u);

    if (filename == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    root = create_group(file, browse_group);
    if (root < 0) goto done;

    if (!write_attr_u32(root, "selected_feature_count", browse != 0 ? browse->selected_feature_count : 0u)) goto done;
    if (!write_attr_u32(root, "dataset_count", browse != 0 ? browse->dataset_count : 0u)) goto done;
    if (!write_attr_u32(root, "shard_count", browse != 0 ? browse->shard_count : 0u)) goto done;
    if (!write_attr_u32(root, "partition_count", browse != 0 ? browse->partition_count : 0u)) goto done;
    if (!write_attr_u32(root, "sample_rows_per_partition", browse != 0 ? browse->sample_rows_per_partition : 0u)) goto done;

    if (browse == 0 || browse->selected_feature_count == 0u) {
        ok = 1;
        goto done;
    }

    if (!write_dataset_1d(root, "selected_feature_indices", H5T_NATIVE_UINT32, selected, browse->selected_feature_indices)) goto done;
    if (!write_dataset_1d(root, "gene_sum", H5T_NATIVE_FLOAT, selected, browse->gene_sum)) goto done;
    if (!write_dataset_1d(root, "gene_detected", H5T_NATIVE_FLOAT, selected, browse->gene_detected)) goto done;
    if (!write_dataset_1d(root, "gene_sq_sum", H5T_NATIVE_FLOAT, selected, browse->gene_sq_sum)) goto done;

    if (browse->dataset_count != 0u
        && !write_dataset_1d(root,
                             "dataset_feature_mean",
                             H5T_NATIVE_FLOAT,
                             (hsize_t) browse->dataset_count * selected,
                             browse->dataset_feature_mean)) goto done;

    if (browse->shard_count != 0u
        && !write_dataset_1d(root,
                             "shard_feature_mean",
                             H5T_NATIVE_FLOAT,
                             (hsize_t) browse->shard_count * selected,
                             browse->shard_feature_mean)) goto done;

    if (browse->partition_count != 0u) {
        const hsize_t row_count = (hsize_t) browse->partition_count * (hsize_t) browse->sample_rows_per_partition;
        const hsize_t value_count = row_count * selected;
        if (!write_dataset_1d(root,
                              "partition_sample_row_offsets",
                              H5T_NATIVE_UINT32,
                              (hsize_t) browse->partition_count + 1u,
                              browse->partition_sample_row_offsets)) goto done;
        if (!write_dataset_1d(root,
                              "partition_sample_global_rows",
                              H5T_NATIVE_UINT64,
                              row_count,
                              browse->partition_sample_global_rows)) goto done;
        if (!write_dataset_1d(root,
                              "partition_sample_values",
                              H5T_NATIVE_FLOAT,
                              value_count,
                              browse->partition_sample_values)) goto done;
    }

    ok = 1;

done:
    if (root >= 0) H5Gclose(root);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_dataset_preprocess_h5(const char *filename,
                                const dataset_preprocess_view *preprocess) {
    hid_t file = (hid_t) -1;
    hid_t root = (hid_t) -1;
    hid_t cell_qc = (hid_t) -1;
    hid_t gene_qc = (hid_t) -1;
    int ok = 0;
    const char *assay = (preprocess != 0 && preprocess->assay != 0) ? preprocess->assay : "";
    const char *matrix_orientation = (preprocess != 0 && preprocess->matrix_orientation != 0) ? preprocess->matrix_orientation : "";
    const char *matrix_state = (preprocess != 0 && preprocess->matrix_state != 0) ? preprocess->matrix_state : "";
    const char *pipeline_scope = (preprocess != 0 && preprocess->pipeline_scope != 0) ? preprocess->pipeline_scope : "";
    const char *raw_matrix_name = (preprocess != 0 && preprocess->raw_matrix_name != 0) ? preprocess->raw_matrix_name : "";
    const char *active_matrix_name = (preprocess != 0 && preprocess->active_matrix_name != 0) ? preprocess->active_matrix_name : "";
    const char *feature_namespace = (preprocess != 0 && preprocess->feature_namespace != 0) ? preprocess->feature_namespace : "";
    const char *mito_prefix = (preprocess != 0 && preprocess->mito_prefix != 0) ? preprocess->mito_prefix : "";
    const hsize_t rows = (hsize_t) (preprocess != 0 ? preprocess->rows : 0u);
    const hsize_t cols = (hsize_t) (preprocess != 0 ? preprocess->cols : 0u);

    if (filename == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    root = create_group(file, preprocess_group);
    if (root < 0) goto done;

    if (!write_attr_string(root, "assay", assay)
        || !write_attr_string(root, "matrix_orientation", matrix_orientation)
        || !write_attr_string(root, "matrix_state", matrix_state)
        || !write_attr_string(root, "pipeline_scope", pipeline_scope)
        || !write_attr_string(root, "raw_matrix_name", raw_matrix_name)
        || !write_attr_string(root, "active_matrix_name", active_matrix_name)
        || !write_attr_string(root, "feature_namespace", feature_namespace)
        || !write_attr_string(root, "mito_prefix", mito_prefix)
        || !write_attr_u32(root, "raw_counts_available", preprocess != 0 ? preprocess->raw_counts_available : 0u)
        || !write_attr_u32(root, "processed_matrix_available", preprocess != 0 ? preprocess->processed_matrix_available : 0u)
        || !write_attr_u32(root, "normalized_log1p_metrics", preprocess != 0 ? preprocess->normalized_log1p_metrics : 0u)
        || !write_attr_u32(root, "hvg_available", preprocess != 0 ? preprocess->hvg_available : 0u)
        || !write_attr_u32(root, "mark_mito_from_feature_names", preprocess != 0 ? preprocess->mark_mito_from_feature_names : 0u)
        || !write_attr_u64(root, "rows", preprocess != 0 ? preprocess->rows : 0u)
        || !write_attr_u32(root, "cols", preprocess != 0 ? preprocess->cols : 0u)
        || !write_attr_u64(root, "nnz", preprocess != 0 ? preprocess->nnz : 0u)
        || !write_attr_u32(root, "partitions_processed", preprocess != 0 ? preprocess->partitions_processed : 0u)
        || !write_attr_u32(root, "mito_feature_count", preprocess != 0 ? preprocess->mito_feature_count : 0u)
        || !write_attr_f32(root, "target_sum", preprocess != 0 ? preprocess->target_sum : 0.0f)
        || !write_attr_f32(root, "min_counts", preprocess != 0 ? preprocess->min_counts : 0.0f)
        || !write_attr_u32(root, "min_genes", preprocess != 0 ? preprocess->min_genes : 0u)
        || !write_attr_f32(root, "max_mito_fraction", preprocess != 0 ? preprocess->max_mito_fraction : 0.0f)
        || !write_attr_f32(root, "min_gene_sum", preprocess != 0 ? preprocess->min_gene_sum : 0.0f)
        || !write_attr_f32(root, "min_detected_cells", preprocess != 0 ? preprocess->min_detected_cells : 0.0f)
        || !write_attr_f32(root, "min_variance", preprocess != 0 ? preprocess->min_variance : 0.0f)
        || !write_attr_f64(root, "kept_cells", preprocess != 0 ? preprocess->kept_cells : 0.0)
        || !write_attr_u32(root, "kept_genes", preprocess != 0 ? preprocess->kept_genes : 0u)
        || !write_attr_f64(root, "gene_sum_checksum", preprocess != 0 ? preprocess->gene_sum_checksum : 0.0)) {
        goto done;
    }

    cell_qc = create_group(root, "cell_qc");
    gene_qc = create_group(root, "gene_qc");
    if (cell_qc < 0 || gene_qc < 0) goto done;

    if (!write_attr_u64(cell_qc, "rows", preprocess != 0 ? preprocess->rows : 0u)
        || !write_attr_u32(gene_qc, "cols", preprocess != 0 ? preprocess->cols : 0u)) {
        goto done;
    }

    if (rows != 0u) {
        if ((preprocess == 0)
            || !write_dataset_1d(cell_qc, "total_counts", H5T_NATIVE_FLOAT, rows, preprocess->cell_total_counts)
            || !write_dataset_1d(cell_qc, "mito_counts", H5T_NATIVE_FLOAT, rows, preprocess->cell_mito_counts)
            || !write_dataset_1d(cell_qc, "max_counts", H5T_NATIVE_FLOAT, rows, preprocess->cell_max_counts)
            || !write_dataset_1d(cell_qc, "detected_genes", H5T_NATIVE_UINT32, rows, preprocess->cell_detected_genes)
            || !write_dataset_1d(cell_qc, "keep", H5T_NATIVE_UINT8, rows, preprocess->cell_keep)) {
            goto done;
        }
    }

    if (cols != 0u) {
        if ((preprocess == 0)
            || !write_dataset_1d(gene_qc, "sum", H5T_NATIVE_FLOAT, cols, preprocess->gene_sum)
            || !write_dataset_1d(gene_qc, "sq_sum", H5T_NATIVE_FLOAT, cols, preprocess->gene_sq_sum)
            || !write_dataset_1d(gene_qc, "detected_cells", H5T_NATIVE_FLOAT, cols, preprocess->gene_detected_cells)
            || !write_dataset_1d(gene_qc, "keep", H5T_NATIVE_UINT8, cols, preprocess->gene_keep)
            || !write_dataset_1d(gene_qc, "flags", H5T_NATIVE_UINT8, cols, preprocess->gene_flags)) {
            goto done;
        }
    }

    ok = 1;

done:
    if (gene_qc >= 0) H5Gclose(gene_qc);
    if (cell_qc >= 0) H5Gclose(cell_qc);
    if (root >= 0) H5Gclose(root);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_dataset_execution_h5(const char *filename,
                               const dataset_execution_view *execution) {
    hid_t file = (hid_t) -1;
    hid_t root = (hid_t) -1;
    int ok = 0;

    if (filename == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    root = create_group(file, execution_group);
    if (root < 0) goto done;

    if (!write_attr_u32(root, "partition_count", execution != 0 ? execution->partition_count : 0u)) goto done;
    if (!write_attr_u32(root, "shard_count", execution != 0 ? execution->shard_count : 0u)) goto done;
    if (!write_attr_u32(root, "preferred_base_format", execution != 0 ? execution->preferred_base_format : dataset_execution_format_unknown)) goto done;

    if (execution == 0) {
        ok = 1;
        goto done;
    }

    if (execution->partition_count != 0u) {
        if (!write_dataset_1d(root,
                              "partition_execution_formats",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->partition_count,
                              execution->partition_execution_formats)) goto done;
        if (!write_dataset_1d(root,
                              "partition_blocked_ell_block_sizes",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->partition_count,
                              execution->partition_blocked_ell_block_sizes)) goto done;
        if (!write_dataset_1d(root,
                              "partition_blocked_ell_bucket_counts",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->partition_count,
                              execution->partition_blocked_ell_bucket_counts)) goto done;
        if (!write_dataset_1d(root,
                              "partition_blocked_ell_fill_ratios",
                              H5T_NATIVE_FLOAT,
                              (hsize_t) execution->partition_count,
                              execution->partition_blocked_ell_fill_ratios)) goto done;
        if (!write_dataset_1d(root,
                              "partition_execution_bytes",
                              H5T_NATIVE_UINT64,
                              (hsize_t) execution->partition_count,
                              execution->partition_execution_bytes)) goto done;
        if (!write_dataset_1d(root,
                              "partition_blocked_ell_bytes",
                              H5T_NATIVE_UINT64,
                              (hsize_t) execution->partition_count,
                              execution->partition_blocked_ell_bytes)) goto done;
        if (!write_dataset_1d(root,
                              "partition_bucketed_blocked_ell_bytes",
                              H5T_NATIVE_UINT64,
                              (hsize_t) execution->partition_count,
                              execution->partition_bucketed_blocked_ell_bytes)) goto done;
        if (execution->partition_sliced_ell_slice_counts != 0
            && !write_dataset_1d(root,
                                 "partition_sliced_ell_slice_counts",
                                 H5T_NATIVE_UINT32,
                                 (hsize_t) execution->partition_count,
                                 execution->partition_sliced_ell_slice_counts)) goto done;
        if (execution->partition_sliced_ell_bytes != 0
            && !write_dataset_1d(root,
                                 "partition_sliced_ell_bytes",
                                 H5T_NATIVE_UINT64,
                                 (hsize_t) execution->partition_count,
                                 execution->partition_sliced_ell_bytes)) goto done;
        if (execution->partition_bucketed_sliced_ell_bytes != 0
            && !write_dataset_1d(root,
                                 "partition_bucketed_sliced_ell_bytes",
                                 H5T_NATIVE_UINT64,
                                 (hsize_t) execution->partition_count,
                                 execution->partition_bucketed_sliced_ell_bytes)) goto done;
    }

    if (execution->shard_count != 0u) {
        if (!write_dataset_1d(root,
                              "shard_execution_formats",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->shard_count,
                              execution->shard_execution_formats)) goto done;
        if (!write_dataset_1d(root,
                              "shard_blocked_ell_block_sizes",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->shard_count,
                              execution->shard_blocked_ell_block_sizes)) goto done;
        if (!write_dataset_1d(root,
                              "shard_bucketed_partition_counts",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->shard_count,
                              execution->shard_bucketed_partition_counts)) goto done;
        if (!write_dataset_1d(root,
                              "shard_bucketed_segment_counts",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->shard_count,
                              execution->shard_bucketed_segment_counts)) goto done;
        if (!write_dataset_1d(root,
                              "shard_blocked_ell_fill_ratios",
                              H5T_NATIVE_FLOAT,
                              (hsize_t) execution->shard_count,
                              execution->shard_blocked_ell_fill_ratios)) goto done;
        if (!write_dataset_1d(root,
                              "shard_execution_bytes",
                              H5T_NATIVE_UINT64,
                              (hsize_t) execution->shard_count,
                              execution->shard_execution_bytes)) goto done;
        if (!write_dataset_1d(root,
                              "shard_bucketed_blocked_ell_bytes",
                              H5T_NATIVE_UINT64,
                              (hsize_t) execution->shard_count,
                              execution->shard_bucketed_blocked_ell_bytes)) goto done;
        if (execution->shard_sliced_ell_slice_counts != 0
            && !write_dataset_1d(root,
                                 "shard_sliced_ell_slice_counts",
                                 H5T_NATIVE_UINT32,
                                 (hsize_t) execution->shard_count,
                                 execution->shard_sliced_ell_slice_counts)) goto done;
        if (execution->shard_bucketed_sliced_ell_bytes != 0
            && !write_dataset_1d(root,
                                 "shard_bucketed_sliced_ell_bytes",
                                 H5T_NATIVE_UINT64,
                                 (hsize_t) execution->shard_count,
                                 execution->shard_bucketed_sliced_ell_bytes)) goto done;
        if (!write_dataset_1d(root,
                              "shard_preferred_pair_ids",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->shard_count,
                              execution->shard_preferred_pair_ids)) goto done;
        if (execution->shard_owner_node_ids != 0
            && !write_dataset_1d(root,
                                 "shard_owner_node_ids",
                                 H5T_NATIVE_UINT32,
                                 (hsize_t) execution->shard_count,
                                 execution->shard_owner_node_ids)) goto done;
        if (execution->shard_owner_rank_ids != 0
            && !write_dataset_1d(root,
                                 "shard_owner_rank_ids",
                                 H5T_NATIVE_UINT32,
                                 (hsize_t) execution->shard_count,
                                 execution->shard_owner_rank_ids)) goto done;
    }

    ok = 1;

done:
    if (root >= 0) H5Gclose(root);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_dataset_runtime_service_h5(const char *filename,
                                     const dataset_runtime_service_view *runtime_service) {
    hid_t file = (hid_t) -1;
    hid_t root = (hid_t) -1;
    dataset_runtime_service_view defaults;
    const dataset_runtime_service_view *view = runtime_service;
    int ok = 0;

    if (filename == 0) return 0;
    init(&defaults);
    if (view == 0) {
        defaults.service_mode = dataset_runtime_service_mode_local_cache;
        defaults.live_write_mode = dataset_live_write_mode_read_only;
        defaults.prefer_pack_delivery = 1u;
        defaults.maintenance_lock_blocks_overwrite = 1u;
        defaults.canonical_generation = 1u;
        defaults.execution_plan_generation = 1u;
        defaults.pack_generation = 1u;
        defaults.service_epoch = 1u;
        defaults.active_read_generation = 1u;
        defaults.staged_write_generation = 1u;
        view = &defaults;
    }

    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    root = create_group(file, runtime_service_group);
    if (root < 0) goto done;

    if (!write_attr_u32(root, "service_mode", view->service_mode)
        || !write_attr_u32(root, "live_write_mode", view->live_write_mode)
        || !write_attr_u32(root, "prefer_pack_delivery", view->prefer_pack_delivery)
        || !write_attr_u32(root, "remote_pack_delivery", view->remote_pack_delivery)
        || !write_attr_u32(root, "single_reader_coordinator", view->single_reader_coordinator)
        || !write_attr_u32(root, "maintenance_lock_blocks_overwrite", view->maintenance_lock_blocks_overwrite)
        || !write_attr_u64(root, "canonical_generation", view->canonical_generation)
        || !write_attr_u64(root, "execution_plan_generation", view->execution_plan_generation)
        || !write_attr_u64(root, "pack_generation", view->pack_generation)
        || !write_attr_u64(root, "service_epoch", view->service_epoch)
        || !write_attr_u64(root, "active_read_generation", view->active_read_generation)
        || !write_attr_u64(root, "staged_write_generation", view->staged_write_generation)) {
        goto done;
    }

    ok = 1;

done:
    if (root >= 0) H5Gclose(root);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_standard_csr_partition_h5(const char *filename,
                                unsigned long partition_id,
                                const sparse::compressed *part) {
    hid_t file = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    hid_t d_indptr = (hid_t) -1;
    hid_t d_indices = (hid_t) -1;
    hid_t d_values = (hid_t) -1;
    std::uint64_t *partition_indptr_offsets = 0;
    std::uint64_t *partition_nnz_offsets = 0;
    std::uint64_t num_partitions = 0;
    int ok = 0;

    if (filename == 0 || part == 0 || part->axis != sparse::compressed_by_row) return 0;

    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    if (!read_attr_u64(file, "num_partitions", &num_partitions)) goto done;
    if (partition_id >= num_partitions) goto done;

    partition_indptr_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    partition_nnz_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    if ((num_partitions != 0) && (partition_indptr_offsets == 0 || partition_nnz_offsets == 0)) goto done;

    payload = H5Gopen2(file, payload_standard_group, H5P_DEFAULT);
    if (payload < 0) goto done;
    if (!read_dataset_1d(payload, "partition_indptr_offsets", H5T_NATIVE_UINT64, partition_indptr_offsets)) goto done;
    if (!read_dataset_1d(payload, "partition_nnz_offsets", H5T_NATIVE_UINT64, partition_nnz_offsets)) goto done;
    d_indptr = H5Dopen2(payload, "indptr", H5P_DEFAULT);
    d_indices = H5Dopen2(payload, "indices", H5P_DEFAULT);
    d_values = H5Dopen2(payload, "values", H5P_DEFAULT);
    if (d_indptr < 0 || d_indices < 0 || d_values < 0) goto done;
    if (!read_hyperslab_1d(d_indptr, H5T_NATIVE_UINT32, 0, 0, 0)) goto done;

    {
        hsize_t off[1];
        hsize_t dims[1];
        hid_t filespace = (hid_t) -1;
        hid_t memspace = (hid_t) -1;

        off[0] = (hsize_t) partition_indptr_offsets[partition_id];
        dims[0] = (hsize_t) part->rows + 1u;
        filespace = H5Dget_space(d_indptr);
        if (filespace < 0) goto done;
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, off, 0, dims, 0) < 0) {
            H5Sclose(filespace);
            goto done;
        }
        memspace = H5Screate_simple(1, dims, 0);
        if (memspace < 0) {
            H5Sclose(filespace);
            goto done;
        }
        if (H5Dwrite(d_indptr, H5T_NATIVE_UINT32, memspace, filespace, H5P_DEFAULT, part->majorPtr) < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            goto done;
        }
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    {
        hsize_t off[1];
        hsize_t dims[1];
        hid_t filespace = (hid_t) -1;
        hid_t memspace = (hid_t) -1;

        off[0] = (hsize_t) partition_nnz_offsets[partition_id];
        dims[0] = (hsize_t) part->nnz;
        filespace = H5Dget_space(d_indices);
        if (filespace < 0) goto done;
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, off, 0, dims, 0) < 0) {
            H5Sclose(filespace);
            goto done;
        }
        memspace = H5Screate_simple(1, dims, 0);
        if (memspace < 0) {
            H5Sclose(filespace);
            goto done;
        }
        if (H5Dwrite(d_indices, H5T_NATIVE_UINT32, memspace, filespace, H5P_DEFAULT, part->minorIdx) < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            goto done;
        }
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    {
        hsize_t off[1];
        hsize_t dims[1];
        hid_t filespace = (hid_t) -1;
        hid_t memspace = (hid_t) -1;

        off[0] = (hsize_t) partition_nnz_offsets[partition_id];
        dims[0] = (hsize_t) part->nnz;
        filespace = H5Dget_space(d_values);
        if (filespace < 0) goto done;
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, off, 0, dims, 0) < 0) {
            H5Sclose(filespace);
            goto done;
        }
        memspace = H5Screate_simple(1, dims, 0);
        if (memspace < 0) {
            H5Sclose(filespace);
            goto done;
        }
        if (H5Dwrite(d_values, H5T_NATIVE_UINT16, memspace, filespace, H5P_DEFAULT, part->val) < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            goto done;
        }
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    ok = 1;

done:
    std::free(partition_indptr_offsets);
    std::free(partition_nnz_offsets);
    if (d_values >= 0) H5Dclose(d_values);
    if (d_indices >= 0) H5Dclose(d_indices);
    if (d_indptr >= 0) H5Dclose(d_indptr);
    if (payload >= 0) H5Gclose(payload);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_blocked_ell_partition_h5(const char *filename,
                               unsigned long partition_id,
                               const sparse::blocked_ell *part) {
    hid_t file = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    hid_t d_block_idx = (hid_t) -1;
    hid_t d_values = (hid_t) -1;
    std::uint64_t *partition_block_idx_offsets = 0;
    std::uint64_t *partition_value_offsets = 0;
    std::uint64_t num_partitions = 0;
    const std::size_t row_blocks = sparse::row_block_count(part);
    const std::size_t ell_width = sparse::ell_width_blocks(part);
    int ok = 0;

    if (filename == 0 || part == 0) return 0;

    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    if (!read_attr_u64(file, "num_partitions", &num_partitions)) goto done;
    if (partition_id >= num_partitions) goto done;

    partition_block_idx_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    partition_value_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    if ((num_partitions != 0) && (partition_block_idx_offsets == 0 || partition_value_offsets == 0)) goto done;

    payload = H5Gopen2(file, payload_blocked_ell_group, H5P_DEFAULT);
    if (payload < 0) goto done;
    if (!read_dataset_1d(payload, "partition_block_idx_offsets", H5T_NATIVE_UINT64, partition_block_idx_offsets)) goto done;
    if (!read_dataset_1d(payload, "partition_value_offsets", H5T_NATIVE_UINT64, partition_value_offsets)) goto done;
    d_block_idx = H5Dopen2(payload, "block_col_idx", H5P_DEFAULT);
    d_values = H5Dopen2(payload, "values", H5P_DEFAULT);
    if (d_block_idx < 0 || d_values < 0) goto done;

    {
        hsize_t off[1];
        hsize_t dims[1];
        hid_t filespace = (hid_t) -1;
        hid_t memspace = (hid_t) -1;

        off[0] = (hsize_t) partition_block_idx_offsets[partition_id];
        dims[0] = (hsize_t) (row_blocks * ell_width);
        filespace = H5Dget_space(d_block_idx);
        if (filespace < 0) goto done;
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, off, 0, dims, 0) < 0) {
            H5Sclose(filespace);
            goto done;
        }
        memspace = H5Screate_simple(1, dims, 0);
        if (memspace < 0) {
            H5Sclose(filespace);
            goto done;
        }
        if (H5Dwrite(d_block_idx, H5T_NATIVE_UINT32, memspace, filespace, H5P_DEFAULT, part->blockColIdx) < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            goto done;
        }
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    {
        hsize_t off[1];
        hsize_t dims[1];
        hid_t filespace = (hid_t) -1;
        hid_t memspace = (hid_t) -1;

        off[0] = (hsize_t) partition_value_offsets[partition_id];
        dims[0] = (hsize_t) ((std::size_t) part->rows * (std::size_t) part->ell_cols);
        filespace = H5Dget_space(d_values);
        if (filespace < 0) goto done;
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, off, 0, dims, 0) < 0) {
            H5Sclose(filespace);
            goto done;
        }
        memspace = H5Screate_simple(1, dims, 0);
        if (memspace < 0) {
            H5Sclose(filespace);
            goto done;
        }
        if (H5Dwrite(d_values, H5T_NATIVE_UINT16, memspace, filespace, H5P_DEFAULT, part->val) < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            goto done;
        }
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    ok = 1;

done:
    std::free(partition_block_idx_offsets);
    std::free(partition_value_offsets);
    if (d_values >= 0) H5Dclose(d_values);
    if (d_block_idx >= 0) H5Dclose(d_block_idx);
    if (payload >= 0) H5Gclose(payload);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_sliced_ell_partition_h5(const char *filename,
                                   unsigned long partition_id,
                                   const sparse::sliced_ell *part) {
    hid_t file = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    char *buffer = 0;
    unsigned char *blob = 0;
    std::size_t blob_bytes = 0u;
    char dataset_name[64];
    std::FILE *fp = 0;
    int ok = 0;

    if (filename == 0 || part == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    payload = H5Gopen2(file, payload_sliced_ell_group, H5P_DEFAULT);
    if (payload < 0) goto done;
    if (!build_partition_blob_dataset_name(partition_id, dataset_name, sizeof(dataset_name))) goto done;
    fp = open_memstream(&buffer, &blob_bytes);
    if (fp == 0) goto done;
    if (!::cellshard::store(fp, part) || std::fclose(fp) != 0) {
        fp = 0;
        goto done;
    }
    fp = 0;
    blob = (unsigned char *) buffer;
    if (!write_blob_dataset(payload, dataset_name, blob, blob_bytes)) goto done;
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    std::free(buffer);
    if (payload >= 0) H5Gclose(payload);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_bucketed_blocked_ell_shard_h5(const char *filename,
                                         unsigned long shard_id,
                                         const bucketed_blocked_ell_shard *shard) {
    hid_t file = (hid_t) -1;
    hid_t payload_root = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    unsigned char *blob = 0;
    std::size_t blob_bytes = 0u;
    char dataset_name[64];
    int ok = 0;

    if (filename == 0 || shard == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    payload_root = H5Gopen2(file, payload_group, H5P_DEFAULT);
    if (payload_root < 0) payload_root = create_group(file, payload_group);
    if (payload_root < 0) goto done;
    if (H5Lexists(payload_root, "optimized_blocked_ell", H5P_DEFAULT) > 0) {
        payload = H5Gopen2(payload_root, "optimized_blocked_ell", H5P_DEFAULT);
    } else {
        payload = create_group(payload_root, "optimized_blocked_ell");
    }
    if (payload < 0) goto done;
    if (!build_optimized_shard_dataset_name(shard_id, dataset_name, sizeof(dataset_name))) goto done;
    if (!serialize_optimized_shard(shard, &blob, &blob_bytes)) goto done;
    if (!write_blob_dataset(payload, dataset_name, blob, blob_bytes)) goto done;
    if (H5Aexists(file, "payload_layout") > 0 && H5Adelete(file, "payload_layout") < 0) goto done;
    if (!write_attr_string(file, "payload_layout", payload_layout_optimized_blocked_ell)) goto done;
    ok = 1;

done:
    std::free(blob);
    if (payload >= 0) H5Gclose(payload);
    if (payload_root >= 0) H5Gclose(payload_root);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_bucketed_sliced_ell_shard_h5(const char *filename,
                                        unsigned long shard_id,
                                        const bucketed_sliced_ell_shard *shard) {
    hid_t file = (hid_t) -1;
    hid_t payload_root = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    unsigned char *blob = 0;
    std::size_t blob_bytes = 0u;
    char dataset_name[64];
    int ok = 0;

    if (filename == 0 || shard == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    payload_root = H5Gopen2(file, payload_group, H5P_DEFAULT);
    if (payload_root < 0) payload_root = create_group(file, payload_group);
    if (payload_root < 0) goto done;
    if (H5Lexists(payload_root, "optimized_sliced_ell", H5P_DEFAULT) > 0) {
        payload = H5Gopen2(payload_root, "optimized_sliced_ell", H5P_DEFAULT);
    } else {
        payload = create_group(payload_root, "optimized_sliced_ell");
    }
    if (payload < 0) goto done;
    if (!build_optimized_shard_dataset_name(shard_id, dataset_name, sizeof(dataset_name))) goto done;
    if (!serialize_optimized_sliced_shard(shard, &blob, &blob_bytes)) goto done;
    if (!write_blob_dataset(payload, dataset_name, blob, blob_bytes)) goto done;
    ok = 1;

done:
    std::free(blob);
    if (payload >= 0) H5Gclose(payload);
    if (payload_root >= 0) H5Gclose(payload_root);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int bind_dataset_h5(shard_storage *s, const char *path) {
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
    return 1;
}

int bind_dataset_h5_cache(shard_storage *s, const char *cache_root) {
    dataset_h5_state *state = 0;

    if (s == 0 || s->backend != shard_storage_backend_dataset_h5 || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
    if (state->cache_root != 0 && cache_root != 0 && std::strcmp(state->cache_root, cache_root) != 0) {
        invalidate_dataset_h5_cache(s);
        std::free(state->cache_instance_dir);
        std::free(state->cache_manifest_path);
        state->cache_instance_dir = 0;
        state->cache_manifest_path = 0;
    }
    if (cache_root == 0 || *cache_root == 0) return 1;
    if (!ensure_directory_exists(cache_root)) return 0;
    if (!assign_owned_string(&state->cache_root, cache_root)) return 0;
    return 1;
}

int get_dataset_h5_execution_metadata(const shard_storage *s,
                                     dataset_execution_view *execution) {
    const dataset_h5_state *state = 0;
    if (execution == 0) return 0;
    std::memset(execution, 0, sizeof(*execution));
    if (s == 0 || s->backend != shard_storage_backend_dataset_h5 || s->backend_state == 0) return 0;
    state = (const dataset_h5_state *) s->backend_state;
    execution->partition_count = (std::uint32_t) state->num_partitions;
    execution->partition_execution_formats = state->partition_execution_formats;
    execution->partition_blocked_ell_block_sizes = state->partition_blocked_ell_block_sizes;
    execution->partition_blocked_ell_bucket_counts = state->partition_blocked_ell_bucket_counts;
    execution->partition_blocked_ell_fill_ratios = state->partition_blocked_ell_fill_ratios;
    execution->partition_execution_bytes = state->partition_execution_bytes;
    execution->partition_blocked_ell_bytes = state->partition_blocked_ell_bytes;
    execution->partition_bucketed_blocked_ell_bytes = state->partition_bucketed_blocked_ell_bytes;
    execution->partition_sliced_ell_slice_counts = state->partition_sliced_ell_slice_counts;
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
    execution->shard_bucketed_sliced_ell_bytes = state->shard_bucketed_sliced_ell_bytes;
    execution->shard_preferred_pair_ids = state->shard_preferred_pair_ids;
    execution->shard_owner_node_ids = state->shard_owner_node_ids;
    execution->shard_owner_rank_ids = state->shard_owner_rank_ids;
    execution->preferred_base_format = state->preferred_base_format;
    return 1;
}

int get_dataset_h5_runtime_service(const shard_storage *s,
                                  dataset_runtime_service_view *runtime_service) {
    const dataset_h5_state *state = 0;
    if (runtime_service == 0) return 0;
    init(runtime_service);
    if (s == 0 || s->backend != shard_storage_backend_dataset_h5 || s->backend_state == 0) return 0;
    state = (const dataset_h5_state *) s->backend_state;
    *runtime_service = state->runtime_service;
    return 1;
}

int set_dataset_h5_cache_budget_bytes(shard_storage *s, std::uint64_t bytes) {
    dataset_h5_state *state = 0;
    dataset_h5_cache_runtime *runtime = 0;
    if (s == 0 || s->backend != shard_storage_backend_dataset_h5 || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
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
    dataset_h5_state *state = 0;
    if (s == 0 || s->backend != shard_storage_backend_dataset_h5 || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
    state->predictor_enabled = enabled != 0 ? 1 : 0;
    return 1;
}

int pin_dataset_h5_cache_shard(shard_storage *s, unsigned long shard_id) {
    dataset_h5_state *state = 0;
    dataset_h5_cache_runtime *runtime = 0;
    if (s == 0 || s->backend != shard_storage_backend_dataset_h5 || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
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
    dataset_h5_state *state = 0;
    dataset_h5_cache_runtime *runtime = 0;
    if (s == 0 || s->backend != shard_storage_backend_dataset_h5 || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
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
    dataset_h5_state *state = 0;
    dataset_h5_cache_runtime *runtime = 0;
    if (s == 0 || s->backend != shard_storage_backend_dataset_h5 || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
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
    dataset_h5_state *state = 0;
    dataset_h5_cache_runtime *runtime = 0;
    unsigned long shard_id = 0ul;
    if (s == 0 || s->backend != shard_storage_backend_dataset_h5 || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
    if (!ensure_dataset_cache_layout(s)) return 0;
    runtime = cache_runtime(state);
    if (runtime == 0) return 0;
    {
        std::lock_guard<std::mutex> lock(runtime->state_mutex);
        for (shard_id = 0ul; shard_id < (unsigned long) state->num_shards; ++shard_id) {
            evict_cached_shard_locked(state, shard_id);
        }
        state->access_clock = 0u;
        state->last_requested_shard = std::numeric_limits<std::uint64_t>::max();
    }
    if (state->cache_manifest_path != 0) ::unlink(state->cache_manifest_path);
    return 1;
}

int load_dataset_compressed_h5_header(const char *filename,
                                     sharded<sparse::compressed> *m,
                                     shard_storage *s) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t nnz = 0;
    std::uint64_t num_partitions = 0;
    std::uint64_t num_shards = 0;
    std::uint64_t num_codecs = 0;
    std::uint64_t *partition_rows = 0;
    std::uint64_t *partition_nnz = 0;
    std::uint32_t *partition_axes = 0;
    std::uint64_t *partition_row_offsets = 0;
    std::uint64_t *shard_offsets = 0;
    unsigned long *part_rows_ul = 0;
    unsigned long *part_nnz_ul = 0;
    unsigned long *part_axes_ul = 0;
    unsigned long *shard_offsets_ul = 0;
    unsigned long i = 0;
    int ok = 0;

    if (filename == 0 || m == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    if (!read_attr_u64(file, "rows", &rows)) goto done;
    if (!read_attr_u64(file, "cols", &cols)) goto done;
    if (!read_attr_u64(file, "nnz", &nnz)) goto done;
    if (!read_attr_u64(file, "num_partitions", &num_partitions)) goto done;
    if (!read_attr_u64(file, "num_shards", &num_shards)) goto done;
    if (!read_attr_u64(file, "num_codecs", &num_codecs)) goto done;

    matrix = H5Gopen2(file, matrix_group, H5P_DEFAULT);
    codecs = H5Gopen2(file, codecs_group, H5P_DEFAULT);
    if (matrix < 0 || codecs < 0) goto done;

    partition_rows = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    partition_nnz = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    partition_axes = (std::uint32_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint32_t));
    partition_row_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions + 1u, sizeof(std::uint64_t));
    shard_offsets = (std::uint64_t *) std::calloc((std::size_t) num_shards + 1u, sizeof(std::uint64_t));
    part_rows_ul = (unsigned long *) std::calloc((std::size_t) num_partitions, sizeof(unsigned long));
    part_nnz_ul = (unsigned long *) std::calloc((std::size_t) num_partitions, sizeof(unsigned long));
    part_axes_ul = (unsigned long *) std::calloc((std::size_t) num_partitions, sizeof(unsigned long));
    shard_offsets_ul = (unsigned long *) std::calloc((std::size_t) num_shards + 1u, sizeof(unsigned long));
    if ((num_partitions != 0) && (partition_rows == 0 || partition_nnz == 0 || partition_axes == 0 || partition_row_offsets == 0 || part_rows_ul == 0 || part_nnz_ul == 0 || part_axes_ul == 0)) goto done;
    if ((num_shards + 1u) != 0u && (shard_offsets == 0 || shard_offsets_ul == 0)) goto done;

    if (!read_dataset_1d(matrix, "partition_rows", H5T_NATIVE_UINT64, partition_rows)) goto done;
    if (!read_dataset_1d(matrix, "partition_nnz", H5T_NATIVE_UINT64, partition_nnz)) goto done;
    if (!read_dataset_1d(matrix, "partition_axes", H5T_NATIVE_UINT32, partition_axes)) goto done;
    if (!read_dataset_1d(matrix, "partition_row_offsets", H5T_NATIVE_UINT64, partition_row_offsets)) goto done;
    if (!read_dataset_1d(matrix, "shard_offsets", H5T_NATIVE_UINT64, shard_offsets)) goto done;

    clear(m);
    init(m);
    for (i = 0; i < (unsigned long) num_partitions; ++i) {
        if (!sharded_from_u64(partition_rows[i], part_rows_ul + i, "partition_rows", filename)) goto done;
        if (!sharded_from_u64(partition_nnz[i], part_nnz_ul + i, "partition_nnz", filename)) goto done;
        part_axes_ul[i] = (unsigned long) partition_axes[i];
    }
    for (i = 0; i <= (unsigned long) num_shards; ++i) {
        if (!sharded_from_u64(shard_offsets[i], shard_offsets_ul + i, "shard_offsets", filename)) goto done;
    }
    if (!define_partitions(m, (unsigned long) cols, (unsigned long) num_partitions, part_rows_ul, part_nnz_ul, part_axes_ul)) goto done;
    if (!reshard(m, (unsigned long) num_shards, shard_offsets_ul)) goto done;
    m->rows = (unsigned long) rows;
    m->nnz = (unsigned long) nnz;

    if (s != 0) {
        dataset_h5_state *state = 0;
        if (!bind_dataset_h5(s, filename)) goto done;
        state = (dataset_h5_state *) s->backend_state;
        state->rows = rows;
        state->cols = cols;
        state->nnz = nnz;
        state->num_partitions = num_partitions;
        state->num_shards = num_shards;
        state->num_codecs = (std::uint32_t) num_codecs;
        state->matrix_family = dataset_matrix_family_compressed;
        if (num_partitions != 0) {
            state->partition_rows = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_nnz = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_aux = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_row_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions + 1u, sizeof(std::uint64_t));
            state->partition_indptr_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_nnz_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_codec_ids = (std::uint32_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint32_t));
            if (state->partition_rows == 0
                || state->partition_nnz == 0
                || state->partition_aux == 0
                || state->partition_row_offsets == 0
                || state->partition_indptr_offsets == 0
                || state->partition_nnz_offsets == 0
                || state->partition_codec_ids == 0) goto done;
            std::memcpy(state->partition_rows, partition_rows, (std::size_t) num_partitions * sizeof(std::uint64_t));
            std::memcpy(state->partition_nnz, partition_nnz, (std::size_t) num_partitions * sizeof(std::uint64_t));
            std::memcpy(state->partition_row_offsets, partition_row_offsets, ((std::size_t) num_partitions + 1u) * sizeof(std::uint64_t));
            for (i = 0; i < (unsigned long) num_partitions; ++i) state->partition_aux[i] = (std::uint64_t) partition_axes[i];
        }
        state->shard_offsets = (std::uint64_t *) std::calloc((std::size_t) num_shards + 1u, sizeof(std::uint64_t));
        if (state->shard_offsets == 0 && (num_shards + 1u) != 0u) goto done;
        if (state->shard_offsets != 0) std::memcpy(state->shard_offsets, shard_offsets, ((std::size_t) num_shards + 1u) * sizeof(std::uint64_t));
        if (num_codecs != 0) {
            state->codecs = (dataset_codec_descriptor *) std::calloc((std::size_t) num_codecs, sizeof(dataset_codec_descriptor));
            if (state->codecs == 0) goto done;
        }
        {
            hid_t payload = H5Gopen2(file, payload_standard_group, H5P_DEFAULT);
            if (payload < 0) goto done;
            if (!read_dataset_1d(payload, "partition_indptr_offsets", H5T_NATIVE_UINT64, state->partition_indptr_offsets)) {
                H5Gclose(payload);
                goto done;
            }
            if (!read_dataset_1d(payload, "partition_nnz_offsets", H5T_NATIVE_UINT64, state->partition_nnz_offsets)) {
                H5Gclose(payload);
                goto done;
            }
            H5Gclose(payload);
        }
        if (!read_dataset_1d(matrix, "partition_codec_ids", H5T_NATIVE_UINT32, state->partition_codec_ids)) goto done;
        if (!load_codec_table(codecs, state->codecs, (std::uint32_t) num_codecs)) goto done;
        if (!build_shard_partition_spans(state)) goto done;
        if (!load_dataset_execution_metadata(file, state)) goto done;
        if (!load_dataset_runtime_service_metadata(file, state)) goto done;
    }

    ok = 1;

done:
    if (!ok && s != 0) clear(s);
    std::free(partition_rows);
    std::free(partition_nnz);
    std::free(partition_axes);
    std::free(partition_row_offsets);
    std::free(shard_offsets);
    std::free(part_rows_ul);
    std::free(part_nnz_ul);
    std::free(part_axes_ul);
    std::free(shard_offsets_ul);
    if (codecs >= 0) H5Gclose(codecs);
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int load_dataset_blocked_ell_h5_header(const char *filename,
                                      sharded<sparse::blocked_ell> *m,
                                      shard_storage *s) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t nnz = 0;
    std::uint64_t num_partitions = 0;
    std::uint64_t num_shards = 0;
    std::uint64_t num_codecs = 0;
    std::uint64_t *partition_rows = 0;
    std::uint64_t *partition_nnz = 0;
    std::uint64_t *partition_aux = 0;
    std::uint64_t *partition_row_offsets = 0;
    std::uint64_t *shard_offsets = 0;
    unsigned long *part_rows_ul = 0;
    unsigned long *part_nnz_ul = 0;
    unsigned long *part_aux_ul = 0;
    unsigned long *shard_offsets_ul = 0;
    unsigned long i = 0;
    int ok = 0;
    int optimized_codec = 0;
    char payload_layout[64];

    if (filename == 0 || m == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    if (!read_attr_u64(file, "rows", &rows)) goto done;
    if (!read_attr_u64(file, "cols", &cols)) goto done;
    if (!read_attr_u64(file, "nnz", &nnz)) goto done;
    if (!read_attr_u64(file, "num_partitions", &num_partitions)) goto done;
    if (!read_attr_u64(file, "num_shards", &num_shards)) goto done;
    if (!read_attr_u64(file, "num_codecs", &num_codecs)) goto done;
    payload_layout[0] = '\0';
    if (!read_attr_string(file, "payload_layout", payload_layout, sizeof(payload_layout))) goto done;

    matrix = H5Gopen2(file, matrix_group, H5P_DEFAULT);
    codecs = H5Gopen2(file, codecs_group, H5P_DEFAULT);
    if (matrix < 0 || codecs < 0) goto done;

    partition_rows = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    partition_nnz = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    partition_aux = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    partition_row_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions + 1u, sizeof(std::uint64_t));
    shard_offsets = (std::uint64_t *) std::calloc((std::size_t) num_shards + 1u, sizeof(std::uint64_t));
    part_rows_ul = (unsigned long *) std::calloc((std::size_t) num_partitions, sizeof(unsigned long));
    part_nnz_ul = (unsigned long *) std::calloc((std::size_t) num_partitions, sizeof(unsigned long));
    part_aux_ul = (unsigned long *) std::calloc((std::size_t) num_partitions, sizeof(unsigned long));
    shard_offsets_ul = (unsigned long *) std::calloc((std::size_t) num_shards + 1u, sizeof(unsigned long));
    if ((num_partitions != 0) && (partition_rows == 0 || partition_nnz == 0 || partition_aux == 0 || partition_row_offsets == 0 || part_rows_ul == 0 || part_nnz_ul == 0 || part_aux_ul == 0)) goto done;
    if ((num_shards + 1u) != 0u && (shard_offsets == 0 || shard_offsets_ul == 0)) goto done;

    if (!read_dataset_1d(matrix, "partition_rows", H5T_NATIVE_UINT64, partition_rows)) goto done;
    if (!read_dataset_1d(matrix, "partition_nnz", H5T_NATIVE_UINT64, partition_nnz)) goto done;
    if (!read_dataset_1d(matrix, "partition_aux", H5T_NATIVE_UINT64, partition_aux)) goto done;
    if (!read_dataset_1d(matrix, "partition_row_offsets", H5T_NATIVE_UINT64, partition_row_offsets)) goto done;
    if (!read_dataset_1d(matrix, "shard_offsets", H5T_NATIVE_UINT64, shard_offsets)) goto done;

    clear(m);
    init(m);
    for (i = 0; i < (unsigned long) num_partitions; ++i) {
        if (!sharded_from_u64(partition_rows[i], part_rows_ul + i, "partition_rows", filename)) goto done;
        if (!sharded_from_u64(partition_nnz[i], part_nnz_ul + i, "partition_nnz", filename)) goto done;
        if (!sharded_from_u64(partition_aux[i], part_aux_ul + i, "partition_aux", filename)) goto done;
    }
    for (i = 0; i <= (unsigned long) num_shards; ++i) {
        if (!sharded_from_u64(shard_offsets[i], shard_offsets_ul + i, "shard_offsets", filename)) goto done;
    }
    if (!define_partitions(m, (unsigned long) cols, (unsigned long) num_partitions, part_rows_ul, part_nnz_ul, part_aux_ul)) goto done;
    if (!reshard(m, (unsigned long) num_shards, shard_offsets_ul)) goto done;
    m->rows = (unsigned long) rows;
    m->nnz = (unsigned long) nnz;

    if (s != 0) {
        dataset_h5_state *state = 0;
        if (!bind_dataset_h5(s, filename)) goto done;
        state = (dataset_h5_state *) s->backend_state;
        state->rows = rows;
        state->cols = cols;
        state->nnz = nnz;
        state->num_partitions = num_partitions;
        state->num_shards = num_shards;
        state->num_codecs = (std::uint32_t) num_codecs;
        if (num_partitions != 0) {
            state->partition_rows = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_nnz = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_aux = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_row_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions + 1u, sizeof(std::uint64_t));
            state->partition_codec_ids = (std::uint32_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint32_t));
            if (state->partition_rows == 0
                || state->partition_nnz == 0
                || state->partition_aux == 0
                || state->partition_row_offsets == 0
                || state->partition_codec_ids == 0) goto done;
            std::memcpy(state->partition_rows, partition_rows, (std::size_t) num_partitions * sizeof(std::uint64_t));
            std::memcpy(state->partition_nnz, partition_nnz, (std::size_t) num_partitions * sizeof(std::uint64_t));
            std::memcpy(state->partition_aux, partition_aux, (std::size_t) num_partitions * sizeof(std::uint64_t));
            std::memcpy(state->partition_row_offsets, partition_row_offsets, ((std::size_t) num_partitions + 1u) * sizeof(std::uint64_t));
        }
        state->shard_offsets = (std::uint64_t *) std::calloc((std::size_t) num_shards + 1u, sizeof(std::uint64_t));
        if (state->shard_offsets == 0 && (num_shards + 1u) != 0u) goto done;
        if (state->shard_offsets != 0) std::memcpy(state->shard_offsets, shard_offsets, ((std::size_t) num_shards + 1u) * sizeof(std::uint64_t));
        if (num_codecs != 0) {
            state->codecs = (dataset_codec_descriptor *) std::calloc((std::size_t) num_codecs, sizeof(dataset_codec_descriptor));
            if (state->codecs == 0) goto done;
        }
        if (!read_dataset_1d(matrix, "partition_codec_ids", H5T_NATIVE_UINT32, state->partition_codec_ids)) goto done;
        if (!load_codec_table(codecs, state->codecs, (std::uint32_t) num_codecs)) goto done;
        optimized_codec = std::strcmp(payload_layout, payload_layout_optimized_blocked_ell) == 0;
        state->matrix_family = optimized_codec ? dataset_matrix_family_optimized_blocked_ell : dataset_matrix_family_blocked_ell;
        if (!optimized_codec) {
            state->partition_block_idx_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_value_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->shard_block_idx_offsets = (std::uint64_t *) std::calloc((std::size_t) num_shards + 1u, sizeof(std::uint64_t));
            state->shard_value_offsets = (std::uint64_t *) std::calloc((std::size_t) num_shards + 1u, sizeof(std::uint64_t));
            if ((num_partitions != 0u && (state->partition_block_idx_offsets == 0 || state->partition_value_offsets == 0))
                || (num_shards != 0u && (state->shard_block_idx_offsets == 0 || state->shard_value_offsets == 0))) {
                goto done;
            }
            {
                hid_t payload = H5Gopen2(file, payload_blocked_ell_group, H5P_DEFAULT);
                if (payload < 0) goto done;
                if (!read_dataset_1d(payload, "partition_block_idx_offsets", H5T_NATIVE_UINT64, state->partition_block_idx_offsets)) {
                    H5Gclose(payload);
                    goto done;
                }
                if (!read_dataset_1d(payload, "partition_value_offsets", H5T_NATIVE_UINT64, state->partition_value_offsets)) {
                    H5Gclose(payload);
                    goto done;
                }
                if (!read_dataset_1d(payload, "shard_block_idx_offsets", H5T_NATIVE_UINT64, state->shard_block_idx_offsets)) {
                    H5Gclose(payload);
                    goto done;
                }
                if (!read_dataset_1d(payload, "shard_value_offsets", H5T_NATIVE_UINT64, state->shard_value_offsets)) {
                    H5Gclose(payload);
                    goto done;
                }
                H5Gclose(payload);
            }
        }
        if (!build_shard_partition_spans(state)) goto done;
        if (!load_dataset_execution_metadata(file, state)) goto done;
        if (!load_dataset_runtime_service_metadata(file, state)) goto done;
    }

    ok = 1;

done:
    if (!ok && s != 0) clear(s);
    std::free(partition_rows);
    std::free(partition_nnz);
    std::free(partition_aux);
    std::free(partition_row_offsets);
    std::free(shard_offsets);
    std::free(part_rows_ul);
    std::free(part_nnz_ul);
    std::free(part_aux_ul);
    std::free(shard_offsets_ul);
    if (codecs >= 0) H5Gclose(codecs);
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int load_dataset_sliced_ell_h5_header(const char *filename,
                                      sharded<sparse::sliced_ell> *m,
                                      shard_storage *s) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t nnz = 0;
    std::uint64_t num_partitions = 0;
    std::uint64_t num_shards = 0;
    std::uint64_t num_codecs = 0;
    std::uint64_t *partition_rows = 0;
    std::uint64_t *partition_nnz = 0;
    std::uint64_t *partition_aux = 0;
    std::uint64_t *partition_row_offsets = 0;
    std::uint64_t *shard_offsets = 0;
    unsigned long *part_rows_ul = 0;
    unsigned long *part_nnz_ul = 0;
    unsigned long *part_aux_ul = 0;
    unsigned long *shard_offsets_ul = 0;
    unsigned long i = 0;
    int ok = 0;

    if (filename == 0 || m == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    if (!read_attr_u64(file, "rows", &rows)) goto done;
    if (!read_attr_u64(file, "cols", &cols)) goto done;
    if (!read_attr_u64(file, "nnz", &nnz)) goto done;
    if (!read_attr_u64(file, "num_partitions", &num_partitions)) goto done;
    if (!read_attr_u64(file, "num_shards", &num_shards)) goto done;
    if (!read_attr_u64(file, "num_codecs", &num_codecs)) goto done;

    matrix = H5Gopen2(file, matrix_group, H5P_DEFAULT);
    codecs = H5Gopen2(file, codecs_group, H5P_DEFAULT);
    if (matrix < 0 || codecs < 0) goto done;

    partition_rows = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    partition_nnz = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    partition_aux = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    partition_row_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions + 1u, sizeof(std::uint64_t));
    shard_offsets = (std::uint64_t *) std::calloc((std::size_t) num_shards + 1u, sizeof(std::uint64_t));
    part_rows_ul = (unsigned long *) std::calloc((std::size_t) num_partitions, sizeof(unsigned long));
    part_nnz_ul = (unsigned long *) std::calloc((std::size_t) num_partitions, sizeof(unsigned long));
    part_aux_ul = (unsigned long *) std::calloc((std::size_t) num_partitions, sizeof(unsigned long));
    shard_offsets_ul = (unsigned long *) std::calloc((std::size_t) num_shards + 1u, sizeof(unsigned long));
    if ((num_partitions != 0) && (partition_rows == 0 || partition_nnz == 0 || partition_aux == 0 || partition_row_offsets == 0 || part_rows_ul == 0 || part_nnz_ul == 0 || part_aux_ul == 0)) goto done;
    if ((num_shards + 1u) != 0u && (shard_offsets == 0 || shard_offsets_ul == 0)) goto done;

    if (!read_dataset_1d(matrix, "partition_rows", H5T_NATIVE_UINT64, partition_rows)) goto done;
    if (!read_dataset_1d(matrix, "partition_nnz", H5T_NATIVE_UINT64, partition_nnz)) goto done;
    if (!read_dataset_1d(matrix, "partition_aux", H5T_NATIVE_UINT64, partition_aux)) goto done;
    if (!read_dataset_1d(matrix, "partition_row_offsets", H5T_NATIVE_UINT64, partition_row_offsets)) goto done;
    if (!read_dataset_1d(matrix, "shard_offsets", H5T_NATIVE_UINT64, shard_offsets)) goto done;

    clear(m);
    init(m);
    for (i = 0; i < (unsigned long) num_partitions; ++i) {
        if (!sharded_from_u64(partition_rows[i], part_rows_ul + i, "partition_rows", filename)) goto done;
        if (!sharded_from_u64(partition_nnz[i], part_nnz_ul + i, "partition_nnz", filename)) goto done;
        if (!sharded_from_u64(partition_aux[i], part_aux_ul + i, "partition_aux", filename)) goto done;
    }
    for (i = 0; i <= (unsigned long) num_shards; ++i) {
        if (!sharded_from_u64(shard_offsets[i], shard_offsets_ul + i, "shard_offsets", filename)) goto done;
    }
    if (!define_partitions(m, (unsigned long) cols, (unsigned long) num_partitions, part_rows_ul, part_nnz_ul, part_aux_ul)) goto done;
    if (!reshard(m, (unsigned long) num_shards, shard_offsets_ul)) goto done;
    m->rows = (unsigned long) rows;
    m->nnz = (unsigned long) nnz;

    if (s != 0) {
        dataset_h5_state *state = 0;
        if (!bind_dataset_h5(s, filename)) goto done;
        state = (dataset_h5_state *) s->backend_state;
        state->rows = rows;
        state->cols = cols;
        state->nnz = nnz;
        state->num_partitions = num_partitions;
        state->num_shards = num_shards;
        state->num_codecs = (std::uint32_t) num_codecs;
        state->matrix_family = dataset_matrix_family_sliced_ell;
        if (num_partitions != 0) {
            state->partition_rows = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_nnz = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_aux = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_row_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions + 1u, sizeof(std::uint64_t));
            state->partition_codec_ids = (std::uint32_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint32_t));
            if (state->partition_rows == 0
                || state->partition_nnz == 0
                || state->partition_aux == 0
                || state->partition_row_offsets == 0
                || state->partition_codec_ids == 0) goto done;
            std::memcpy(state->partition_rows, partition_rows, (std::size_t) num_partitions * sizeof(std::uint64_t));
            std::memcpy(state->partition_nnz, partition_nnz, (std::size_t) num_partitions * sizeof(std::uint64_t));
            std::memcpy(state->partition_aux, partition_aux, (std::size_t) num_partitions * sizeof(std::uint64_t));
            std::memcpy(state->partition_row_offsets, partition_row_offsets, ((std::size_t) num_partitions + 1u) * sizeof(std::uint64_t));
        }
        state->shard_offsets = (std::uint64_t *) std::calloc((std::size_t) num_shards + 1u, sizeof(std::uint64_t));
        if (state->shard_offsets == 0 && (num_shards + 1u) != 0u) goto done;
        if (state->shard_offsets != 0) std::memcpy(state->shard_offsets, shard_offsets, ((std::size_t) num_shards + 1u) * sizeof(std::uint64_t));
        if (num_codecs != 0) {
            state->codecs = (dataset_codec_descriptor *) std::calloc((std::size_t) num_codecs, sizeof(dataset_codec_descriptor));
            if (state->codecs == 0) goto done;
        }
        if (!read_dataset_1d(matrix, "partition_codec_ids", H5T_NATIVE_UINT32, state->partition_codec_ids)) goto done;
        if (!load_codec_table(codecs, state->codecs, (std::uint32_t) num_codecs)) goto done;
        if (!build_shard_partition_spans(state)) goto done;
        if (!load_dataset_execution_metadata(file, state)) goto done;
        if (!load_dataset_runtime_service_metadata(file, state)) goto done;
    }

    ok = 1;

done:
    if (!ok && s != 0) clear(s);
    std::free(partition_rows);
    std::free(partition_nnz);
    std::free(partition_aux);
    std::free(partition_row_offsets);
    std::free(shard_offsets);
    std::free(part_rows_ul);
    std::free(part_nnz_ul);
    std::free(part_aux_ul);
    std::free(shard_offsets_ul);
    if (codecs >= 0) H5Gclose(codecs);
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int fetch_dataset_compressed_h5_partition(sharded<sparse::compressed> *m,
                                    const shard_storage *s,
                                    unsigned long partition_id) {
    shard_storage *storage = const_cast<shard_storage *>(s);
    dataset_h5_state *state = 0;
    if (m == 0 || storage == 0 || storage->backend != shard_storage_backend_dataset_h5 || partition_id >= m->num_partitions || storage->backend_state == 0) return 0;
    state = (dataset_h5_state *) storage->backend_state;
    if (!ensure_cached_shard_ready(storage, (unsigned long) state->partition_shard_ids[partition_id])) return 0;
    return load_compressed_part_from_cached_pack(m, state, partition_id);
}

int fetch_dataset_compressed_h5_shard(sharded<sparse::compressed> *m,
                                     const shard_storage *s,
                                     unsigned long shard_id) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    dataset_h5_state *state = 0;

    if (m == 0 || s == 0 || s->backend_state == 0 || shard_id >= m->num_shards) return 0;
    state = (dataset_h5_state *) s->backend_state;
    if (!ensure_cached_shard_ready(const_cast<shard_storage *>(s), shard_id)) return 0;
    begin = first_partition_in_shard(m, shard_id);
    end = last_partition_in_shard(m, shard_id);
    for (i = begin; i < end; ++i) {
        if (!load_compressed_part_from_cached_pack(m, state, i)) return 0;
    }
    return 1;
}

int prefetch_dataset_compressed_h5_partition_cache(const sharded<sparse::compressed> *m,
                                             shard_storage *s,
                                             unsigned long partition_id) {
    dataset_h5_state *state = 0;
    if (m == 0 || s == 0 || s->backend != shard_storage_backend_dataset_h5 || partition_id >= m->num_partitions || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
    return ensure_cached_shard_ready(s, (unsigned long) state->partition_shard_ids[partition_id]);
}

int prefetch_dataset_compressed_h5_shard_cache(const sharded<sparse::compressed> *m,
                                              shard_storage *s,
                                              unsigned long shard_id) {
    if (m == 0 || s == 0 || shard_id >= m->num_shards) return 0;
    return ensure_cached_shard_ready(s, shard_id);
}

int fetch_dataset_blocked_ell_h5_partition(sharded<sparse::blocked_ell> *m,
                                     const shard_storage *s,
                                     unsigned long partition_id) {
    shard_storage *storage = const_cast<shard_storage *>(s);
    dataset_h5_state *state = 0;
    if (m == 0 || storage == 0 || storage->backend != shard_storage_backend_dataset_h5 || partition_id >= m->num_partitions || storage->backend_state == 0) return 0;
    state = (dataset_h5_state *) storage->backend_state;
    if (!ensure_cached_shard_ready(storage, (unsigned long) state->partition_shard_ids[partition_id])) return 0;
    return load_blocked_ell_part_from_cached_pack(m, state, partition_id);
}

int fetch_dataset_blocked_ell_h5_shard(sharded<sparse::blocked_ell> *m,
                                      const shard_storage *s,
                                      unsigned long shard_id) {
    const unsigned long begin = first_partition_in_shard(m, shard_id);
    const unsigned long end = last_partition_in_shard(m, shard_id);
    unsigned long i = 0;
    dataset_h5_state *state = 0;

    if (m == 0 || s == 0 || shard_id >= m->num_shards || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
    if (!ensure_cached_shard_ready(const_cast<shard_storage *>(s), shard_id)) return 0;
    for (i = begin; i < end; ++i) {
        if (!load_blocked_ell_part_from_cached_pack(m, state, i)) return 0;
    }
    return 1;
}

int fetch_dataset_sliced_ell_h5_partition(sharded<sparse::sliced_ell> *m,
                                          const shard_storage *s,
                                          unsigned long partition_id) {
    shard_storage *storage = const_cast<shard_storage *>(s);
    dataset_h5_state *state = 0;
    if (m == 0 || storage == 0 || storage->backend != shard_storage_backend_dataset_h5 || partition_id >= m->num_partitions || storage->backend_state == 0) return 0;
    state = (dataset_h5_state *) storage->backend_state;
    if (!ensure_cached_shard_ready(storage, (unsigned long) state->partition_shard_ids[partition_id])) return 0;
    return load_sliced_ell_part_from_cached_pack(m, state, partition_id);
}

int fetch_dataset_sliced_ell_h5_shard(sharded<sparse::sliced_ell> *m,
                                      const shard_storage *s,
                                      unsigned long shard_id) {
    const unsigned long begin = first_partition_in_shard(m, shard_id);
    const unsigned long end = last_partition_in_shard(m, shard_id);
    unsigned long i = 0;
    dataset_h5_state *state = 0;

    if (m == 0 || s == 0 || shard_id >= m->num_shards || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
    if (!ensure_cached_shard_ready(const_cast<shard_storage *>(s), shard_id)) return 0;
    for (i = begin; i < end; ++i) {
        if (!load_sliced_ell_part_from_cached_pack(m, state, i)) return 0;
    }
    return 1;
}

int prefetch_dataset_blocked_ell_h5_partition_cache(const sharded<sparse::blocked_ell> *m,
                                              shard_storage *s,
                                              unsigned long partition_id) {
    dataset_h5_state *state = 0;
    if (m == 0 || s == 0 || s->backend != shard_storage_backend_dataset_h5 || partition_id >= m->num_partitions || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
    return ensure_cached_shard_ready(s, (unsigned long) state->partition_shard_ids[partition_id]);
}

int prefetch_dataset_blocked_ell_h5_shard_cache(const sharded<sparse::blocked_ell> *m,
                                               shard_storage *s,
                                               unsigned long shard_id) {
    if (m == 0 || s == 0 || shard_id >= m->num_shards) return 0;
    return ensure_cached_shard_ready(s, shard_id);
}

int fetch_dataset_sliced_ell_h5_execution_partition(bucketed_sliced_ell_partition *out,
                                                    const sharded<sparse::sliced_ell> *m,
                                                    const shard_storage *s,
                                                    unsigned long partition_id) {
    shard_storage *storage = const_cast<shard_storage *>(s);
    dataset_h5_state *state = 0;
    unsigned long shard_id = 0ul;

    if (out == 0 || m == 0 || storage == 0 || storage->backend != shard_storage_backend_dataset_h5
        || partition_id >= m->num_partitions || storage->backend_state == 0) {
        return 0;
    }
    state = (dataset_h5_state *) storage->backend_state;
    shard_id = (unsigned long) state->partition_shard_ids[partition_id];
    if (!ensure_execution_pack_ready(storage, state, shard_id)) return 0;
    if (load_sliced_execution_partition_from_pack(state, shard_id, partition_id, out)) return 1;
    if (!ensure_optimized_sliced_ell_payload_open(state) || !load_optimized_sliced_ell_shard_payload(state, shard_id)) {
        if (!fetch_dataset_sliced_ell_h5_partition(const_cast<sharded<sparse::sliced_ell> *>(m), storage, partition_id)) return 0;
        if (m->parts[partition_id] == 0) return 0;
        return build_bucketed_sliced_execution_partition(out,
                                                         m->parts[partition_id],
                                                         state->partition_blocked_ell_bucket_counts != 0
                                                             ? state->partition_blocked_ell_bucket_counts[partition_id]
                                                             : 1u,
                                                         0);
    }
    {
        const std::uint64_t begin = state->shard_part_begin[shard_id];
        const std::uint64_t local_partition = partition_id - begin;
        if (local_partition >= state->loaded_optimized_sliced_shard.partition_count) return 0;
        return clone_bucketed_sliced_partition(out, state->loaded_optimized_sliced_shard.partitions + local_partition);
    }
}

int warm_dataset_blocked_ell_h5_cache_range(const char *filename,
                                           const char *cache_root,
                                           unsigned long shard_begin,
                                           unsigned long shard_end) {
    sharded<sparse::blocked_ell> matrix;
    shard_storage storage;
    unsigned long shard_id = 0ul;
    int ok = 0;

    if (filename == 0 || cache_root == 0 || *cache_root == '\0') return 0;

    init(&matrix);
    init(&storage);
    if (!load_dataset_blocked_ell_h5_header(filename, &matrix, &storage)) goto done;
    if (!bind_dataset_h5_cache(&storage, cache_root)) goto done;
    if (shard_begin > matrix.num_shards) goto done;
    if (shard_end > matrix.num_shards) shard_end = matrix.num_shards;
    for (shard_id = shard_begin; shard_id < shard_end; ++shard_id) {
        if (!prefetch_dataset_blocked_ell_h5_shard_cache(&matrix, &storage, shard_id)) goto done;
    }
    ok = 1;

done:
    clear(&storage);
    clear(&matrix);
    return ok;
}

int warm_dataset_blocked_ell_h5_cache(const char *filename,
                                     const char *cache_root) {
    sharded<sparse::blocked_ell> matrix;
    shard_storage storage;
    unsigned long shard_id = 0ul;
    int ok = 0;

    if (filename == 0 || cache_root == 0 || *cache_root == '\0') return 0;

    init(&matrix);
    init(&storage);
    if (!load_dataset_blocked_ell_h5_header(filename, &matrix, &storage)) goto done;
    if (!bind_dataset_h5_cache(&storage, cache_root)) goto done;
    for (shard_id = 0ul; shard_id < matrix.num_shards; ++shard_id) {
        if (!prefetch_dataset_blocked_ell_h5_shard_cache(&matrix, &storage, shard_id)) goto done;
    }
    ok = 1;

done:
    clear(&storage);
    clear(&matrix);
    return ok;
}

int warm_dataset_blocked_ell_h5_execution_cache_range(const char *filename,
                                                     const char *cache_root,
                                                     unsigned long shard_begin,
                                                     unsigned long shard_end) {
    sharded<sparse::blocked_ell> matrix;
    shard_storage storage;
    dataset_h5_state *state = 0;
    unsigned long shard_id = 0ul;
    int ok = 0;

    if (filename == 0 || cache_root == 0 || *cache_root == '\0') return 0;

    init(&matrix);
    init(&storage);
    if (!load_dataset_blocked_ell_h5_header(filename, &matrix, &storage)) goto done;
    if (!bind_dataset_h5_cache(&storage, cache_root)) goto done;
    if (shard_begin > matrix.num_shards) goto done;
    if (shard_end > matrix.num_shards) shard_end = matrix.num_shards;
    state = storage.backend_state != 0 ? (dataset_h5_state *) storage.backend_state : 0;
    if (state == 0) goto done;
    for (shard_id = shard_begin; shard_id < shard_end; ++shard_id) {
        if (!ensure_execution_pack_ready(&storage, state, shard_id)) goto done;
    }
    ok = 1;

done:
    clear(&storage);
    clear(&matrix);
    return ok;
}

int warm_dataset_blocked_ell_h5_execution_cache(const char *filename,
                                               const char *cache_root) {
    sharded<sparse::blocked_ell> matrix;
    shard_storage storage;
    dataset_h5_state *state = 0;
    unsigned long shard_id = 0ul;
    int ok = 0;

    if (filename == 0 || cache_root == 0 || *cache_root == '\0') return 0;

    init(&matrix);
    init(&storage);
    if (!load_dataset_blocked_ell_h5_header(filename, &matrix, &storage)) goto done;
    if (!bind_dataset_h5_cache(&storage, cache_root)) goto done;
    state = storage.backend_state != 0 ? (dataset_h5_state *) storage.backend_state : 0;
    if (state == 0) goto done;
    for (shard_id = 0ul; shard_id < matrix.num_shards; ++shard_id) {
        if (!ensure_execution_pack_ready(&storage, state, shard_id)) goto done;
    }
    ok = 1;

done:
    clear(&storage);
    clear(&matrix);
    return ok;
}

int fetch_dataset_blocked_ell_h5_execution_partition(bucketed_blocked_ell_partition *out,
                                                    const sharded<sparse::blocked_ell> *m,
                                                    const shard_storage *s,
                                                    unsigned long partition_id) {
    shard_storage *storage = const_cast<shard_storage *>(s);
    dataset_h5_state *state = 0;
    unsigned long shard_id = 0ul;

    if (out == 0 || m == 0 || storage == 0 || storage->backend != shard_storage_backend_dataset_h5
        || partition_id >= m->num_partitions || storage->backend_state == 0) {
        return 0;
    }
    state = (dataset_h5_state *) storage->backend_state;
    shard_id = (unsigned long) state->partition_shard_ids[partition_id];
    if (!ensure_execution_pack_ready(storage, state, shard_id)) return 0;
    if (load_execution_partition_from_pack(state, shard_id, partition_id, out)) return 1;
    if (state->matrix_family == dataset_matrix_family_optimized_blocked_ell) {
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
