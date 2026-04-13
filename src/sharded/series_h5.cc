#include "series_h5.cuh"

#include "disk.cuh"
#include "sharded_host.cuh"

#include <hdf5.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <limits>
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

static const char series_magic[] = "CSH5S1";
static const char root_group[] = "/";
static const char matrix_group[] = "/matrix";
static const char datasets_group[] = "/datasets";
static const char provenance_group[] = "/provenance";
static const char codecs_group[] = "/codecs";
static const char embedded_metadata_group[] = "/embedded_metadata";
static const char observation_metadata_group[] = "/observation_metadata";
static const char browse_group[] = "/browse";
static const char execution_group[] = "/execution";
static const char payload_group[] = "/payload";
static const char payload_standard_group[] = "/payload/standard_csr";
static const char payload_blocked_ell_group[] = "/payload/blocked_ell";
static const char payload_layout_shard_packed[] = "shard_packed";
static const std::uint32_t series_cache_schema_version = 1u;
static const std::uint64_t shard_pack_payload_alignment = 4096u;

enum {
    series_matrix_family_none = 0u,
    series_matrix_family_compressed = 1u,
    series_matrix_family_blocked_ell = 2u
};

enum {
    series_cache_shard_missing = 0u,
    series_cache_shard_queued = 1u,
    series_cache_shard_building = 2u,
    series_cache_shard_ready = 3u,
    series_cache_shard_failed = 4u
};

struct series_h5_cache_runtime {
    std::mutex state_mutex;
    std::condition_variable state_cv;
    std::deque<unsigned long> shard_queue;
    std::thread reader_thread;
    bool reader_started;
    bool stop_requested;
    std::mutex *shard_file_mutexes;

    explicit series_h5_cache_runtime(std::size_t shard_count)
        : reader_started(false),
          stop_requested(false),
          shard_file_mutexes(shard_count != 0u ? new std::mutex[shard_count] : nullptr) {}

    ~series_h5_cache_runtime() {
        delete[] shard_file_mutexes;
    }
};

struct series_h5_state {
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
    series_codec_descriptor *codecs;
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

inline void series_h5_state_init(series_h5_state *state) {
    state->file = (hid_t) -1;
    state->rows = 0u;
    state->cols = 0u;
    state->nnz = 0u;
    state->num_partitions = 0;
    state->num_shards = 0;
    state->num_codecs = 0;
    state->matrix_family = series_matrix_family_none;
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

void close_series_h5_backend(shard_storage *s);

inline void series_h5_state_clear(series_h5_state *state) {
    unsigned long shard_i = 0ul;
    if (state != 0 && state->cache_runtime != 0) {
        series_h5_cache_runtime *runtime = (series_h5_cache_runtime *) state->cache_runtime;
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
    delete (series_h5_cache_runtime *) state->cache_runtime;
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
    state->matrix_family = series_matrix_family_none;
}

inline hid_t create_group(hid_t parent, const char *path) {
    hid_t group = H5Gcreate2(parent, path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (group >= 0) return group;
    return H5Gopen2(parent, path, H5P_DEFAULT);
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

inline int write_text_column(hid_t group, const char *name, const series_text_column_view *column) {
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
    return std::strcmp(got, series_magic) == 0;
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

inline std::uint64_t local_dim_limit() {
    return (std::uint64_t) std::numeric_limits<types::dim_t>::max();
}

inline std::uint64_t local_nnz_limit() {
    return (std::uint64_t) std::numeric_limits<types::nnz_t>::max();
}

inline std::uint64_t local_index_limit() {
    return (std::uint64_t) std::numeric_limits<types::idx_t>::max();
}

inline int fail_series_u32_limit(const char *filename,
                                 const char *scope,
                                 std::uint64_t id,
                                 const char *field,
                                 std::uint64_t value,
                                 std::uint64_t limit) {
    std::fprintf(stderr,
                 "cellshard: %s exceeds the current u32 execution limit while writing %s (%s=%llu, %s=%llu, limit=%llu)\n",
                 scope != 0 ? scope : "series payload",
                 filename != 0 ? filename : "<memory>",
                 scope != 0 && std::strcmp(scope, "part") == 0 ? "partition_id" : "id",
                 (unsigned long long) id,
                 field != 0 ? field : "value",
                 (unsigned long long) value,
                 (unsigned long long) limit);
    return 0;
}

inline void warn_series_u32_limit(const char *filename,
                                  const char *scope,
                                  std::uint64_t id,
                                  const char *field,
                                  std::uint64_t value,
                                  std::uint64_t limit) {
    std::fprintf(stderr,
                 "cellshard: warning: %s exceeds the current u32 execution limit for %s (%s=%llu, %s=%llu, limit=%llu)\n",
                 scope != 0 ? scope : "series payload",
                 filename != 0 ? filename : "<memory>",
                 scope != 0 && std::strcmp(scope, "shard") == 0 ? "shard_id" : "id",
                 (unsigned long long) id,
                 field != 0 ? field : "value",
                 (unsigned long long) value,
                 (unsigned long long) limit);
}

inline int reserve_blocked_ell_shard_scratch(series_h5_state *state,
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

int open_series_h5_backend(shard_storage *s);
void close_series_h5_backend(shard_storage *s);

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
    const std::uint32_t schema_version = series_h5_schema_version;
    std::uint64_t h = 1469598103934665603ull;
    if (source_path != 0) h = fnv1a_mix(h, source_path, std::strlen(source_path));
    h = fnv1a_mix(h, &size_bytes, sizeof(size_bytes));
    h = fnv1a_mix(h, &mtime_ns, sizeof(mtime_ns));
    h = fnv1a_mix(h, &matrix_family, sizeof(matrix_family));
    h = fnv1a_mix(h, &num_partitions, sizeof(num_partitions));
    h = fnv1a_mix(h, &num_shards, sizeof(num_shards));
    h = fnv1a_mix(h, &schema_version, sizeof(schema_version));
    h = fnv1a_mix(h, series_magic, std::strlen(series_magic));
    return h;
}

inline int build_cache_instance_path(const char *cache_root,
                                     std::uint64_t fingerprint,
                                     char *path,
                                     std::size_t cap) {
    if (cache_root == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/%016llx", cache_root, (unsigned long long) fingerprint) > 0;
}

inline int build_cache_manifest_path(const char *cache_instance_dir,
                                     char *path,
                                     std::size_t cap) {
    if (cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/manifest.txt", cache_instance_dir) > 0;
}

inline int build_shard_pack_path(const series_h5_state *state,
                                 unsigned long shard_id,
                                 char *path,
                                 std::size_t cap) {
    if (state == 0 || state->cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/shard.%lu.pack", state->cache_instance_dir, shard_id) > 0;
}

inline int build_shard_pack_temp_path(const series_h5_state *state,
                                      unsigned long shard_id,
                                      char *path,
                                      std::size_t cap) {
    if (state == 0 || state->cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/shard.%lu.pack.tmp", state->cache_instance_dir, shard_id) > 0;
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

inline series_h5_cache_runtime *cache_runtime(series_h5_state *state) {
    return state != 0 ? (series_h5_cache_runtime *) state->cache_runtime : 0;
}

inline int refresh_series_source_stat(const char *source_path,
                                      series_h5_state *state) {
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

inline int ensure_cache_tracking_allocated(series_h5_state *state) {
    std::size_t count = 0u;
    if (state == 0) return 0;
    if (state->cache_runtime == 0) {
        state->cache_runtime = new series_h5_cache_runtime((std::size_t) state->num_shards);
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

inline std::uint64_t estimate_shard_pack_bytes(const series_h5_state *state, unsigned long shard_id) {
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
    if (state->matrix_family == series_matrix_family_compressed) {
        if (!compute_shard_pack_locators<sparse::compressed>(state->partition_rows + begin,
                                                             state->partition_nnz + begin,
                                                             state->partition_aux + begin,
                                                             state->cols,
                                                             local_count,
                                                             local_offsets,
                                                             local_sizes)) {
            goto done;
        }
    } else if (state->matrix_family == series_matrix_family_blocked_ell) {
        if (!compute_shard_pack_locators<sparse::blocked_ell>(state->partition_rows + begin,
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

inline int write_series_cache_manifest(const char *source_path,
                                       const series_h5_state *state) {
    std::FILE *fp = 0;
    unsigned long shard_id = 0ul;
    if (source_path == 0 || state == 0 || state->cache_manifest_path == 0) return 0;
    fp = std::fopen(state->cache_manifest_path, "wb");
    if (fp == 0) return 0;
    std::fprintf(fp, "cache_schema_version=%u\n", (unsigned int) series_cache_schema_version);
    std::fprintf(fp, "source_path=%s\n", source_path);
    std::fprintf(fp, "source_size_bytes=%llu\n", (unsigned long long) state->source_size_bytes);
    std::fprintf(fp, "source_mtime_ns=%llu\n", (unsigned long long) state->source_mtime_ns);
    std::fprintf(fp, "matrix_family=%u\n", (unsigned int) state->matrix_family);
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

inline int ensure_series_cache_layout(shard_storage *s) {
    series_h5_state *state = 0;
    char path[4096];
    std::uint64_t fingerprint = 0u;
    unsigned long shard_id = 0ul;
    struct statvfs vfs;

    if (s == 0 || s->backend != shard_storage_backend_series_h5 || s->backend_state == 0 || s->source_path == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    if (!refresh_series_source_stat(s->source_path, state)) return 0;
    if (state->cache_root == 0) {
        if (!build_default_cache_root(s->source_path, path, sizeof(path))) return 0;
        if (!assign_owned_string(&state->cache_root, path)) return 0;
    }
    if (!ensure_directory_exists(state->cache_root)) return 0;
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
    if (!build_cache_manifest_path(state->cache_instance_dir, path, sizeof(path))) return 0;
    if (!assign_owned_string(&state->cache_manifest_path, path)) return 0;
    if (!ensure_cache_tracking_allocated(state)) return 0;
    for (shard_id = 0ul; shard_id < (unsigned long) state->num_shards; ++shard_id) {
        if (state->shard_cache_paths[shard_id] == 0) {
            if (!build_shard_pack_path(state, shard_id, path, sizeof(path))) return 0;
            if (!assign_owned_string(state->shard_cache_paths + shard_id, path)) return 0;
        }
        if (::access(state->shard_cache_paths[shard_id], R_OK) == 0) {
            if (state->shard_cache_state[shard_id] != series_cache_shard_ready) {
                state->shard_cache_state[shard_id] = series_cache_shard_ready;
                state->shard_cache_bytes[shard_id] = estimate_shard_pack_bytes(state, shard_id);
                state->cache_resident_bytes += state->shard_cache_bytes[shard_id];
            }
        }
    }
    if (!write_series_cache_manifest(s->source_path, state)) return 0;
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

inline int build_shard_partition_spans(series_h5_state *state) {
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

inline int load_series_h5_state(hid_t file, series_h5_state *state) {
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
        state->codecs = (series_codec_descriptor *) std::calloc((std::size_t) state->num_codecs, sizeof(series_codec_descriptor));
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

inline int load_codec_table(hid_t codecs, series_codec_descriptor *descs, std::uint32_t count) {
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

inline const series_codec_descriptor *find_codec(const series_h5_state *state, std::uint32_t codec_id) {
    std::uint32_t i = 0;
    if (state == 0) return 0;
    for (i = 0; i < state->num_codecs; ++i) {
        if (state->codecs[i].codec_id == codec_id) return state->codecs + i;
    }
    return 0;
}

inline int ensure_standard_payload_open(series_h5_state *state) {
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

inline int ensure_blocked_ell_payload_open(series_h5_state *state) {
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

inline int load_blocked_ell_shard_payload(series_h5_state *state,
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

inline int prepare_blocked_ell_parts_from_state(const series_h5_state *state,
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

inline int fill_blocked_ell_parts_from_loaded_shard(const series_h5_state *state,
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

inline int load_or_materialize_blocked_ell_parts(sharded<sparse::blocked_ell> *m,
                                                 series_h5_state *state,
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

inline void close_cached_shard_file(series_h5_state *state, unsigned long shard_id) {
    series_h5_cache_runtime *runtime = cache_runtime(state);
    if (state == 0 || runtime == 0 || shard_id >= state->num_shards || state->shard_cache_files == 0) return;
    std::lock_guard<std::mutex> file_lock(runtime->shard_file_mutexes[shard_id]);
    if (state->shard_cache_files[shard_id] != 0) {
        std::fclose(state->shard_cache_files[shard_id]);
        state->shard_cache_files[shard_id] = 0;
    }
}

inline int ensure_cached_shard_file_open(series_h5_state *state, unsigned long shard_id) {
    series_h5_cache_runtime *runtime = cache_runtime(state);
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
inline void compute_cached_part_locator(const series_h5_state *state,
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
                                                 series_h5_state *state,
                                                 unsigned long partition_id) {
    const unsigned long shard_id = state != 0 && state->partition_shard_ids != 0 ? (unsigned long) state->partition_shard_ids[partition_id] : 0ul;
    series_h5_cache_runtime *runtime = cache_runtime(state);
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
                                                  series_h5_state *state,
                                                  unsigned long partition_id) {
    const unsigned long shard_id = state != 0 && state->partition_shard_ids != 0 ? (unsigned long) state->partition_shard_ids[partition_id] : 0ul;
    series_h5_cache_runtime *runtime = cache_runtime(state);
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

inline int materialize_compressed_shard_pack(shard_storage *s, series_h5_state *state, unsigned long shard_id) {
    const std::uint64_t begin = state != 0 && state->shard_part_begin != 0 ? state->shard_part_begin[shard_id] : 0u;
    const std::uint64_t end = state != 0 && state->shard_part_end != 0 ? state->shard_part_end[shard_id] : 0u;
    const std::uint64_t partition_count = end >= begin ? (end - begin) : 0u;
    sparse::compressed **parts = 0;
    char tmp_path[4096];
    char final_path[4096];
    std::uint64_t local = 0u;
    int ok = 0;

    if (s == 0 || state == 0 || shard_id >= state->num_shards) return 0;
    if (!open_series_h5_backend(s) || !ensure_standard_payload_open(state)) return 0;
    if (partition_count != 0u) {
        parts = (sparse::compressed **) std::calloc((std::size_t) partition_count, sizeof(sparse::compressed *));
        if (parts == 0) return 0;
    }
    for (local = 0u; local < partition_count; ++local) {
        const std::uint64_t partition_id = begin + local;
        const series_codec_descriptor *codec = find_codec(state, state->partition_codec_ids[partition_id]);
        sparse::compressed *part = new sparse::compressed;
        sparse::init(part,
                     (types::dim_t) state->partition_rows[partition_id],
                     (types::dim_t) state->cols,
                     (types::nnz_t) state->partition_nnz[partition_id],
                     (types::u32) state->partition_aux[partition_id]);
        if (codec == 0 || codec->family != series_codec_family_standard_csr) {
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

inline int materialize_blocked_ell_shard_pack(shard_storage *s, series_h5_state *state, unsigned long shard_id) {
    const std::uint64_t begin = state != 0 && state->shard_part_begin != 0 ? state->shard_part_begin[shard_id] : 0u;
    const std::uint64_t end = state != 0 && state->shard_part_end != 0 ? state->shard_part_end[shard_id] : 0u;
    const std::uint64_t partition_count = end >= begin ? (end - begin) : 0u;
    sparse::blocked_ell **parts = 0;
    char tmp_path[4096];
    char final_path[4096];
    int ok = 0;

    if (s == 0 || state == 0 || shard_id >= state->num_shards) return 0;
    if (!open_series_h5_backend(s) || !load_blocked_ell_shard_payload(state, shard_id)) return 0;
    if (partition_count != 0u) {
        parts = (sparse::blocked_ell **) std::calloc((std::size_t) partition_count, sizeof(sparse::blocked_ell *));
        if (parts == 0) return 0;
    }
    if (!prepare_blocked_ell_parts_from_state(state, (unsigned long) begin, (unsigned long) end, parts)) goto done;
    if (!fill_blocked_ell_parts_from_loaded_shard(state, shard_id, (unsigned long) begin, (unsigned long) end, parts)) goto done;
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

inline int materialize_shard_pack(shard_storage *s, series_h5_state *state, unsigned long shard_id) {
    if (state == 0) return 0;
    if (state->matrix_family == series_matrix_family_compressed) return materialize_compressed_shard_pack(s, state, shard_id);
    if (state->matrix_family == series_matrix_family_blocked_ell) return materialize_blocked_ell_shard_pack(s, state, shard_id);
    return 0;
}

inline void touch_shard_locked(series_h5_state *state, unsigned long shard_id) {
    if (state == 0 || shard_id >= state->num_shards) return;
    state->shard_access_count[shard_id] += 1u;
    state->shard_last_access_tick[shard_id] = ++state->access_clock;
    state->last_requested_shard = shard_id;
}

inline std::uint64_t shard_eviction_score(const series_h5_state *state,
                                          unsigned long shard_id,
                                          std::uint64_t now_tick) {
    const std::uint64_t age = now_tick >= state->shard_last_access_tick[shard_id] ? (now_tick - state->shard_last_access_tick[shard_id]) : 0u;
    const std::uint64_t accesses = state->shard_access_count[shard_id];
    return (age + 1u) * (state->shard_cache_bytes[shard_id] + 1u) / (accesses + 1u);
}

inline void evict_cached_shard_locked(series_h5_state *state, unsigned long shard_id) {
    if (state == 0 || shard_id >= state->num_shards) return;
    close_cached_shard_file(state, shard_id);
    if (state->shard_cache_paths != 0 && state->shard_cache_paths[shard_id] != 0) ::unlink(state->shard_cache_paths[shard_id]);
    if (state->shard_cache_state[shard_id] == series_cache_shard_ready) {
        if (state->cache_resident_bytes >= state->shard_cache_bytes[shard_id]) state->cache_resident_bytes -= state->shard_cache_bytes[shard_id];
        else state->cache_resident_bytes = 0u;
    }
    state->shard_cache_state[shard_id] = series_cache_shard_missing;
    state->shard_cache_bytes[shard_id] = 0u;
    state->shard_access_count[shard_id] = 0u;
    state->shard_last_access_tick[shard_id] = 0u;
    state->shard_pin_count[shard_id] = 0u;
}

inline void maybe_evict_cached_shards_locked(series_h5_state *state, unsigned long keep_shard_id) {
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
            if (state->shard_cache_state[shard_id] != series_cache_shard_ready) continue;
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
    series_h5_state *state = s != 0 ? (series_h5_state *) s->backend_state : 0;
    series_h5_cache_runtime *runtime = cache_runtime(state);
    if (state == 0 || runtime == 0) return;
    for (;;) {
        unsigned long shard_id = 0ul;
        {
            std::unique_lock<std::mutex> lock(runtime->state_mutex);
            runtime->state_cv.wait(lock, [&]() { return runtime->stop_requested || !runtime->shard_queue.empty(); });
            if (runtime->stop_requested) break;
            shard_id = runtime->shard_queue.front();
            runtime->shard_queue.pop_front();
            state->shard_cache_state[shard_id] = series_cache_shard_building;
        }

        const int ok = materialize_shard_pack(s, state, shard_id);

        {
            std::lock_guard<std::mutex> lock(runtime->state_mutex);
            if (ok) {
                if (state->shard_cache_state[shard_id] != series_cache_shard_ready) {
                    state->shard_cache_bytes[shard_id] = estimate_shard_pack_bytes(state, shard_id);
                    state->cache_resident_bytes += state->shard_cache_bytes[shard_id];
                }
                state->shard_cache_state[shard_id] = series_cache_shard_ready;
                touch_shard_locked(state, shard_id);
                maybe_evict_cached_shards_locked(state, shard_id);
            } else {
                state->shard_cache_state[shard_id] = series_cache_shard_failed;
            }
            runtime->state_cv.notify_all();
        }
    }
}

inline int ensure_cache_reader_started(shard_storage *s) {
    series_h5_state *state = 0;
    series_h5_cache_runtime *runtime = 0;
    if (s == 0 || s->backend_state == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    runtime = cache_runtime(state);
    if (runtime == 0) return 0;
    if (runtime->reader_started) return 1;
    runtime->stop_requested = false;
    runtime->reader_thread = std::thread(reader_materialize_loop, s);
    runtime->reader_started = true;
    return 1;
}

inline int ensure_cached_shard_ready(shard_storage *s, unsigned long shard_id) {
    series_h5_state *state = 0;
    series_h5_cache_runtime *runtime = 0;
    if (s == 0 || s->backend_state == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    if (!ensure_series_cache_layout(s)) return 0;
    runtime = cache_runtime(state);
    if (runtime == 0 || shard_id >= state->num_shards) return 0;
    if (!ensure_cache_reader_started(s)) return 0;

    {
        std::unique_lock<std::mutex> lock(runtime->state_mutex);
        if (state->shard_cache_state[shard_id] == series_cache_shard_ready) {
            touch_shard_locked(state, shard_id);
            return 1;
        }
        if (state->shard_cache_state[shard_id] == series_cache_shard_missing
            || state->shard_cache_state[shard_id] == series_cache_shard_failed) {
            state->shard_cache_state[shard_id] = series_cache_shard_queued;
            runtime->shard_queue.push_back(shard_id);
            runtime->state_cv.notify_all();
        }
        runtime->state_cv.wait(lock, [&]() {
            return state->shard_cache_state[shard_id] == series_cache_shard_ready
                || state->shard_cache_state[shard_id] == series_cache_shard_failed;
        });
        if (state->shard_cache_state[shard_id] != series_cache_shard_ready) return 0;
        touch_shard_locked(state, shard_id);
    }
    return 1;
}

int open_series_h5_backend(shard_storage *s) {
    series_h5_state *state = 0;
    if (s == 0 || s->source_path == 0 || s->backend_state == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    if (state->file >= 0) return 1;
    state->file = H5Fopen(s->source_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    return state->file >= 0;
}

void close_series_h5_backend(shard_storage *s) {
    series_h5_state *state = 0;
    if (s == 0 || s->backend_state == 0) return;
    state = (series_h5_state *) s->backend_state;
    series_h5_state_clear(state);
    std::free(state);
    s->backend_state = 0;
    s->open_backend = 0;
    s->close_backend = 0;
    s->backend = shard_storage_backend_none;
}

} // namespace

int create_series_compressed_h5(const char *filename,
                                const series_layout_view *layout,
                                const series_dataset_table_view *datasets,
                                const series_provenance_view *provenance) {
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
                     "cellshard: series column count exceeds the current u32 execution limit while writing %s (cols=%llu, limit=%llu)\n",
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
            ok = fail_series_u32_limit(filename, "part", i, "rows", layout->partition_rows[i], dim_limit);
            goto done;
        }
        if (layout->partition_nnz[i] > nnz_limit) {
            ok = fail_series_u32_limit(filename, "part", i, "nnz", layout->partition_nnz[i], nnz_limit);
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
        if (row_end - row_begin > dim_limit) warn_series_u32_limit(filename, "shard", shard_i, "rows", row_end - row_begin, dim_limit);
        if (shard_nnz > nnz_limit) warn_series_u32_limit(filename, "shard", shard_i, "nnz", shard_nnz, nnz_limit);
        shard_part_begin = part_end;
    }

    file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) goto done;
    if (!write_attr_string(file, "cellshard_magic", series_magic)) goto done;
    if (!write_attr_u32(file, "schema_version", series_h5_schema_version)) goto done;
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

int create_series_blocked_ell_h5(const char *filename,
                                 const series_layout_view *layout,
                                 const series_dataset_table_view *datasets,
                                 const series_provenance_view *provenance) {
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
                     "cellshard: series column count exceeds the current u32 execution limit while writing %s (cols=%llu, limit=%llu)\n",
                     filename,
                     (unsigned long long) layout->cols,
                     (unsigned long long) idx_limit);
        return 0;
    }

    partition_aux = (std::uint64_t *) std::calloc((std::size_t) layout->num_partitions, sizeof(std::uint64_t));
    partition_block_idx_offsets = (std::uint64_t *) std::calloc((std::size_t) layout->num_partitions, sizeof(std::uint64_t));
    partition_value_offsets = (std::uint64_t *) std::calloc((std::size_t) layout->num_partitions, sizeof(std::uint64_t));
    shard_block_idx_offsets = (std::uint64_t *) std::calloc((std::size_t) layout->num_shards + 1u, sizeof(std::uint64_t));
    shard_value_offsets = (std::uint64_t *) std::calloc((std::size_t) layout->num_shards + 1u, sizeof(std::uint64_t));
    if ((layout->num_partitions != 0) && (partition_aux == 0 || partition_block_idx_offsets == 0 || partition_value_offsets == 0)) goto done;
    if (layout->num_shards != 0 && (shard_block_idx_offsets == 0 || shard_value_offsets == 0)) goto done;

    for (i = 0; i < layout->num_partitions; ++i) {
        const std::uint64_t part_block_idx = (std::uint64_t) blocked_ell_part_block_index_count(layout->partition_rows[i], layout->partition_aux[i]);
        const std::uint64_t part_values = (std::uint64_t) blocked_ell_part_value_count(layout->partition_rows[i], layout->partition_aux[i]);
        if (layout->partition_rows[i] > dim_limit) {
            ok = fail_series_u32_limit(filename, "part", i, "rows", layout->partition_rows[i], dim_limit);
            goto done;
        }
        if (layout->partition_nnz[i] > nnz_limit) {
            ok = fail_series_u32_limit(filename, "part", i, "nnz", layout->partition_nnz[i], nnz_limit);
            goto done;
        }
        if (part_block_idx > idx_limit) {
            ok = fail_series_u32_limit(filename, "part", i, "block_col_idx_count", part_block_idx, idx_limit);
            goto done;
        }
        if (part_values > nnz_limit) {
            ok = fail_series_u32_limit(filename, "part", i, "value_count", part_values, nnz_limit);
            goto done;
        }
        partition_aux[i] = layout->partition_aux[i];
        partition_block_idx_offsets[i] = total_block_idx;
        partition_value_offsets[i] = total_values;
        total_block_idx += part_block_idx;
        total_values += part_values;
    }

    file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) goto done;
    if (!write_attr_string(file, "cellshard_magic", series_magic)) goto done;
    if (!write_attr_u32(file, "schema_version", series_h5_schema_version)) goto done;
    if (!write_attr_string(file, "matrix_format", "blocked_ell")) goto done;
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
    payload = payload_root >= 0 ? create_group(payload_root, "blocked_ell") : (hid_t) -1;
    if (matrix < 0 || dsets < 0 || prov < 0 || codecs < 0 || payload_root < 0 || payload < 0) goto done;

    if (!write_dataset_1d(matrix, "partition_rows", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, layout->partition_rows)) goto done;
    if (!write_dataset_1d(matrix, "partition_nnz", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, layout->partition_nnz)) goto done;
    if (layout->partition_axes != 0) {
        if (!write_dataset_1d(matrix, "partition_axes", H5T_NATIVE_UINT32, (hsize_t) layout->num_partitions, layout->partition_axes)) goto done;
    }
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
            if (row_end - row_begin > dim_limit) warn_series_u32_limit(filename, "shard", shard_i, "rows", row_end - row_begin, dim_limit);
            if (shard_nnz > nnz_limit) warn_series_u32_limit(filename, "shard", shard_i, "nnz", shard_nnz, nnz_limit);
            if (shard_block_idx_offsets[shard_i + 1u] - shard_block_idx_offsets[shard_i] > idx_limit) {
                warn_series_u32_limit(filename,
                                      "shard",
                                      shard_i,
                                      "block_col_idx_count",
                                      shard_block_idx_offsets[shard_i + 1u] - shard_block_idx_offsets[shard_i],
                                      idx_limit);
            }
            if (shard_value_offsets[shard_i + 1u] - shard_value_offsets[shard_i] > nnz_limit) {
                warn_series_u32_limit(filename,
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

int append_series_embedded_metadata_h5(const char *filename,
                                       const series_embedded_metadata_view *metadata) {
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
        const series_metadata_table_view *view = metadata->tables + i;
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

int append_series_observation_metadata_h5(const char *filename,
                                          const series_observation_metadata_view *metadata) {
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
        const series_observation_metadata_column_view *view = metadata->columns + i;
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

        if (view->type == series_observation_metadata_type_text) {
            if (view->text_values.count != metadata->rows
                || !write_text_column(column, "values", &view->text_values)) {
                H5Gclose(column);
                goto done;
            }
        } else if (view->type == series_observation_metadata_type_float32) {
            if ((rows != 0u && view->float32_values == 0)
                || !write_dataset_1d(column, "values", H5T_NATIVE_FLOAT, rows, view->float32_values)) {
                H5Gclose(column);
                goto done;
            }
        } else if (view->type == series_observation_metadata_type_uint8) {
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

int append_series_browse_cache_h5(const char *filename,
                                  const series_browse_cache_view *browse) {
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

int append_series_execution_h5(const char *filename,
                               const series_execution_view *execution) {
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
    if (!write_attr_u32(root, "preferred_base_format", execution != 0 ? execution->preferred_base_format : series_execution_format_unknown)) goto done;

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
                              "shard_preferred_pair_ids",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->shard_count,
                              execution->shard_preferred_pair_ids)) goto done;
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

int bind_series_h5(shard_storage *s, const char *path) {
    std::size_t len = 0;
    char *copy = 0;
    series_h5_state *state = 0;

    if (s == 0) return 0;
    if (s->close_backend != 0) s->close_backend(s);
    std::free(s->source_path);
    s->source_path = 0;
    if (path == 0) return 1;

    len = std::strlen(path);
    copy = (char *) std::malloc(len + 1u);
    state = (series_h5_state *) std::calloc(1u, sizeof(series_h5_state));
    if (copy == 0 || state == 0) {
        std::free(copy);
        std::free(state);
        return 0;
    }
    std::memcpy(copy, path, len + 1u);
    series_h5_state_init(state);
    s->source_path = copy;
    s->backend = shard_storage_backend_series_h5;
    s->backend_state = state;
    s->open_backend = open_series_h5_backend;
    s->close_backend = close_series_h5_backend;
    return 1;
}

int bind_series_h5_cache(shard_storage *s, const char *cache_root) {
    series_h5_state *state = 0;

    if (s == 0 || s->backend != shard_storage_backend_series_h5 || s->backend_state == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    if (state->cache_root != 0 && cache_root != 0 && std::strcmp(state->cache_root, cache_root) != 0) {
        invalidate_series_h5_cache(s);
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

int set_series_h5_cache_budget_bytes(shard_storage *s, std::uint64_t bytes) {
    series_h5_state *state = 0;
    series_h5_cache_runtime *runtime = 0;
    if (s == 0 || s->backend != shard_storage_backend_series_h5 || s->backend_state == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    if (!ensure_series_cache_layout(s)) return 0;
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

int set_series_h5_cache_predictor_enabled(shard_storage *s, int enabled) {
    series_h5_state *state = 0;
    if (s == 0 || s->backend != shard_storage_backend_series_h5 || s->backend_state == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    state->predictor_enabled = enabled != 0 ? 1 : 0;
    return 1;
}

int pin_series_h5_cache_shard(shard_storage *s, unsigned long shard_id) {
    series_h5_state *state = 0;
    series_h5_cache_runtime *runtime = 0;
    if (s == 0 || s->backend != shard_storage_backend_series_h5 || s->backend_state == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    if (!ensure_cached_shard_ready(s, shard_id)) return 0;
    runtime = cache_runtime(state);
    if (runtime == 0) return 0;
    {
        std::lock_guard<std::mutex> lock(runtime->state_mutex);
        state->shard_pin_count[shard_id] += 1u;
    }
    return 1;
}

int unpin_series_h5_cache_shard(shard_storage *s, unsigned long shard_id) {
    series_h5_state *state = 0;
    series_h5_cache_runtime *runtime = 0;
    if (s == 0 || s->backend != shard_storage_backend_series_h5 || s->backend_state == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    if (!ensure_series_cache_layout(s) || shard_id >= state->num_shards) return 0;
    runtime = cache_runtime(state);
    if (runtime == 0) return 0;
    {
        std::lock_guard<std::mutex> lock(runtime->state_mutex);
        if (state->shard_pin_count[shard_id] != 0u) state->shard_pin_count[shard_id] -= 1u;
        maybe_evict_cached_shards_locked(state, (unsigned long) state->num_shards);
    }
    return 1;
}

int evict_series_h5_cache_shard(shard_storage *s, unsigned long shard_id) {
    series_h5_state *state = 0;
    series_h5_cache_runtime *runtime = 0;
    if (s == 0 || s->backend != shard_storage_backend_series_h5 || s->backend_state == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    if (!ensure_series_cache_layout(s) || shard_id >= state->num_shards) return 0;
    runtime = cache_runtime(state);
    if (runtime == 0) return 0;
    {
        std::lock_guard<std::mutex> lock(runtime->state_mutex);
        evict_cached_shard_locked(state, shard_id);
    }
    return 1;
}

int invalidate_series_h5_cache(shard_storage *s) {
    series_h5_state *state = 0;
    series_h5_cache_runtime *runtime = 0;
    unsigned long shard_id = 0ul;
    if (s == 0 || s->backend != shard_storage_backend_series_h5 || s->backend_state == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    if (!ensure_series_cache_layout(s)) return 0;
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

int load_series_compressed_h5_header(const char *filename,
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
        series_h5_state *state = 0;
        if (!bind_series_h5(s, filename)) goto done;
        state = (series_h5_state *) s->backend_state;
        state->rows = rows;
        state->cols = cols;
        state->nnz = nnz;
        state->num_partitions = num_partitions;
        state->num_shards = num_shards;
        state->num_codecs = (std::uint32_t) num_codecs;
        state->matrix_family = series_matrix_family_compressed;
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
            state->codecs = (series_codec_descriptor *) std::calloc((std::size_t) num_codecs, sizeof(series_codec_descriptor));
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

int load_series_blocked_ell_h5_header(const char *filename,
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
        series_h5_state *state = 0;
        if (!bind_series_h5(s, filename)) goto done;
        state = (series_h5_state *) s->backend_state;
        state->rows = rows;
        state->cols = cols;
        state->nnz = nnz;
        state->num_partitions = num_partitions;
        state->num_shards = num_shards;
        state->num_codecs = (std::uint32_t) num_codecs;
        state->matrix_family = series_matrix_family_blocked_ell;
        if (num_partitions != 0) {
            state->partition_rows = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_nnz = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_aux = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_row_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions + 1u, sizeof(std::uint64_t));
            state->partition_block_idx_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_value_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
            state->partition_codec_ids = (std::uint32_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint32_t));
            if (state->partition_rows == 0
                || state->partition_nnz == 0
                || state->partition_aux == 0
                || state->partition_row_offsets == 0
                || state->partition_block_idx_offsets == 0
                || state->partition_value_offsets == 0
                || state->partition_codec_ids == 0) goto done;
            std::memcpy(state->partition_rows, partition_rows, (std::size_t) num_partitions * sizeof(std::uint64_t));
            std::memcpy(state->partition_nnz, partition_nnz, (std::size_t) num_partitions * sizeof(std::uint64_t));
            std::memcpy(state->partition_aux, partition_aux, (std::size_t) num_partitions * sizeof(std::uint64_t));
            std::memcpy(state->partition_row_offsets, partition_row_offsets, ((std::size_t) num_partitions + 1u) * sizeof(std::uint64_t));
        }
        state->shard_offsets = (std::uint64_t *) std::calloc((std::size_t) num_shards + 1u, sizeof(std::uint64_t));
        if (state->shard_offsets == 0 && (num_shards + 1u) != 0u) goto done;
        if (state->shard_offsets != 0) std::memcpy(state->shard_offsets, shard_offsets, ((std::size_t) num_shards + 1u) * sizeof(std::uint64_t));
        state->shard_block_idx_offsets = (std::uint64_t *) std::calloc((std::size_t) num_shards + 1u, sizeof(std::uint64_t));
        state->shard_value_offsets = (std::uint64_t *) std::calloc((std::size_t) num_shards + 1u, sizeof(std::uint64_t));
        if (state->shard_block_idx_offsets == 0 || state->shard_value_offsets == 0) goto done;
        if (num_codecs != 0) {
            state->codecs = (series_codec_descriptor *) std::calloc((std::size_t) num_codecs, sizeof(series_codec_descriptor));
            if (state->codecs == 0) goto done;
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
        if (!read_dataset_1d(matrix, "partition_codec_ids", H5T_NATIVE_UINT32, state->partition_codec_ids)) goto done;
        if (!load_codec_table(codecs, state->codecs, (std::uint32_t) num_codecs)) goto done;
        if (!build_shard_partition_spans(state)) goto done;
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

int fetch_series_compressed_h5_partition(sharded<sparse::compressed> *m,
                                    const shard_storage *s,
                                    unsigned long partition_id) {
    shard_storage *storage = const_cast<shard_storage *>(s);
    series_h5_state *state = 0;
    if (m == 0 || storage == 0 || storage->backend != shard_storage_backend_series_h5 || partition_id >= m->num_partitions || storage->backend_state == 0) return 0;
    state = (series_h5_state *) storage->backend_state;
    if (!ensure_cached_shard_ready(storage, (unsigned long) state->partition_shard_ids[partition_id])) return 0;
    return load_compressed_part_from_cached_pack(m, state, partition_id);
}

int fetch_series_compressed_h5_shard(sharded<sparse::compressed> *m,
                                     const shard_storage *s,
                                     unsigned long shard_id) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    series_h5_state *state = 0;

    if (m == 0 || s == 0 || s->backend_state == 0 || shard_id >= m->num_shards) return 0;
    state = (series_h5_state *) s->backend_state;
    if (!ensure_cached_shard_ready(const_cast<shard_storage *>(s), shard_id)) return 0;
    begin = first_partition_in_shard(m, shard_id);
    end = last_partition_in_shard(m, shard_id);
    for (i = begin; i < end; ++i) {
        if (!load_compressed_part_from_cached_pack(m, state, i)) return 0;
    }
    return 1;
}

int prefetch_series_compressed_h5_partition_cache(const sharded<sparse::compressed> *m,
                                             shard_storage *s,
                                             unsigned long partition_id) {
    series_h5_state *state = 0;
    if (m == 0 || s == 0 || s->backend != shard_storage_backend_series_h5 || partition_id >= m->num_partitions || s->backend_state == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    return ensure_cached_shard_ready(s, (unsigned long) state->partition_shard_ids[partition_id]);
}

int prefetch_series_compressed_h5_shard_cache(const sharded<sparse::compressed> *m,
                                              shard_storage *s,
                                              unsigned long shard_id) {
    if (m == 0 || s == 0 || shard_id >= m->num_shards) return 0;
    return ensure_cached_shard_ready(s, shard_id);
}

int fetch_series_blocked_ell_h5_partition(sharded<sparse::blocked_ell> *m,
                                     const shard_storage *s,
                                     unsigned long partition_id) {
    shard_storage *storage = const_cast<shard_storage *>(s);
    series_h5_state *state = 0;
    if (m == 0 || storage == 0 || storage->backend != shard_storage_backend_series_h5 || partition_id >= m->num_partitions || storage->backend_state == 0) return 0;
    state = (series_h5_state *) storage->backend_state;
    if (!ensure_cached_shard_ready(storage, (unsigned long) state->partition_shard_ids[partition_id])) return 0;
    return load_blocked_ell_part_from_cached_pack(m, state, partition_id);
}

int fetch_series_blocked_ell_h5_shard(sharded<sparse::blocked_ell> *m,
                                      const shard_storage *s,
                                      unsigned long shard_id) {
    const unsigned long begin = first_partition_in_shard(m, shard_id);
    const unsigned long end = last_partition_in_shard(m, shard_id);
    unsigned long i = 0;
    series_h5_state *state = 0;

    if (m == 0 || s == 0 || shard_id >= m->num_shards || s->backend_state == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    if (!ensure_cached_shard_ready(const_cast<shard_storage *>(s), shard_id)) return 0;
    for (i = begin; i < end; ++i) {
        if (!load_blocked_ell_part_from_cached_pack(m, state, i)) return 0;
    }
    return 1;
}

int prefetch_series_blocked_ell_h5_partition_cache(const sharded<sparse::blocked_ell> *m,
                                              shard_storage *s,
                                              unsigned long partition_id) {
    series_h5_state *state = 0;
    if (m == 0 || s == 0 || s->backend != shard_storage_backend_series_h5 || partition_id >= m->num_partitions || s->backend_state == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    return ensure_cached_shard_ready(s, (unsigned long) state->partition_shard_ids[partition_id]);
}

int prefetch_series_blocked_ell_h5_shard_cache(const sharded<sparse::blocked_ell> *m,
                                               shard_storage *s,
                                               unsigned long shard_id) {
    if (m == 0 || s == 0 || shard_id >= m->num_shards) return 0;
    return ensure_cached_shard_ready(s, shard_id);
}

} // namespace cellshard
