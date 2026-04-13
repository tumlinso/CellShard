#pragma once

#include "../formats/compressed.cuh"
#include "../formats/blocked_ell.cuh"
#include "../real.cuh"
#include "shard_paths.cuh"
#include "sharded.cuh"

#include <cstdint>

namespace cellshard {

enum {
    series_h5_schema_version = 2u
};

// Codec families describe how one stored part payload should be interpreted
// after the lightweight series metadata has already been loaded.
enum {
    series_codec_family_none = 0u,
    series_codec_family_standard_csr = 1u,
    series_codec_family_quantized_csr = 2u,
    series_codec_family_blocked_ell = 3u
};

enum {
    series_execution_format_unknown = 0u,
    series_execution_format_compressed = 1u,
    series_execution_format_blocked_ell = 2u,
    series_execution_format_mixed = 3u
};

struct series_codec_descriptor {
    std::uint32_t codec_id;
    std::uint32_t family;
    std::uint32_t value_code;
    std::uint32_t scale_value_code;
    std::uint32_t bits;
    std::uint32_t flags;
};

struct series_text_column_view {
    std::uint32_t count;
    std::uint32_t bytes;
    const std::uint32_t *offsets;
    const char *data;
};

struct series_dataset_table_view {
    std::uint32_t count;
    series_text_column_view dataset_ids;
    series_text_column_view matrix_paths;
    series_text_column_view feature_paths;
    series_text_column_view barcode_paths;
    series_text_column_view metadata_paths;
    const std::uint32_t *formats;
    const std::uint64_t *row_begin;
    const std::uint64_t *row_end;
    const std::uint64_t *rows;
    const std::uint64_t *cols;
    const std::uint64_t *nnz;
};

// Provenance tables are metadata-only views used at file-build time. They can
// be large on the host, but they are not part of the steady-state fetch path.
struct series_provenance_view {
    series_text_column_view global_barcodes;
    const std::uint32_t *cell_dataset_ids;
    const std::uint64_t *cell_local_indices;

    series_text_column_view feature_ids;
    series_text_column_view feature_names;
    series_text_column_view feature_types;
    const std::uint32_t *feature_dataset_ids;
    const std::uint64_t *feature_local_indices;

    const std::uint64_t *dataset_feature_offsets;
    const std::uint32_t *dataset_feature_to_global;
};

struct series_metadata_table_view {
    std::uint32_t rows;
    std::uint32_t cols;
    series_text_column_view column_names;
    series_text_column_view field_values;
    const std::uint32_t *row_offsets;
};

struct series_embedded_metadata_view {
    std::uint32_t count;
    const std::uint32_t *dataset_indices;
    const std::uint64_t *global_row_begin;
    const std::uint64_t *global_row_end;
    const series_metadata_table_view *tables;
};

enum {
    series_observation_metadata_type_none = 0u,
    series_observation_metadata_type_text = 1u,
    series_observation_metadata_type_float32 = 2u,
    series_observation_metadata_type_uint8 = 3u
};

struct series_observation_metadata_column_view {
    const char *name;
    std::uint32_t type;
    series_text_column_view text_values;
    const float *float32_values;
    const std::uint8_t *uint8_values;
};

struct series_observation_metadata_view {
    std::uint64_t rows;
    std::uint32_t cols;
    const series_observation_metadata_column_view *columns;
};

struct series_browse_cache_view {
    std::uint32_t selected_feature_count;
    const std::uint32_t *selected_feature_indices;
    const float *gene_sum;
    const float *gene_detected;
    const float *gene_sq_sum;

    std::uint32_t dataset_count;
    const float *dataset_feature_mean;

    std::uint32_t shard_count;
    const float *shard_feature_mean;

    std::uint32_t partition_count;
    std::uint32_t sample_rows_per_partition;
    const std::uint32_t *partition_sample_row_offsets;
    const std::uint64_t *partition_sample_global_rows;
    const float *partition_sample_values;
};

struct series_layout_view {
    std::uint64_t rows;
    std::uint64_t cols;
    std::uint64_t nnz;
    std::uint64_t num_partitions;
    std::uint64_t num_shards;

    const std::uint64_t *partition_rows;
    const std::uint64_t *partition_nnz;
    const std::uint32_t *partition_axes;
    const std::uint64_t *partition_aux;
    const std::uint64_t *partition_row_offsets;
    const std::uint32_t *partition_dataset_ids;
    const std::uint32_t *partition_codec_ids;
    const std::uint64_t *shard_offsets;

    const series_codec_descriptor *codecs;
    std::uint32_t num_codecs;
};

struct series_execution_view {
    std::uint32_t partition_count;
    const std::uint32_t *partition_execution_formats;
    const std::uint32_t *partition_blocked_ell_block_sizes;
    const float *partition_blocked_ell_fill_ratios;
    const std::uint64_t *partition_execution_bytes;
    const std::uint64_t *partition_blocked_ell_bytes;

    std::uint32_t shard_count;
    const std::uint32_t *shard_execution_formats;
    const std::uint32_t *shard_blocked_ell_block_sizes;
    const float *shard_blocked_ell_fill_ratios;
    const std::uint64_t *shard_execution_bytes;
    const std::uint32_t *shard_preferred_pair_ids;

    std::uint32_t preferred_base_format;
};

// Create/append helpers are whole-file synchronous HDF5 operations.
int create_series_compressed_h5(const char *filename,
                                const series_layout_view *layout,
                                const series_dataset_table_view *datasets,
                                const series_provenance_view *provenance);
int create_series_blocked_ell_h5(const char *filename,
                                 const series_layout_view *layout,
                                 const series_dataset_table_view *datasets,
                                 const series_provenance_view *provenance);

int append_series_embedded_metadata_h5(const char *filename,
                                       const series_embedded_metadata_view *metadata);

int append_series_observation_metadata_h5(const char *filename,
                                          const series_observation_metadata_view *metadata);

int append_series_browse_cache_h5(const char *filename,
                                  const series_browse_cache_view *browse);

int append_series_execution_h5(const char *filename,
                               const series_execution_view *execution);

int append_standard_csr_partition_h5(const char *filename,
                                unsigned long partition_id,
                                const sparse::compressed *part);
int append_blocked_ell_partition_h5(const char *filename,
                               unsigned long partition_id,
                               const sparse::blocked_ell *part);

// Header load binds a lazy shard_storage backend; fetch/prefetch calls are the
// points that actually materialize partition payloads or populate local caches.
int bind_series_h5(shard_storage *s, const char *path);
int bind_series_h5_cache(shard_storage *s, const char *cache_root);
int set_series_h5_cache_budget_bytes(shard_storage *s, std::uint64_t bytes);
int set_series_h5_cache_predictor_enabled(shard_storage *s, int enabled);
int pin_series_h5_cache_shard(shard_storage *s, unsigned long shard_id);
int unpin_series_h5_cache_shard(shard_storage *s, unsigned long shard_id);
int evict_series_h5_cache_shard(shard_storage *s, unsigned long shard_id);
int invalidate_series_h5_cache(shard_storage *s);
int load_series_compressed_h5_header(const char *filename,
                                     sharded<sparse::compressed> *m,
                                     shard_storage *s);
int load_series_blocked_ell_h5_header(const char *filename,
                                      sharded<sparse::blocked_ell> *m,
                                      shard_storage *s);
int prefetch_series_compressed_h5_partition_cache(const sharded<sparse::compressed> *m,
                                             shard_storage *s,
                                             unsigned long partition_id);
int prefetch_series_compressed_h5_shard_cache(const sharded<sparse::compressed> *m,
                                              shard_storage *s,
                                              unsigned long shard_id);
int fetch_series_compressed_h5_partition(sharded<sparse::compressed> *m,
                                    const shard_storage *s,
                                    unsigned long partition_id);
int fetch_series_compressed_h5_shard(sharded<sparse::compressed> *m,
                                     const shard_storage *s,
                                     unsigned long shard_id);
int prefetch_series_blocked_ell_h5_partition_cache(const sharded<sparse::blocked_ell> *m,
                                              shard_storage *s,
                                              unsigned long partition_id);
int prefetch_series_blocked_ell_h5_shard_cache(const sharded<sparse::blocked_ell> *m,
                                               shard_storage *s,
                                               unsigned long shard_id);
int fetch_series_blocked_ell_h5_partition(sharded<sparse::blocked_ell> *m,
                                     const shard_storage *s,
                                     unsigned long partition_id);
int fetch_series_blocked_ell_h5_shard(sharded<sparse::blocked_ell> *m,
                                      const shard_storage *s,
                                      unsigned long shard_id);

// Temporary compatibility wrappers for repo-internal callers while the new
// cache manager surface propagates through the tree.
inline int bind_series_h5_partition_cache(shard_storage *s, const char *cache_dir) {
    return bind_series_h5_cache(s, cache_dir);
}

inline int prefetch_series_compressed_h5_partition_to_cache(const sharded<sparse::compressed> *m,
                                                       const shard_storage *s,
                                                       unsigned long partition_id) {
    return prefetch_series_compressed_h5_partition_cache(m, const_cast<shard_storage *>(s), partition_id);
}

inline int prefetch_series_compressed_h5_shard_to_cache(const sharded<sparse::compressed> *m,
                                                        const shard_storage *s,
                                                        unsigned long shard_id) {
    return prefetch_series_compressed_h5_shard_cache(m, const_cast<shard_storage *>(s), shard_id);
}

inline int prefetch_series_blocked_ell_h5_partition_to_cache(const sharded<sparse::blocked_ell> *m,
                                                        const shard_storage *s,
                                                        unsigned long partition_id) {
    return prefetch_series_blocked_ell_h5_partition_cache(m, const_cast<shard_storage *>(s), partition_id);
}

inline int prefetch_series_blocked_ell_h5_shard_to_cache(const sharded<sparse::blocked_ell> *m,
                                                         const shard_storage *s,
                                                         unsigned long shard_id) {
    return prefetch_series_blocked_ell_h5_shard_cache(m, const_cast<shard_storage *>(s), shard_id);
}

} // namespace cellshard
