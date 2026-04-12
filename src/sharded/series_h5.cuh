#pragma once

#include "../formats/compressed.cuh"
#include "../real.cuh"
#include "shard_paths.cuh"
#include "sharded.cuh"

#include <cstdint>

namespace cellshard {

enum {
    series_h5_schema_version = 1u
};

// Codec families describe how one stored part payload should be interpreted
// after the lightweight series metadata has already been loaded.
enum {
    series_codec_family_none = 0u,
    series_codec_family_standard_csr = 1u,
    series_codec_family_quantized_csr = 2u
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

    std::uint32_t part_count;
    std::uint32_t sample_rows_per_part;
    const std::uint32_t *part_sample_row_offsets;
    const std::uint64_t *part_sample_global_rows;
    const float *part_sample_values;
};

struct series_layout_view {
    std::uint64_t rows;
    std::uint64_t cols;
    std::uint64_t nnz;
    std::uint64_t num_parts;
    std::uint64_t num_shards;

    const std::uint64_t *part_rows;
    const std::uint64_t *part_nnz;
    const std::uint32_t *part_axes;
    const std::uint64_t *part_row_offsets;
    const std::uint32_t *part_dataset_ids;
    const std::uint32_t *part_codec_ids;
    const std::uint64_t *shard_offsets;

    const series_codec_descriptor *codecs;
    std::uint32_t num_codecs;
};

// Create/append helpers are whole-file synchronous HDF5 operations.
int create_series_compressed_h5(const char *filename,
                                const series_layout_view *layout,
                                const series_dataset_table_view *datasets,
                                const series_provenance_view *provenance);

int append_series_embedded_metadata_h5(const char *filename,
                                       const series_embedded_metadata_view *metadata);

int append_series_browse_cache_h5(const char *filename,
                                  const series_browse_cache_view *browse);

int append_standard_csr_part_h5(const char *filename,
                                unsigned long part_id,
                                const sparse::compressed *part);

// Header load binds a lazy shard_storage backend; fetch/prefetch calls are the
// points that actually materialize part payloads or populate local caches.
int bind_series_h5(shard_storage *s, const char *path);
int bind_series_h5_part_cache(shard_storage *s, const char *cache_dir);
int load_series_compressed_h5_header(const char *filename,
                                     sharded<sparse::compressed> *m,
                                     shard_storage *s);
int prefetch_series_compressed_h5_part_to_cache(const sharded<sparse::compressed> *m,
                                                const shard_storage *s,
                                                unsigned long part_id);
int prefetch_series_compressed_h5_shard_to_cache(const sharded<sparse::compressed> *m,
                                                 const shard_storage *s,
                                                 unsigned long shard_id);
int fetch_series_compressed_h5_part(sharded<sparse::compressed> *m,
                                    const shard_storage *s,
                                    unsigned long part_id);
int fetch_series_compressed_h5_shard(sharded<sparse::compressed> *m,
                                     const shard_storage *s,
                                     unsigned long shard_id);

} // namespace cellshard
