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

enum {
    series_codec_family_none = 0u,
    series_codec_family_standard_csr = 1u,
    series_codec_family_microscaled_csr = 2u
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

int create_series_compressed_h5(const char *filename,
                                const series_layout_view *layout,
                                const series_dataset_table_view *datasets,
                                const series_provenance_view *provenance);

int append_standard_csr_part_h5(const char *filename,
                                unsigned long part_id,
                                const sparse::compressed *part);

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
