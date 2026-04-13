#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cellshard::exporting {

struct series_dataset_summary {
    std::string dataset_id;
    std::string matrix_path;
    std::string feature_path;
    std::string barcode_path;
    std::string metadata_path;
    std::uint32_t format = 0u;
    std::uint64_t row_begin = 0u;
    std::uint64_t row_end = 0u;
    std::uint64_t rows = 0u;
    std::uint64_t cols = 0u;
    std::uint64_t nnz = 0u;
};

struct series_codec_summary {
    std::uint32_t codec_id = 0u;
    std::uint32_t family = 0u;
    std::uint32_t value_code = 0u;
    std::uint32_t scale_value_code = 0u;
    std::uint32_t bits = 0u;
    std::uint32_t flags = 0u;
};

struct series_partition_summary {
    std::uint64_t partition_id = 0u;
    std::uint64_t row_begin = 0u;
    std::uint64_t row_end = 0u;
    std::uint64_t rows = 0u;
    std::uint64_t nnz = 0u;
    std::uint64_t aux = 0u;
    std::uint32_t dataset_id = 0u;
    std::uint32_t axis = 0u;
    std::uint32_t codec_id = 0u;
};

struct series_shard_summary {
    std::uint64_t shard_id = 0u;
    std::uint64_t partition_begin = 0u;
    std::uint64_t partition_end = 0u;
    std::uint64_t row_begin = 0u;
    std::uint64_t row_end = 0u;
};

struct observation_metadata_column {
    std::string name;
    std::uint32_t type = 0u;
    std::vector<std::string> text_values;
    std::vector<float> float32_values;
    std::vector<std::uint8_t> uint8_values;
};

struct series_summary {
    std::string path;
    std::string matrix_format;
    std::string payload_layout;
    std::uint64_t rows = 0u;
    std::uint64_t cols = 0u;
    std::uint64_t nnz = 0u;
    std::uint64_t num_partitions = 0u;
    std::uint64_t num_shards = 0u;
    std::uint64_t num_datasets = 0u;
    std::vector<series_dataset_summary> datasets;
    std::vector<series_partition_summary> partitions;
    std::vector<series_shard_summary> shards;
    std::vector<series_codec_summary> codecs;
    std::vector<std::string> obs_names;
    std::vector<std::string> var_ids;
    std::vector<std::string> var_names;
    std::vector<std::string> var_types;
};

struct csr_matrix_export {
    std::uint64_t rows = 0u;
    std::uint64_t cols = 0u;
    std::vector<std::int64_t> indptr;
    std::vector<std::int64_t> indices;
    std::vector<float> data;
};

struct anndata_export {
    series_summary summary;
    std::vector<observation_metadata_column> obs_columns;
    csr_matrix_export x;
};

bool load_series_summary(const char *path, series_summary *out, std::string *error = nullptr);
bool load_series_as_csr(const char *path, csr_matrix_export *out, std::string *error = nullptr);
bool load_series_for_anndata(const char *path, anndata_export *out, std::string *error = nullptr);

} // namespace cellshard::exporting
