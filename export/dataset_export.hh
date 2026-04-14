#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cellshard::exporting {

struct source_dataset_summary {
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

struct dataset_codec_summary {
    std::uint32_t codec_id = 0u;
    std::uint32_t family = 0u;
    std::uint32_t value_code = 0u;
    std::uint32_t scale_value_code = 0u;
    std::uint32_t bits = 0u;
    std::uint32_t flags = 0u;
};

struct dataset_partition_summary {
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

struct dataset_shard_summary {
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

struct embedded_metadata_table {
    std::uint32_t dataset_index = 0u;
    std::uint64_t row_begin = 0u;
    std::uint64_t row_end = 0u;
    std::uint32_t rows = 0u;
    std::uint32_t cols = 0u;
    std::vector<std::string> column_names;
    std::vector<std::string> field_values;
    std::vector<std::uint32_t> row_offsets;
};

struct execution_partition_metadata {
    std::uint64_t partition_id = 0u;
    std::uint64_t row_begin = 0u;
    std::uint64_t row_end = 0u;
    std::uint64_t rows = 0u;
    std::uint64_t nnz = 0u;
    std::uint64_t aux = 0u;
    std::uint32_t dataset_id = 0u;
    std::uint32_t axis = 0u;
    std::uint32_t codec_id = 0u;
    std::uint32_t execution_format = 0u;
    std::uint32_t blocked_ell_block_size = 0u;
    std::uint32_t blocked_ell_bucket_count = 0u;
    float blocked_ell_fill_ratio = 0.0f;
    std::uint64_t execution_bytes = 0u;
    std::uint64_t blocked_ell_bytes = 0u;
    std::uint64_t bucketed_blocked_ell_bytes = 0u;
};

struct execution_shard_metadata {
    std::uint64_t shard_id = 0u;
    std::uint64_t partition_begin = 0u;
    std::uint64_t partition_end = 0u;
    std::uint64_t row_begin = 0u;
    std::uint64_t row_end = 0u;
    std::uint32_t execution_format = 0u;
    std::uint32_t blocked_ell_block_size = 0u;
    std::uint32_t bucketed_partition_count = 0u;
    std::uint32_t bucketed_segment_count = 0u;
    float blocked_ell_fill_ratio = 0.0f;
    std::uint64_t execution_bytes = 0u;
    std::uint64_t bucketed_blocked_ell_bytes = 0u;
    std::uint32_t preferred_pair = 0u;
    std::uint32_t owner_node_id = 0u;
    std::uint32_t owner_rank_id = 0u;
};

struct runtime_service_metadata {
    std::uint32_t service_mode = 0u;
    std::uint32_t live_write_mode = 0u;
    std::uint32_t prefer_pack_delivery = 0u;
    std::uint32_t remote_pack_delivery = 0u;
    std::uint32_t single_reader_coordinator = 0u;
    std::uint32_t maintenance_lock_blocks_overwrite = 0u;
    std::uint64_t canonical_generation = 0u;
    std::uint64_t execution_plan_generation = 0u;
    std::uint64_t pack_generation = 0u;
    std::uint64_t service_epoch = 0u;
    std::uint64_t active_read_generation = 0u;
    std::uint64_t staged_write_generation = 0u;
};

struct client_snapshot_ref {
    std::uint64_t snapshot_id = 0u;
    std::uint64_t canonical_generation = 0u;
    std::uint64_t execution_plan_generation = 0u;
    std::uint64_t pack_generation = 0u;
    std::uint64_t service_epoch = 0u;
};

struct dataset_summary {
    std::string path;
    std::string matrix_format;
    std::string payload_layout;
    std::uint64_t rows = 0u;
    std::uint64_t cols = 0u;
    std::uint64_t nnz = 0u;
    std::uint64_t num_partitions = 0u;
    std::uint64_t num_shards = 0u;
    std::uint64_t num_datasets = 0u;
    std::vector<source_dataset_summary> datasets;
    std::vector<dataset_partition_summary> partitions;
    std::vector<dataset_shard_summary> shards;
    std::vector<dataset_codec_summary> codecs;
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
    dataset_summary summary;
    std::vector<observation_metadata_column> obs_columns;
    csr_matrix_export x;
};

struct global_metadata_snapshot {
    std::uint64_t snapshot_id = 0u;
    dataset_summary summary;
    std::vector<embedded_metadata_table> embedded_metadata;
    std::uint64_t observation_metadata_rows = 0u;
    std::vector<observation_metadata_column> observation_metadata;
    std::vector<execution_partition_metadata> execution_partitions;
    std::vector<execution_shard_metadata> execution_shards;
    runtime_service_metadata runtime_service;
};

bool load_dataset_summary(const char *path, dataset_summary *out, std::string *error = nullptr);
bool load_dataset_as_csr(const char *path, csr_matrix_export *out, std::string *error = nullptr);
bool load_dataset_rows_as_csr(const char *path,
                              const std::uint64_t *row_indices,
                              std::size_t row_count,
                              csr_matrix_export *out,
                              std::string *error = nullptr);
bool load_dataset_for_anndata(const char *path, anndata_export *out, std::string *error = nullptr);
bool load_dataset_global_metadata_snapshot(const char *path, global_metadata_snapshot *out, std::string *error = nullptr);
client_snapshot_ref make_client_snapshot_ref(const global_metadata_snapshot &snapshot);
bool validate_client_snapshot_ref(const global_metadata_snapshot &owner_snapshot,
                                  const client_snapshot_ref &request,
                                  std::string *error = nullptr);
bool serialize_global_metadata_snapshot(const global_metadata_snapshot &snapshot,
                                        std::vector<std::uint8_t> *out,
                                        std::string *error = nullptr);
bool deserialize_global_metadata_snapshot(const void *data,
                                          std::size_t bytes,
                                          global_metadata_snapshot *out,
                                          std::string *error = nullptr);

} // namespace cellshard::exporting
