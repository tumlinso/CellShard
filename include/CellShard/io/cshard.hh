#pragma once

#include "../export/dataset_export.hh"
#include "multi_assay.hh"
#include "cshard/spec.hh"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cellshard {
struct bucketed_blocked_ell_shard;
struct bucketed_sliced_ell_shard;
}

namespace cellshard::cshard {

struct table_view {
    struct column {
        std::string name;
        std::uint32_t type = spec::table_column_unknown;
        std::vector<std::string> text_values;
        std::vector<float> float32_values;
        std::vector<std::uint8_t> uint8_values;
    };

    std::uint64_t rows = 0u;
    std::vector<column> columns;

    table_view head(std::size_t count) const;
};

struct description {
    std::string path;
    std::uint32_t version_major = 0u;
    std::uint32_t version_minor = 0u;
    std::uint64_t rows = 0u;
    std::uint64_t cols = 0u;
    std::uint64_t nnz = 0u;
    std::uint64_t partitions = 0u;
    std::uint64_t feature_order_hash = 0u;
    std::string canonical_layout;
    bool has_pack_manifest = false;
};

struct assay_description {
    std::string assay_id;
    dataset_assay_semantics semantics{};
    std::uint64_t global_observation_count = 0u;
    std::uint64_t rows = 0u;
    std::uint64_t cols = 0u;
    std::uint64_t nnz = 0u;
    std::uint64_t feature_order_hash = 0u;
    std::uint32_t matrix_descriptor_begin = 0u;
    std::uint32_t matrix_descriptor_count = 0u;
    table_view feature_table;
    std::vector<std::uint32_t> global_to_assay_row;
    std::vector<std::uint32_t> assay_row_to_global;
};

struct multi_assay_description {
    std::uint64_t global_observation_count = 0u;
    std::uint32_t pairing_kind = dataset_pairing_none;
    table_view global_obs;
    std::vector<assay_description> assays;
};

struct paired_rows {
    std::uint32_t global_observation = 0u;
    std::vector<std::uint32_t> assay_rows;
};

class cshard_file {
public:
    struct section_record {
        spec::section_entry entry{};
    };

    cshard_file() = default;
    static cshard_file open(const std::string &path);
    static bool validate(const std::string &path, std::string *error = nullptr);

    const std::string &path() const noexcept { return path_; }
    description describe() const;
    std::uint32_t canonical_matrix() const noexcept { return canonical_layout_; }
    exporting::csr_matrix_export read_rows(std::uint64_t start, std::uint64_t count) const;
    table_view obs() const { return obs_; }
    table_view var() const { return var_; }
    bool has_pack_manifest() const noexcept { return has_pack_manifest_; }
    bool is_multi_assay() const noexcept { return !assays_.empty(); }
    std::uint32_t assay_count() const noexcept { return (std::uint32_t) assays_.size(); }
    const assay_description& assay(std::uint32_t index) const;
    const assay_description* find_assay(const std::string &assay_id) const;
    paired_rows resolve_paired_rows(std::uint32_t global_observation) const;
    exporting::csr_matrix_export read_assay_rows(const std::string &assay_id,
                                                 std::uint64_t start,
                                                 std::uint64_t count) const;
    multi_assay_description multi_assay() const;

private:
    std::string path_;
    spec::header header_{};
    std::uint32_t canonical_layout_ = spec::matrix_layout_unknown;
    std::vector<section_record> sections_;
    std::vector<spec::matrix_descriptor> matrices_;
    table_view obs_;
    table_view var_;
    std::uint32_t pairing_kind_ = dataset_pairing_none;
    std::uint64_t global_observation_count_ = 0u;
    std::vector<assay_description> assays_;
    bool has_pack_manifest_ = false;
};

struct writer_options {
    std::uint64_t feature_order_hash = 0u;
};

struct csr_assay_input {
    std::string assay_id;
    dataset_assay_semantics semantics{};
    exporting::csr_matrix_export matrix;
    table_view features;
    std::vector<std::uint32_t> global_to_assay_row;
    std::vector<std::uint32_t> assay_row_to_global;
    std::uint64_t feature_order_hash = 0u;
};

struct optimized_blocked_ell_shard_input {
    std::string assay_id;
    std::uint64_t global_row_begin = 0u;
    std::uint64_t global_row_end = 0u;
    std::uint64_t local_row_begin = 0u;
    std::uint64_t local_row_end = 0u;
    const bucketed_blocked_ell_shard *shard = nullptr;
    const void *serialized_blob = nullptr;
    std::size_t serialized_blob_bytes = 0u;
};

struct optimized_sliced_ell_shard_input {
    std::string assay_id;
    std::uint64_t global_row_begin = 0u;
    std::uint64_t global_row_end = 0u;
    std::uint64_t local_row_begin = 0u;
    std::uint64_t local_row_end = 0u;
    const bucketed_sliced_ell_shard *shard = nullptr;
    const void *serialized_blob = nullptr;
    std::size_t serialized_blob_bytes = 0u;
};

struct optimized_blocked_ell_assay_input {
    std::string assay_id;
    dataset_assay_semantics semantics{};
    table_view features;
    std::vector<std::uint32_t> global_to_assay_row;
    std::vector<std::uint32_t> assay_row_to_global;
    std::vector<optimized_blocked_ell_shard_input> shards;
    std::uint64_t feature_order_hash = 0u;
};

struct optimized_sliced_ell_assay_input {
    std::string assay_id;
    dataset_assay_semantics semantics{};
    table_view features;
    std::vector<std::uint32_t> global_to_assay_row;
    std::vector<std::uint32_t> assay_row_to_global;
    std::vector<optimized_sliced_ell_shard_input> shards;
    std::uint64_t feature_order_hash = 0u;
};

bool write_csr(const std::string &path,
               const exporting::csr_matrix_export &csr,
               const table_view &obs,
               const table_view &var,
               const writer_options &options,
               std::string *error = nullptr);

bool write_multi_assay_csr(const std::string &path,
                           const table_view &global_obs,
                           const std::vector<csr_assay_input> &assays,
                           std::uint32_t pairing_kind,
                           const writer_options &options,
                           std::string *error = nullptr);

bool write_multi_assay_optimized_blocked_ell(const std::string &path,
                                             const table_view &global_obs,
                                             const std::vector<optimized_blocked_ell_assay_input> &assays,
                                             std::uint32_t pairing_kind,
                                             const writer_options &options,
                                             std::string *error = nullptr);

bool write_multi_assay_optimized_sliced_ell(const std::string &path,
                                            const table_view &global_obs,
                                            const std::vector<optimized_sliced_ell_assay_input> &assays,
                                            std::uint32_t pairing_kind,
                                            const writer_options &options,
                                            std::string *error = nullptr);

bool convert_csh5_to_cshard(const std::string &input_path,
                            const std::string &output_path,
                            const writer_options &options = {},
                            std::string *error = nullptr);

const char *layout_name(std::uint32_t layout) noexcept;

} // namespace cellshard::cshard
