#pragma once

#include "../export/dataset_export.hh"
#include "cshard/spec.hh"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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

private:
    std::string path_;
    spec::header header_{};
    std::uint32_t canonical_layout_ = spec::matrix_layout_unknown;
    std::vector<section_record> sections_;
    std::vector<spec::matrix_descriptor> matrices_;
    table_view obs_;
    table_view var_;
    bool has_pack_manifest_ = false;
};

struct writer_options {
    std::uint64_t feature_order_hash = 0u;
};

bool write_csr(const std::string &path,
               const exporting::csr_matrix_export &csr,
               const table_view &obs,
               const table_view &var,
               const writer_options &options,
               std::string *error = nullptr);

bool convert_csh5_to_cshard(const std::string &input_path,
                            const std::string &output_path,
                            const writer_options &options = {},
                            std::string *error = nullptr);

const char *layout_name(std::uint32_t layout) noexcept;

} // namespace cellshard::cshard
