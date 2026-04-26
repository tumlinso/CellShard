#pragma once

#include "../common/fingerprints.hh"
#include "../common/generation.hh"
#include "../common/layout.hh"
#include "../common/partition.hh"

#include <cstdint>

namespace cellshard::cshard::spec {

inline constexpr std::uint64_t magic = 0x3130445241485343ull; // "CSHARD01"
inline constexpr std::uint32_t version_major = 1u;
inline constexpr std::uint32_t version_minor = 0u;
inline constexpr std::uint32_t endian_tag = 0x01020304u;
inline constexpr std::uint64_t payload_alignment = 64u;

enum section_kind : std::uint32_t {
    section_kind_unknown = 0u,
    section_kind_metadata = 1u,
    section_kind_section_directory = 2u,
    section_kind_matrix_directory = 3u,
    section_kind_obs_table_directory = 4u,
    section_kind_var_table_directory = 5u,
    section_kind_provenance_directory = 6u,
    section_kind_pack_manifest_directory = 7u,
    section_kind_payload = 8u,
    section_kind_blocked_ell_block_col_idx = 16u,
    section_kind_blocked_ell_values = 17u,
    section_kind_sliced_ell_slice_row_offsets = 18u,
    section_kind_sliced_ell_slice_widths = 19u,
    section_kind_sliced_ell_col_idx = 20u,
    section_kind_sliced_ell_values = 21u,
    section_kind_csr_row_ptr = 22u,
    section_kind_csr_col_idx = 23u,
    section_kind_csr_values = 24u,
    section_kind_table_string_offsets = 32u,
    section_kind_table_string_bytes = 33u,
    section_kind_table_float32_values = 34u,
    section_kind_table_uint8_values = 35u
};

enum dtype : std::uint32_t {
    dtype_unknown = 0u,
    dtype_u8 = 1u,
    dtype_u16 = 2u,
    dtype_u32 = 3u,
    dtype_u64 = 4u,
    dtype_i64 = 5u,
    dtype_f16 = 6u,
    dtype_f32 = 7u
};

enum matrix_layout : std::uint32_t {
    matrix_layout_unknown = 0u,
    matrix_layout_blocked_ell = 1u,
    matrix_layout_sliced_ell = 2u,
    matrix_layout_csr = 3u
};

enum table_column_type : std::uint32_t {
    table_column_unknown = 0u,
    table_column_text = 1u,
    table_column_float32 = 2u,
    table_column_uint8 = 3u
};

enum provenance_kind : std::uint32_t {
    provenance_kind_unknown = 0u,
    provenance_kind_source_dataset = 1u,
    provenance_kind_preprocess = 2u,
    provenance_kind_external_reference = 3u
};

#pragma pack(push, 1)

struct header {
    std::uint64_t magic;
    std::uint32_t version_major;
    std::uint32_t version_minor;
    std::uint32_t endian_tag;
    std::uint32_t header_bytes;
    std::uint64_t file_size;
    std::uint64_t section_directory_offset;
    std::uint32_t section_directory_count;
    std::uint32_t metadata_count;
    std::uint64_t metadata_offset;
    std::uint64_t matrix_directory_offset;
    std::uint32_t matrix_directory_count;
    std::uint32_t obs_table_column_count;
    std::uint64_t obs_table_directory_offset;
    std::uint32_t var_table_column_count;
    std::uint64_t var_table_directory_offset;
    std::uint64_t provenance_directory_offset;
    std::uint32_t provenance_count;
    std::uint32_t pack_manifest_count;
    std::uint64_t pack_manifest_offset;
    std::uint64_t payload_offset;
    std::uint64_t payload_bytes;
    std::uint64_t rows;
    std::uint64_t cols;
    std::uint64_t nnz;
    std::uint64_t feature_order_hash;
    std::uint32_t canonical_layout;
    std::uint32_t flags;
    std::uint32_t reserved32;
    std::uint64_t reserved[10];
};

struct metadata_record {
    char key[64];
    char value[192];
};

struct section_entry {
    std::uint32_t kind;
    std::uint32_t id;
    std::uint64_t offset;
    std::uint64_t bytes;
    std::uint64_t element_count;
    std::uint32_t dtype;
    std::uint32_t alignment;
    std::uint32_t flags;
    std::uint32_t reserved;
};

struct matrix_descriptor {
    std::uint32_t matrix_id;
    std::uint32_t partition_id;
    std::uint32_t layout;
    std::uint32_t value_dtype;
    std::uint32_t index_dtype;
    std::uint32_t reserved0;
    std::uint64_t row_begin;
    std::uint64_t rows;
    std::uint64_t cols;
    std::uint64_t nnz;
    std::uint32_t block_size;
    std::uint32_t ell_width;
    std::uint32_t slice_count;
    std::uint32_t slice_rows;
    std::uint64_t aux0;
    std::uint64_t aux1;
    std::uint32_t section_a_id;
    std::uint32_t section_b_id;
    std::uint32_t section_c_id;
    std::uint32_t section_d_id;
    std::uint64_t reserved[4];
};

struct table_column_descriptor {
    char name[64];
    std::uint32_t column_type;
    std::uint32_t row_count;
    std::uint32_t data_dtype;
    std::uint32_t reserved0;
    std::uint32_t offsets_section_id;
    std::uint32_t data_section_id;
    std::uint64_t reserved[4];
};

struct provenance_descriptor {
    std::uint32_t kind;
    std::uint32_t section_id;
    std::uint64_t reserved[4];
};

struct pack_manifest_descriptor {
    dataset_generation_ref generation;
    std::uint32_t path_offsets_section_id;
    std::uint32_t path_bytes_section_id;
    std::uint64_t reserved[4];
};

#pragma pack(pop)

static_assert(sizeof(header) == 256, "cshard v1 header must stay fixed-width");

} // namespace cellshard::cshard::spec
