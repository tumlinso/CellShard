#pragma once

#include <CellShard/artifact/atom_store/format_v1.hh>
#include <CellShard/artifact/atom_store/identity_v1.hh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::artifact::atom_store {

inline constexpr std::uint64_t arena_header_bytes_v1 = 256;
inline constexpr std::uint64_t arena_directory_entry_bytes_v1 = 96;
inline constexpr std::uint64_t arena_min_alignment_v1 = 64;

enum class arena_section_kind_v1 : std::uint32_t {
    atom_dictionary = 1,
    domain_order_dictionary = 2,
    coverage_index = 3,
    dependency_index = 4,
    payload_descriptors = 5,
    payload_bytes = 6,
    lowering_records = 7,
    provenance = 8,
};

enum arena_section_flags_v1 : std::uint32_t {
    arena_section_required_v1 = 1u << 0,
    arena_section_optional_v1 = 1u << 1,
};

struct arena_header_v1 {
    std::array<std::byte, 8> magic = file_magic_v1;
    std::uint32_t schema_version = schema_version_v1;
    std::uint32_t endian_marker = endian_marker_v1;
    std::uint64_t header_bytes = arena_header_bytes_v1;
    std::uint64_t total_bytes = 0;
    semantic_identity_v1 store_identity{};
    semantic_identity_v1 catalog_identity{};
    semantic_identity_v1 structure_identity{};
    semantic_identity_v1 certification_identity{};
    std::uint64_t section_directory_offset = 0;
    std::uint64_t section_count = 0;
    std::uint64_t section_directory_bytes = 0;
    content_digest_v1 content_digest{};
    std::array<std::byte, 96> reserved{};
};

struct arena_directory_entry_v1 {
    arena_section_kind_v1 kind{};
    std::uint32_t flags = 0;
    std::uint64_t alignment = 0;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
    std::uint64_t record_bytes = 0;
    std::uint64_t record_count = 0;
    content_digest_v1 content_digest{};
    std::array<std::byte, 8> reserved{};
};

[[nodiscard]] constexpr bool is_power_of_two_v1(std::uint64_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

[[nodiscard]] constexpr bool valid_arena_header_shape_v1(const arena_header_v1 &header) noexcept {
    for (std::size_t index = 0; index < file_magic_v1.size(); ++index) {
        if (header.magic[index] != file_magic_v1[index]) return false;
    }
    if (header.schema_version != schema_version_v1
        || header.endian_marker != endian_marker_v1 || header.header_bytes != arena_header_bytes_v1
        || header.total_bytes < header.header_bytes || header.section_count == 0
        || header.section_directory_offset < header.header_bytes
        || (header.section_directory_offset % arena_min_alignment_v1) != 0) return false;
    if (header.section_count > UINT64_MAX / arena_directory_entry_bytes_v1) return false;
    const auto expected_bytes = header.section_count * arena_directory_entry_bytes_v1;
    return header.section_directory_bytes == expected_bytes
        && header.section_directory_offset <= header.total_bytes
        && expected_bytes <= header.total_bytes - header.section_directory_offset;
}

[[nodiscard]] constexpr bool valid_arena_directory_entry_shape_v1(
    const arena_directory_entry_v1 &entry, std::uint64_t total_bytes) noexcept {
    const auto known_flags = arena_section_required_v1 | arena_section_optional_v1;
    if ((entry.flags & ~known_flags) != 0
        || (((entry.flags & arena_section_required_v1) != 0)
            == ((entry.flags & arena_section_optional_v1) != 0))
        || !is_power_of_two_v1(entry.alignment) || entry.alignment < arena_min_alignment_v1
        || (entry.offset % entry.alignment) != 0 || entry.offset > total_bytes
        || entry.bytes > total_bytes - entry.offset) return false;
    if (entry.record_bytes == 0) return entry.record_count == 0;
    return entry.record_count <= entry.bytes / entry.record_bytes
        && entry.record_count * entry.record_bytes == entry.bytes;
}

static_assert(sizeof(arena_header_v1) == arena_header_bytes_v1);
static_assert(sizeof(arena_directory_entry_v1) == arena_directory_entry_bytes_v1);
static_assert(std::is_trivially_copyable<arena_header_v1>::value);
static_assert(std::is_trivially_copyable<arena_directory_entry_v1>::value);

} // namespace cellshard::artifact::atom_store
