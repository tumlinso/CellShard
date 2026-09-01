#pragma once
#include <CellShard/artifact/atom_store/arena_v1.hh>
#include <cstddef>
#include <cstdint>
namespace cellshard::artifact::atom_store {
struct writer_section_source_v1 {
    arena_section_kind_v1 kind{}; std::uint32_t flags = 0; std::uint64_t alignment = 0;
    const std::byte *data = nullptr; std::uint64_t bytes = 0;
    std::uint64_t record_bytes = 0; std::uint64_t record_count = 0;
};
struct writer_requirements_v1 { std::uint64_t total_bytes = 0; std::uint64_t directory_offset = 0; std::uint64_t directory_bytes = 0; };
enum class writer_status_v1 : std::uint32_t { success, invalid_input, overflow, insufficient_capacity };
[[nodiscard]] writer_status_v1 atom_store_writer_requirements_v1(const writer_section_source_v1 *sections, std::size_t count, writer_requirements_v1 *out) noexcept;
[[nodiscard]] writer_status_v1 fill_atom_store_v1(const arena_header_v1 &identity_header, const writer_section_source_v1 *sections, std::size_t count, std::byte *output, std::size_t capacity) noexcept;
}
