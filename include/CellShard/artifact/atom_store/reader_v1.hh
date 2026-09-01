#pragma once
#include <CellShard/artifact/atom_store/arena_v1.hh>
#include <cstddef>
#include <cstdint>
namespace cellshard::artifact::atom_store {
enum class reader_status_v1 : std::uint32_t { success, short_metadata, invalid_header, invalid_directory, insufficient_entries };
struct metadata_inspection_v1 { arena_header_v1 header{}; std::size_t section_count = 0; bool payload_verification_required = true; };
[[nodiscard]] reader_status_v1 inspect_atom_store_metadata_v1(const std::byte *metadata, std::size_t metadata_bytes, arena_directory_entry_v1 *entries, std::size_t entry_capacity, metadata_inspection_v1 *out) noexcept;
}
