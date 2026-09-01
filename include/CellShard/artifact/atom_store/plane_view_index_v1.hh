#pragma once
#include <CellShard/artifact/atom_store/identity_v1.hh>
#include <cstdint>
#include <limits>
#include <type_traits>
namespace cellshard::artifact::atom_store {
struct plane_index_record_v1 {
    semantic_identity_v1 atom{};
    std::uint64_t plane_identity = 0;
    std::uint64_t value_generation = 0;
    std::uint64_t scalar_encoding_identity = 0;
    std::uint64_t element_count = 0;
};
struct physical_view_index_record_v1 {
    std::uint64_t plane_identity = 0;
    materialization_identity_v1 materialization{};
    std::uint64_t target_identity = 0;
    std::uint64_t byte_offset = 0;
    std::uint64_t byte_count = 0;
    std::uint32_t alignment = 0;
    std::uint32_t reserved = 0;
    content_digest_v1 content{};
};
[[nodiscard]] constexpr bool view_digest_nonzero_v1(const content_digest_v1 &d) noexcept {
    for (auto byte : d.bytes) if (byte != std::byte{0}) return true;
    return false;
}
[[nodiscard]] constexpr bool valid_plane_index_record_v1(const plane_index_record_v1 &r) noexcept {
    return r.atom.valid() && r.plane_identity != 0 && r.value_generation != 0
        && r.scalar_encoding_identity != 0 && r.element_count != 0;
}
[[nodiscard]] constexpr bool valid_physical_view_index_record_v1(
    const physical_view_index_record_v1 &r) noexcept {
    return r.plane_identity != 0 && r.materialization.valid() && r.target_identity != 0
        && r.byte_count != 0 && r.byte_offset <= std::numeric_limits<std::uint64_t>::max() - r.byte_count
        && r.alignment != 0 && (r.alignment & (r.alignment - 1u)) == 0
        && r.byte_offset % r.alignment == 0 && valid_content_digest_v1(r.content)
        && view_digest_nonzero_v1(r.content);
}
static_assert(std::is_trivially_copyable<plane_index_record_v1>::value);
static_assert(std::is_trivially_copyable<physical_view_index_record_v1>::value);
} // namespace cellshard::artifact::atom_store
