#pragma once

#include <CellShard/artifact/atom_store/identity_v1.hh>
#include <CellShard/identity.hh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::artifact::atom_store {

struct atom_frame_map_record_v1 {
    semantic_identity_v1 atom{};
    std::uint64_t frame_index = 0;
    std::uint64_t frame_count = 0;
    std::uint64_t logical_offset = 0;
    std::uint64_t logical_bytes = 0;
    std::uint64_t first_extent_slice = 0;
    std::uint64_t extent_slice_count = 0;
};

struct frame_extent_slice_v1 {
    semantic_identity_v1 atom{};
    std::uint64_t frame_index = 0;
    cellshard::storage_object_id object{};
    cellshard::extent_id extent{};
    std::uint64_t extent_offset = 0;
    std::uint64_t frame_offset = 0;
    std::uint64_t bytes = 0;
};

[[nodiscard]] constexpr bool valid_atom_frame_map_record_v1(
    const atom_frame_map_record_v1 &record) noexcept {
    return record.atom.valid() && record.frame_count != 0
        && record.frame_index < record.frame_count && record.logical_bytes != 0
        && record.logical_offset <= std::numeric_limits<std::uint64_t>::max()
            - record.logical_bytes
        && record.extent_slice_count != 0
        && record.first_extent_slice <= std::numeric_limits<std::uint64_t>::max()
            - record.extent_slice_count;
}

[[nodiscard]] constexpr bool valid_frame_extent_slice_v1(
    const frame_extent_slice_v1 &slice) noexcept {
    return slice.atom.valid() && slice.object.valid() && slice.extent.valid()
        && slice.bytes != 0
        && slice.extent_offset <= std::numeric_limits<std::uint64_t>::max() - slice.bytes
        && slice.frame_offset <= std::numeric_limits<std::uint64_t>::max() - slice.bytes;
}

// Slices are serialized in increasing frame_offset order and must cover the
// frame exactly, with no gap or overlap.
[[nodiscard]] constexpr bool frame_extent_slices_cover_v1(
    const atom_frame_map_record_v1 &frame,
    const frame_extent_slice_v1 *slices,
    std::size_t slice_count) noexcept {
    if (!valid_atom_frame_map_record_v1(frame) || slices == nullptr
        || slice_count != frame.extent_slice_count) return false;
    std::uint64_t covered = 0;
    for (std::size_t index = 0; index < slice_count; ++index) {
        const auto &slice = slices[index];
        if (!valid_frame_extent_slice_v1(slice) || !(slice.atom == frame.atom)
            || slice.frame_index != frame.frame_index || slice.frame_offset != covered)
            return false;
        covered += slice.bytes;
    }
    return covered == frame.logical_bytes;
}

static_assert(std::is_trivially_copyable<atom_frame_map_record_v1>::value);
static_assert(std::is_trivially_copyable<frame_extent_slice_v1>::value);

} // namespace cellshard::artifact::atom_store
