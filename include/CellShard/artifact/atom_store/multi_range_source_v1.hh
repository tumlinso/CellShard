#pragma once
#include <CellShard/artifact/atom_store/frame_extent_map_v1.hh>
#include <cstddef>
#include <cstdint>
namespace cellshard::artifact::atom_store {
enum class range_read_status_v1 : std::uint32_t { success, invalid_mapping, insufficient_output, short_read, digest_mismatch };
using exact_range_read_fn_v1 = bool (*)(void *, cellshard::storage_object_id, cellshard::extent_id, std::uint64_t, std::byte *, std::size_t);
[[nodiscard]] range_read_status_v1 read_exact_atom_frame_v1(const atom_frame_map_record_v1 &frame, const frame_extent_slice_v1 *slices, std::size_t slice_count, const content_digest_v1 &expected, exact_range_read_fn_v1 read, void *context, std::byte *output, std::size_t output_bytes) noexcept;
}
