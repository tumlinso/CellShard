#pragma once

#include <CellShard/runtime/v2/atom_source.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::runtime_v2 {

struct read_span {
    storage_object_id object{};
    std::uint64_t object_offset = 0;
    std::uint64_t bytes = 0;
    std::uint64_t staging_offset = 0;
};

struct read_copy {
    std::uint32_t span_index = 0;
    std::uint64_t span_offset = 0;
    std::uint64_t destination_offset = 0;
    std::uint64_t bytes = 0;
};

struct read_plan {
    array_view<read_span> spans{};
    array_view<read_copy> copies{};
    std::uint64_t staging_bytes = 0;
    std::uint64_t requested_bytes = 0;
};

[[nodiscard]] status_code build_read_plan(
    array_view<atom_range> sorted_ranges, std::uint64_t maximum_gap_bytes,
    std::uint64_t maximum_span_bytes, read_span *span_storage,
    std::size_t span_capacity, read_copy *copy_storage,
    std::size_t copy_capacity, read_plan *out) noexcept;

static_assert(std::is_trivially_copyable_v<read_span>);
static_assert(std::is_trivially_copyable_v<read_copy>);
static_assert(std::is_trivially_copyable_v<read_plan>);

} // namespace cellshard::runtime_v2
