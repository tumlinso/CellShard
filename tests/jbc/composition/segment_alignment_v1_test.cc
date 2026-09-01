#include <CellShard/compiler/composition/segment_alignment_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<composition::exact_segment_v1, 3> segments{{
        {2, 10}, {9, 20}, {(std::uint64_t{1} << 50u), 30}}};
    const composition::segment_partition_view_v1 left{
        cellshard::structure_id{1}, cellshard::domain_id{2},
        cellshard::order_id{3}, segments.data(), segments.size()};
    const composition::segment_partition_view_v1 right{
        cellshard::structure_id{4}, cellshard::domain_id{2},
        cellshard::order_id{5}, segments.data(), segments.size()};
    std::array<composition::segment_alignment_entry_v1, 3> entries{};
    composition::segment_alignment_view_v1 output{};
    assert(composition::compose_segment_alignment_v1(
               cellshard::structure_id{6}, left, right, entries.data(),
               entries.size(), &output).aligned());
    assert(entries[2].segment_identity == (std::uint64_t{1} << 50u));
    assert(output.left_order != output.right_order);

    auto different_segments = segments;
    different_segments[1].logical_item_count = 21;
    auto mismatch = right;
    mismatch.segments = different_segments.data();
    assert(composition::compose_segment_alignment_v1(
               cellshard::structure_id{6}, left, mismatch, entries.data(),
               entries.size(), &output).code
           == composition::segment_alignment_code_v1::segment_extent_mismatch);
}
