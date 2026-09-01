#include <CellShard/compiler/composition/coverage_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<std::uint64_t, 3> left_items{{1, 5, 9}};
    const std::array<std::uint64_t, 3> right_items{{2, 6, (std::uint64_t{1} << 40u)}};
    const composition::exact_coverage_view_v1 left{
        cellshard::structure_id{1}, cellshard::domain_id{2},
        cellshard::order_id{3}, left_items.data(), left_items.size()};
    const composition::exact_coverage_view_v1 right{
        cellshard::structure_id{4}, cellshard::domain_id{2},
        cellshard::order_id{3}, right_items.data(), right_items.size()};
    std::array<std::uint64_t, 6> merged{};
    composition::exact_coverage_view_v1 output{};
    assert(composition::compose_disjoint_union_v1(
               cellshard::structure_id{5}, left, right,
               {merged.data(), merged.size()}, &output).composed());
    assert((merged == std::array<std::uint64_t, 6>{
                          1, 2, 5, 6, 9, (std::uint64_t{1} << 40u)}));
    assert(output.logical_item_count == 6);

    const std::array<std::uint64_t, 2> overlap_items{{5, 7}};
    auto overlap = right;
    overlap.logical_item_ids = overlap_items.data();
    overlap.logical_item_count = overlap_items.size();
    assert(composition::compose_disjoint_union_v1(
               cellshard::structure_id{5}, left, overlap,
               {merged.data(), merged.size()}, &output).code
           == composition::coverage_composition_code_v1::overlapping_inputs);

    auto wrong_order = right;
    wrong_order.order = cellshard::order_id{8};
    assert(composition::compose_disjoint_union_v1(
               cellshard::structure_id{5}, left, wrong_order,
               {merged.data(), merged.size()}, &output).code
           == composition::coverage_composition_code_v1::input_order_mismatch);
}
