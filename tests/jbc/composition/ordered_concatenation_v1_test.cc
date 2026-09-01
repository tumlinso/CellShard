#include <CellShard/compiler/composition/ordered_concatenation_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<std::uint64_t, 3> left_items{{1, 5, 9}};
    const std::array<std::uint64_t, 2> right_items{{2, (std::uint64_t{1} << 44u)}};
    const composition::exact_coverage_view_v1 left{
        cellshard::structure_id{1}, cellshard::domain_id{2},
        cellshard::order_id{3}, left_items.data(), left_items.size()};
    const composition::exact_coverage_view_v1 right{
        cellshard::structure_id{4}, cellshard::domain_id{2},
        cellshard::order_id{5}, right_items.data(), right_items.size()};
    std::array<std::uint64_t, 5> items{};
    std::array<std::uint64_t, 3> offsets{};
    composition::ordered_concatenation_view_v1 output{};
    assert(composition::compose_ordered_concatenation_v1(
               cellshard::structure_id{6}, cellshard::order_id{7}, left, right,
               {items.data(), items.size(), offsets.data(), offsets.size()},
               &output).composed());
    assert((items == std::array<std::uint64_t, 5>{
                         1, 5, 9, 2, (std::uint64_t{1} << 44u)}));
    assert((offsets == std::array<std::uint64_t, 3>{0, 3, 5}));
    assert(output.order == cellshard::order_id{7});

    const std::array<std::uint64_t, 2> overlap_items{{5, 8}};
    auto overlap = right;
    overlap.logical_item_ids = overlap_items.data();
    assert(composition::compose_ordered_concatenation_v1(
               cellshard::structure_id{6}, cellshard::order_id{7}, left,
               overlap,
               {items.data(), items.size(), offsets.data(), offsets.size()},
               &output).code
           == composition::ordered_concatenation_code_v1::overlapping_inputs);
}
