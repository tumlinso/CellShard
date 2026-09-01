#include <CellShard/compiler/composition/coverage_difference_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<std::uint64_t, 5> left_items{{1, 5, 9, 12, (std::uint64_t{1} << 46u)}};
    const std::array<std::uint64_t, 4> right_items{{2, 5, 12, (std::uint64_t{1} << 46u)}};
    const composition::exact_coverage_view_v1 left{
        cellshard::structure_id{1}, cellshard::domain_id{2},
        cellshard::order_id{3}, left_items.data(), left_items.size()};
    const composition::exact_coverage_view_v1 right{
        cellshard::structure_id{4}, cellshard::domain_id{2},
        cellshard::order_id{3}, right_items.data(), right_items.size()};
    std::array<std::uint64_t, 5> items{};
    composition::exact_coverage_view_v1 output{};
    assert(composition::compose_coverage_difference_v1(
               cellshard::structure_id{5}, left, right,
               {items.data(), items.size()}, &output).composed());
    assert(output.logical_item_count == 2);
    assert(items[0] == 1 && items[1] == 9);

    assert(composition::compose_coverage_difference_v1(
               cellshard::structure_id{6}, left, left,
               {items.data(), items.size()}, &output).composed());
    assert(output.logical_item_count == 0);
    assert(composition::validate_exact_coverage_v1(output).composed());

    auto wrong_order = right;
    wrong_order.order = cellshard::order_id{9};
    assert(composition::compose_coverage_difference_v1(
               cellshard::structure_id{5}, left, wrong_order,
               {items.data(), items.size()}, &output).code
           == composition::coverage_composition_code_v1::input_order_mismatch);
}
