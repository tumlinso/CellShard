#include <CellShard/compiler/composition/coverage_intersection_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<std::uint64_t, 5> left_items{{1, 5, 9, 12, (std::uint64_t{1} << 43u)}};
    const std::array<std::uint64_t, 4> right_items{{2, 5, 12, (std::uint64_t{1} << 43u)}};
    const composition::exact_coverage_view_v1 left{
        cellshard::structure_id{1}, cellshard::domain_id{2},
        cellshard::order_id{3}, left_items.data(), left_items.size()};
    const composition::exact_coverage_view_v1 right{
        cellshard::structure_id{4}, cellshard::domain_id{2},
        cellshard::order_id{3}, right_items.data(), right_items.size()};
    std::array<std::uint64_t, 4> items{};
    composition::exact_coverage_view_v1 output{};
    assert(composition::compose_coverage_intersection_v1(
               cellshard::structure_id{5}, left, right,
               {items.data(), items.size()}, &output).composed());
    assert(output.logical_item_count == 3);
    assert(items[0] == 5 && items[1] == 12
           && items[2] == (std::uint64_t{1} << 43u));

    const std::array<std::uint64_t, 1> disjoint_items{{99}};
    auto disjoint = right;
    disjoint.logical_item_ids = disjoint_items.data();
    disjoint.logical_item_count = disjoint_items.size();
    assert(composition::compose_coverage_intersection_v1(
               cellshard::structure_id{6}, left, disjoint,
               {items.data(), items.size()}, &output).composed());
    assert(output.logical_item_count == 0);
    assert(composition::validate_exact_coverage_v1(output).composed());
}
