#include <CellShard/compiler/composition/support_union_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<std::uint64_t, 4> left_items{{1, 5, 9, 12}};
    const std::array<std::uint64_t, 4> right_items{{2, 5, 12, (std::uint64_t{1} << 45u)}};
    const composition::exact_coverage_view_v1 left{
        cellshard::structure_id{1}, cellshard::domain_id{2},
        cellshard::order_id{3}, left_items.data(), left_items.size()};
    const composition::exact_coverage_view_v1 right{
        cellshard::structure_id{4}, cellshard::domain_id{2},
        cellshard::order_id{3}, right_items.data(), right_items.size()};
    std::array<std::uint64_t, 8> items{};
    composition::exact_coverage_view_v1 output{};
    const auto result = composition::compose_sparse_support_union_v1(
        cellshard::structure_id{5}, left, right,
        {items.data(), items.size()}, &output);
    assert(result.composed());
    assert(output.logical_item_count == 6);
    assert((std::array<std::uint64_t, 6>{
                items[0], items[1], items[2], items[3], items[4], items[5]}
            == std::array<std::uint64_t, 6>{
                1, 2, 5, 9, 12, (std::uint64_t{1} << 45u)}));

    auto wrong_domain = right;
    wrong_domain.domain = cellshard::domain_id{7};
    assert(composition::compose_sparse_support_union_v1(
               cellshard::structure_id{5}, left, wrong_domain,
               {items.data(), items.size()}, &output).code
           == composition::coverage_composition_code_v1::
                  input_domain_mismatch);
}
