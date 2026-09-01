#include <CellShard/compiler/composition/overlay_application_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<std::uint64_t, 5> base_ids{{1, 5, 9, 12, 20}};
    const std::array<std::uint64_t, 2> addition_ids{{3, (std::uint64_t{1} << 52u)}};
    const std::array<std::uint64_t, 2> removal_ids{{5, 12}};
    const auto view = [](cellshard::structure_id identity,
                          const auto &ids) {
        return composition::exact_coverage_view_v1{
            identity, cellshard::domain_id{2}, cellshard::order_id{3},
            ids.data(), ids.size()};
    };
    const auto base = view(cellshard::structure_id{1}, base_ids);
    const auto additions = view(cellshard::structure_id{4}, addition_ids);
    const auto removals = view(cellshard::structure_id{5}, removal_ids);
    std::array<std::uint64_t, 7> storage{};
    composition::exact_coverage_view_v1 output{};
    assert(composition::compose_overlay_application_v1(
               cellshard::structure_id{6}, base, additions, removals,
               {storage.data(), storage.size()}, &output).applied());
    assert(output.logical_item_count == 5);
    assert((std::array<std::uint64_t, 5>{storage[0], storage[1], storage[2],
                                        storage[3], storage[4]}
            == std::array<std::uint64_t, 5>{
                1, 3, 9, 20, (std::uint64_t{1} << 52u)}));

    const std::array<std::uint64_t, 1> missing_removal_ids{{7}};
    const auto missing = view(cellshard::structure_id{7}, missing_removal_ids);
    assert(composition::compose_overlay_application_v1(
               cellshard::structure_id{6}, base, additions, missing,
               {storage.data(), storage.size()}, &output).code
           == composition::overlay_application_code_v1::removal_not_in_base);
}
