#include <CellShard/compiler/composition/prefix_composition_v1.hh>

#include <array>
#include <cassert>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<composition::prefix_sequence_entry_v1, 3> prefix_entries{{
        {2, 1, 0}, {7, 0, 0}, {(std::uint64_t{1} << 55u), 2, 0}}};
    const std::array<composition::prefix_sequence_entry_v1, 5> full_entries{{
        {2, 1, 0}, {7, 0, 0}, {9, 4, 0}, {12, 3, 0},
        {(std::uint64_t{1} << 55u), 2, 0}}};
    const composition::prefix_sequence_view_v1 prefix{
        cellshard::structure_id{1}, cellshard::domain_id{2},
        cellshard::order_id{3}, composition::prefix_sequence_kind_v1::trajectory,
        {}, prefix_entries.data(), prefix_entries.size(), 0};
    const composition::prefix_sequence_view_v1 full{
        cellshard::structure_id{4}, cellshard::domain_id{2},
        cellshard::order_id{3}, composition::prefix_sequence_kind_v1::trajectory,
        {}, full_entries.data(), full_entries.size(), 0};
    std::array<std::uint8_t, 5> prefix_marks{};
    std::array<std::uint8_t, 5> full_marks{};
    composition::prefix_composition_v1 output{};
    assert(composition::compose_prefix_v1(
               composition::composition_production_id{5}, prefix, full,
               prefix_marks.data(), full_marks.data(), full_marks.size(),
               &output).composed());
    assert(output.prefix_length == 3 && output.full_length == 5);

    auto wrong_entries = full_entries;
    wrong_entries[4].sequence_position = 3;
    wrong_entries[3].sequence_position = 2;
    auto wrong = full;
    wrong.entries = wrong_entries.data();
    assert(composition::compose_prefix_v1(
               composition::composition_production_id{5}, prefix, wrong,
               prefix_marks.data(), full_marks.data(), full_marks.size(),
               &output).code
           == composition::prefix_composition_code_v1::
                  prefix_position_mismatch);
}
