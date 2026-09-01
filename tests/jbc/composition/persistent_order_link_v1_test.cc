#include <CellShard/compiler/composition/persistent_order_link_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<composition::logical_to_local_order_entry_v1, 4> left{{
        {2, 1, 0}, {7, 3, 0}, {9, 0, 0}, {(std::uint64_t{1} << 53u), 2, 0}}};
    const std::array<composition::logical_to_local_order_entry_v1, 4> right{{
        {2, 2, 0}, {7, 0, 0}, {9, 3, 0}, {(std::uint64_t{1} << 53u), 1, 0}}};
    const composition::persistent_order_index_v1 left_index{
        cellshard::domain_id{1}, cellshard::order_id{2}, left.data(),
        left.size(), 0};
    const composition::persistent_order_index_v1 right_index{
        cellshard::domain_id{1}, cellshard::order_id{3}, right.data(),
        right.size(), 0};
    std::array<std::uint8_t, 4> left_marks{};
    std::array<std::uint8_t, 4> right_marks{};
    std::array<std::uint32_t, 4> mapping{};
    composition::persistent_order_link_v1 output{};
    assert(composition::compose_persistent_order_link_v1(
               left_index, right_index,
               {left_marks.data(), right_marks.data(), left_marks.size(),
                mapping.data(), mapping.size()}, &output).linked());
    assert((mapping == std::array<std::uint32_t, 4>{3, 2, 1, 0}));

    auto duplicate = right;
    duplicate[3].local_index = 0;
    auto malformed = right_index;
    malformed.entries = duplicate.data();
    assert(composition::compose_persistent_order_link_v1(
               left_index, malformed,
               {left_marks.data(), right_marks.data(), left_marks.size(),
                mapping.data(), mapping.size()}, &output).code
           == composition::persistent_order_link_code_v1::
                  duplicate_local_index);
}
