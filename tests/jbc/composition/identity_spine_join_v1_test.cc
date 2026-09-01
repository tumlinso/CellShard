#include <CellShard/compiler/composition/identity_spine_join_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<std::uint64_t, 4> identities{{
        2, 9, 11, (std::uint64_t{1} << 48u)}};
    const composition::identity_spine_view_v1 left{
        cellshard::structure_id{1}, cellshard::domain_id{2},
        cellshard::order_id{3}, identities.data(), identities.size()};
    const composition::identity_spine_view_v1 right{
        cellshard::structure_id{4}, cellshard::domain_id{2},
        cellshard::order_id{5}, identities.data(), identities.size()};
    std::array<composition::identity_spine_join_entry_v1, 4> entries{};
    composition::identity_spine_join_view_v1 output{};
    assert(composition::compose_identity_spine_join_v1(
               cellshard::structure_id{6}, left, right, entries.data(),
               entries.size(), &output).joined());
    assert(entries[3].logical_identity == (std::uint64_t{1} << 48u));
    assert(entries[3].left_local_index == 3);
    assert(output.left_order != output.right_order);

    auto mismatched_ids = identities;
    mismatched_ids[2] = 10;
    auto mismatch = right;
    mismatch.logical_identities = mismatched_ids.data();
    assert(composition::compose_identity_spine_join_v1(
               cellshard::structure_id{6}, left, mismatch, entries.data(),
               entries.size(), &output).code
           == composition::identity_spine_join_code_v1::
                  missing_left_identity);
}
