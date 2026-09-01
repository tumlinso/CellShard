#include <CellShard/compiler/composition/multimodal_join_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<std::uint64_t, 3> identities{{
        3, 7, (std::uint64_t{1} << 49u)}};
    const std::array<composition::identity_spine_view_v1, 3> modalities{{
        {cellshard::structure_id{1}, cellshard::domain_id{10},
         cellshard::order_id{11}, identities.data(), identities.size()},
        {cellshard::structure_id{2}, cellshard::domain_id{20},
         cellshard::order_id{21}, identities.data(), identities.size()},
        {cellshard::structure_id{3}, cellshard::domain_id{30},
         cellshard::order_id{31}, identities.data(), identities.size()}}};
    std::array<composition::multimodal_join_entry_v1, 3> entries{};
    composition::multimodal_join_view_v1 output{};
    assert(composition::compose_multimodal_join_v1(
               cellshard::structure_id{40}, cellshard::domain_id{41},
               cellshard::order_id{42}, modalities.data(), modalities.size(),
               entries.data(), entries.size(), &output).joined());
    assert(output.modality_count == 3 && output.entry_count == 3);
    assert(entries[2].logical_identity == (std::uint64_t{1} << 49u));
    assert(entries[2].local_indices[2] == 2);

    auto mismatched_identities = identities;
    mismatched_identities[1] = 6;
    auto mismatch = modalities;
    mismatch[2].logical_identities = mismatched_identities.data();
    assert(composition::compose_multimodal_join_v1(
               cellshard::structure_id{40}, cellshard::domain_id{41},
               cellshard::order_id{42}, mismatch.data(), mismatch.size(),
               entries.data(), entries.size(), &output).code
           == composition::multimodal_join_code_v1::identity_mismatch);
}
