#include <CellShard/compiler/discovery/support_signature/full_relation_rescan_v1.hh>

#include <array>
#include <cassert>

namespace signature =
    cellshard::compiler::discovery::support_signature;

int main() {
    const std::uint64_t destinations[] = {10, 20};
    const std::uint64_t offsets[] = {0, 3, 7};
    const std::uint64_t sources[] = {1, 2, 3, 2, 3, 4, 5};
    const signature::exact_destination_support_view_v1 support{
        destinations, offsets, sources, 2, 7, {1, 1}, {2, 1}, {3, 1}, 4};
    const signature::destination_support_neighborhood_proposal_v1 proposal{
        {4, 1}, 10, 20, {0, 1, 2, 5, 3, 4}};
    const signature::destination_support_neighborhood_view_v1 proposals{
        &proposal, 1, {1, 1}, 4};
    std::array<signature::exact_support_neighborhood_rescan_v1, 1> rescans{};
    std::array<std::uint64_t, 5> output{};
    auto result = signature::rescan_full_relation_support_neighborhoods_v1(
        support, proposals, {5, 1}, 6, {7, 1}, {8, 1},
        {rescans.data(), rescans.size(), output.data(), output.size()});
    assert(result.rescanned());
    assert(rescans[0].shared_source_count == 2);
    assert(rescans[0].first_residual_count == 1);
    assert(rescans[0].second_residual_count == 2);
    assert((rescans[0].shared_source_ids[0] == 2
            && rescans[0].shared_source_ids[1] == 3));
    assert(rescans[0].first_residual_source_ids[0] == 1);
    assert((rescans[0].second_residual_source_ids[0] == 4
            && rescans[0].second_residual_source_ids[1] == 5));
    assert(!signature::authorizes_execution(result.table));
    assert(signature::rescan_full_relation_support_neighborhoods_v1(
               support, proposals, {5, 1}, 6, {7, 1}, {7, 1},
               {rescans.data(), rescans.size(), output.data(), output.size()})
               .code
           == signature::full_relation_rescan_code_v1::
                  provider_self_certification);
}
