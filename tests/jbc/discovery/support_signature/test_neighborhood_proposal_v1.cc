#include <CellShard/compiler/discovery/support_signature/neighborhood_proposal_v1.hh>

#include <array>
#include <cassert>

namespace signature =
    cellshard::compiler::discovery::support_signature;

int main() {
    const std::uint64_t destinations[] = {10, 20, 30};
    const std::uint64_t offsets[] = {0, 3, 7, 9};
    const std::uint64_t sources[] = {1, 2, 3, 2, 3, 4, 5, 8, 9};
    const signature::exact_destination_support_view_v1 support{
        destinations, offsets, sources, 3, 9, {1, 1}, {2, 1}, {3, 1}, 4};
    const signature::exact_support_pair_score_v1 scores[] = {
        {0, 1, 2, 5, 3, 4}, {0, 2, 0, 5, 3, 2}};
    const signature::exact_support_pair_score_view_v1 score_view{
        scores, 2, {1, 1}, 4};
    const cellshard::compiler::atom::atom_persistent_identity_v1 ids[] = {
        {4, 1}};
    std::array<signature::destination_support_neighborhood_proposal_v1, 1>
        output{};
    auto result = signature::build_support_neighborhood_proposals_v1(
        support, score_view, {2, 2, 5, 1, 2}, ids, 1,
        output.data(), output.size());
    assert(result.built() && result.view.proposal_count == 1);
    assert(output[0].first_global_destination_id == 10);
    assert(output[0].second_global_destination_id == 20);
    assert(!signature::authorizes_execution(result.view));
    assert(signature::ratio_at_least_v1(UINT64_MAX, UINT64_MAX,
                                        UINT64_MAX - 1, UINT64_MAX));
    result = signature::build_support_neighborhood_proposals_v1(
        support, score_view, {2, 2, 5, 1, 2}, ids, 0,
        output.data(), output.size());
    assert(result.code == signature::support_neighborhood_code_v1::
                              insufficient_proposal_identities);
}
