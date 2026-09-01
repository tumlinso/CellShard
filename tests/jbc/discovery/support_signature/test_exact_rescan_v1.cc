#include <CellShard/compiler/discovery/support_signature/exact_rescan_v1.hh>

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
    const signature::support_candidate_pair_v1 pairs[] = {
        {0, 1, 2, 0}, {0, 2, 1, 0}};
    signature::support_candidate_pair_view_v1 candidates{
        pairs, 2, 3, 2, 0, {1, 1}, 4};
    std::array<signature::exact_support_pair_score_v1, 2> scores{};
    auto result = signature::rescan_exact_support_pairs_v1(
        support, candidates, scores.data(), scores.size());
    assert(result.rescanned());
    assert(scores[0].shared_support_count == 2);
    assert(scores[0].union_support_count == 5);
    assert(scores[0].first_support_count == 3);
    assert(scores[0].second_support_count == 4);
    assert(scores[1].shared_support_count == 0);
    assert(!signature::authorizes_execution(result.view));
    candidates.relation_generation = 5;
    assert(signature::rescan_exact_support_pairs_v1(
               support, candidates, scores.data(), scores.size()).code
           == signature::exact_support_rescan_code_v1::context_mismatch);
}
