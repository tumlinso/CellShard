#include <CellShard/compiler/discovery/bicluster/overlapping_proposals_v1.hh>

#include <cassert>

namespace bicluster = cellshard::compiler::discovery::bicluster;
namespace evidence = cellshard::compiler::evidence;

int main() {
    const evidence::evidence_identity_v1 first_sources[] = {{1, 1}, {1, 2}};
    const evidence::evidence_identity_v1 second_sources[] = {{1, 2}, {1, 3}};
    const evidence::evidence_identity_v1 destinations[] = {{2, 1}, {2, 2}};
    const bicluster::expanded_bicluster_v1 rectangles[] = {
        {{9, 1}, {3, 1}, first_sources, 2, 2, destinations, 2, 2, 0, 1},
        {{9, 1}, {3, 1}, second_sources, 2, 2, destinations, 2, 2, 1, 1}};
    const bicluster::bicluster_exact_census_v1 censuses[] = {
        {4, 4, 0, 0, 1}, {4, 4, 0, 0, 1}};
    bicluster::bicluster_marginal_utility_v1 promoted{};
    promoted.exact_interaction_count = 4;
    promoted.gross_savings = 100;
    promoted.complete_overhead = 10;
    promoted.net_utility = 90;
    promoted.disposition = bicluster::bicluster_promotion_v1::promote_proposal;
    const bicluster::bicluster_marginal_utility_v1 utilities[] = {
        promoted, promoted};
    bicluster::bicluster_proposal_v1 proposals[2]{};

    auto result = bicluster::emit_bounded_bicluster_proposals_v1(
        rectangles, censuses, utilities, 2, {2, 1, 64}, proposals, 2);
    assert(result.emitted());
    assert(result.proposal_count == 1);
    assert(result.rejected_overlap_count == 1);

    result = bicluster::emit_bounded_bicluster_proposals_v1(
        rectangles, censuses, utilities, 2, {2, 2, 64}, proposals, 2);
    assert(result.emitted());
    assert(result.proposal_count == 2);
    assert(!bicluster::authorizes_execution(proposals[0]));
    return 0;
}
