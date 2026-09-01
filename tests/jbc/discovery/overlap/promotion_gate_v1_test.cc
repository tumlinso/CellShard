#include <CellShard/compiler/discovery/overlap/promotion_gate_v1.hh>

#include <cassert>
#include <cstdint>

using namespace cellshard::compiler;

int main() {
    const std::uint64_t members[]{10, 20};
    const evidence::evidence_identity_v1 communities[]{{1, 1}, {1, 2}};
    const std::uint64_t baseline_offsets[]{0, 1, 2};
    const discovery::overlap::overlap_membership_v1 baseline_memberships[]{
        {0, 0, 1, 1}, {1, 0, 1, 1}};
    const discovery::overlap::bounded_overlap_membership_view_v1 baseline{
        members, baseline_offsets, baseline_memberships, communities,
        2, 2, 2, 1, 0, {2, 1}, {3, 1}, 1};
    assert(discovery::overlap::prove_zero_overlap_equivalence_v1(
               baseline, baseline)
               .equivalent());

    const std::uint64_t overlap_offsets[]{0, 2, 4};
    const discovery::overlap::overlap_membership_v1 overlap_memberships[]{
        {0, 0, 1, 1}, {1, 0, 1, 2}, {0, 0, 1, 2}, {1, 0, 1, 1}};
    const discovery::overlap::bounded_overlap_membership_view_v1 candidate{
        members, overlap_offsets, overlap_memberships, communities,
        2, 4, 2, 2, 0, {2, 2}, {3, 1}, 2};
    const std::uint64_t exact_0[]{10, 20};
    const std::uint64_t exact_1[]{10, 20};
    const discovery::overlap::exact_overlap_candidate_v1 exact_candidates[]{
        {exact_0, 2, {10, 1}, {1, 1}},
        {exact_1, 2, {10, 2}, {1, 2}}};
    const discovery::overlap::exact_overlap_candidate_table_v1 table{
        exact_candidates, 2, {20, 1}, {21, 1}, {22, 1}, 3};
    const discovery::overlap::overlap_stability_cost_v1 score{
        7, 8, 2, 48, 48, 96};
    const discovery::overlap::overlap_promotion_policy_v1 policy{
        3, 4, 1, 96};
    const auto promoted = discovery::overlap::gate_overlap_promotion_v1(
        baseline, candidate, score, table, policy);
    assert(promoted.promoted());
    assert(promoted.decision
           == discovery::overlap::overlap_promotion_decision_v1::
               promote_to_independent_certification);
    assert(!discovery::overlap::authorizes_execution(promoted));

    auto costly = score;
    costly.complete_duplication_cost_bytes = 97;
    assert(discovery::overlap::gate_overlap_promotion_v1(
               baseline, candidate, costly, table, policy)
               .decision
           == discovery::overlap::overlap_promotion_decision_v1::
               reject_duplication_cost);

    auto unstable = score;
    unstable.stability_intersection_count = 1;
    assert(discovery::overlap::gate_overlap_promotion_v1(
               baseline, candidate, unstable, table, policy)
               .decision
           == discovery::overlap::overlap_promotion_decision_v1::
               reject_unstable);

    auto changed = baseline;
    discovery::overlap::overlap_membership_v1 changed_memberships[]{
        baseline_memberships[0], baseline_memberships[1]};
    changed_memberships[1].local_community_index = 0;
    changed.memberships = changed_memberships;
    assert(!discovery::overlap::prove_zero_overlap_equivalence_v1(
                baseline, changed)
                .equivalent());
}
