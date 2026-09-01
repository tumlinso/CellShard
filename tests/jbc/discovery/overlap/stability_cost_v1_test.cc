#include <CellShard/compiler/discovery/overlap/stability_cost_v1.hh>

#include <cassert>
#include <cstdint>

using namespace cellshard::compiler;

int main() {
    const std::uint64_t members[]{10, 20};
    const std::uint64_t offsets[]{0, 2, 4};
    const evidence::evidence_identity_v1 communities[]{{1, 1}, {1, 2}};
    const discovery::overlap::overlap_membership_v1 memberships[]{
        {0, 0, 1, 1}, {1, 0, 1, 2}, {0, 0, 1, 2}, {1, 0, 1, 1}};
    const discovery::overlap::bounded_overlap_membership_view_v1 candidate{
        members, offsets, memberships, communities, 2, 4, 2, 2, 0,
        {2, 1}, {3, 1}, 1};
    const discovery::overlap::bounded_overlap_membership_view_v1 resamples[]{
        candidate, candidate};
    const auto result =
        discovery::overlap::score_overlap_stability_and_duplication_v1(
            candidate, resamples, 2, {24, 3, 2});
    assert(result.scored());
    assert(result.score.stability_intersection_count == 8);
    assert(result.score.stability_union_count == 8);
    assert(result.score.additional_membership_count == 2);
    assert(result.score.persistent_duplication_bytes == 144);
    assert(result.score.expected_movement_bytes == 288);
    assert(result.score.complete_duplication_cost_bytes == 432);

    auto mismatched = candidate;
    const std::uint64_t changed_members[]{10, 21};
    mismatched.global_member_ids = changed_members;
    assert(discovery::overlap::score_overlap_stability_and_duplication_v1(
               candidate, &mismatched, 1, {24, 1, 0})
               .code
           == discovery::overlap::overlap_stability_cost_code_v1::
               member_spine_mismatch);

    assert(discovery::overlap::score_overlap_stability_and_duplication_v1(
               candidate, resamples, 2, {UINT64_MAX, 2, 1})
               .code
           == discovery::overlap::overlap_stability_cost_code_v1::
               byte_overflow);
}
