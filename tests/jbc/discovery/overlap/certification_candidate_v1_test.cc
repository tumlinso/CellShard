#include <CellShard/compiler/discovery/overlap/certification_candidate_v1.hh>

#include <cassert>
#include <cstdint>

using namespace cellshard::compiler;

int main() {
    const std::uint64_t members[]{100, 200, 300};
    const std::uint64_t offsets[]{0, 1, 3, 4};
    const evidence::evidence_identity_v1 communities[]{{1, 1}, {1, 2}};
    const discovery::overlap::overlap_membership_v1 memberships[]{
        {0, 0, 1, 1}, {0, 0, 1, 2}, {1, 0, 1, 2}, {1, 0, 1, 1}};
    const discovery::overlap::bounded_overlap_membership_view_v1 proposal{
        members, offsets, memberships, communities, 3, 4, 2, 2, 0,
        {2, 1}, {3, 1}, 1};
    discovery::overlap::exact_overlap_rescan_member_v1 exact_members[]{
        {100, 0, 0}, {200, 0, 0}, {300, 1, 0}};
    const discovery::overlap::exact_overlap_rescan_view_v1 rescan{
        exact_members, 3, {10, 1}, 4};
    const atom::atom_persistent_identity_v1 candidate_ids[]{
        {20, 1}, {20, 2}};
    discovery::overlap::exact_overlap_candidate_v1 candidates[2]{};
    std::uint64_t exact_ids[3]{};
    const discovery::overlap::exact_overlap_candidate_buffers_v1 buffers{
        candidates, 2, exact_ids, 3};
    const auto result = discovery::overlap::
        build_exact_overlap_certification_candidates_v1(
            proposal, rescan, candidate_ids, {30, 1}, {31, 1}, buffers);
    assert(result.built());
    assert(result.table.candidate_count == 2);
    assert(result.table.candidates[0].member_count == 2);
    assert(result.table.candidates[1].member_count == 1);
    assert(!discovery::overlap::authorizes_execution(result.table));

    assert(discovery::overlap::build_exact_overlap_certification_candidates_v1(
               proposal, rescan, candidate_ids, {30, 1}, {30, 1}, buffers)
               .code
           == discovery::overlap::overlap_candidate_build_code_v1::
               provider_self_certification);

    exact_members[2] = {100, 1, 0};
    assert(discovery::overlap::build_exact_overlap_certification_candidates_v1(
               proposal, rescan, candidate_ids, {30, 1}, {31, 1}, buffers)
               .code
           == discovery::overlap::overlap_candidate_build_code_v1::
               community_not_proposed_for_member);
}
