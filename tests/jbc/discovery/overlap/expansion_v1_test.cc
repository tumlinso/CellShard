#include <CellShard/compiler/discovery/overlap/expansion_v1.hh>

#include <cassert>
#include <cstdint>
#include <vector>

using namespace cellshard::compiler;

int main() {
    const std::uint64_t members[]{100, 200};
    const std::uint64_t baseline_offsets[]{0, 1, 2};
    const evidence::evidence_identity_v1 communities[]{{1, 1},
                                                        {1, 2},
                                                        {1, 3},
                                                        {1, 4}};
    const discovery::overlap::overlap_membership_v1 baseline_memberships[]{
        {0, 0, 1, 1}, {1, 0, 1, 1}};
    const discovery::overlap::bounded_overlap_membership_view_v1 baseline{
        members,
        baseline_offsets,
        baseline_memberships,
        communities,
        2,
        2,
        4,
        1,
        0,
        {2, 1},
        {3, 1},
        1};
    discovery::overlap::overlap_expansion_candidate_v1 candidates[]{
        {100, 1, 0, 1, 3},
        {100, 2, 0, 3, 4},
        {100, 3, 0, UINT64_MAX - 1, UINT64_MAX},
        {200, 0, 0, 2, 3},
        {200, 2, 0, 1, 4}};
    std::vector<std::uint64_t> offsets(3);
    std::vector<discovery::overlap::overlap_membership_v1> output(6);
    std::vector<discovery::overlap::overlap_membership_v1> scratch(3);
    const discovery::overlap::overlap_expansion_buffers_v1 buffers{
        offsets.data(), offsets.size(), output.data(), output.size(),
        scratch.data(), scratch.size()};
    const auto result = discovery::overlap::build_bounded_overlap_expansion_v1(
        baseline, candidates, 5, 3, {4, 1}, 2, buffers);
    assert(result.built());
    assert(result.view.membership_count == 6);
    assert(result.view.memberships[0].local_community_index == 0);
    assert(result.view.memberships[1].local_community_index == 2);
    assert(result.view.memberships[2].local_community_index == 3);
    assert(result.view.memberships[3].local_community_index == 0);
    assert(result.view.memberships[4].local_community_index == 1);
    assert(result.view.memberships[5].local_community_index == 2);

    candidates[2].local_community_index = 2;
    assert(discovery::overlap::build_bounded_overlap_expansion_v1(
               baseline, candidates, 5, 3, {4, 1}, 2, buffers)
               .code
           == discovery::overlap::overlap_expansion_code_v1::
               unordered_or_duplicate_candidate);
}
