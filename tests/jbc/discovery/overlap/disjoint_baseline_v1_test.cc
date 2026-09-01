#include <CellShard/compiler/discovery/overlap/disjoint_baseline_v1.hh>

#include <cassert>
#include <cstdint>
#include <vector>

using namespace cellshard::compiler;

int main() {
    std::vector<std::uint64_t> members(1000);
    std::vector<std::uint32_t> assignments(1000);
    for (std::uint64_t index = 0; index < members.size(); ++index) {
        members[index] = (UINT64_C(1) << 40) + index + 1;
        assignments[index] = static_cast<std::uint32_t>(index % 3);
    }
    const evidence::evidence_identity_v1 communities[]{{1, 1},
                                                        {1, 2},
                                                        {1, 3}};
    const discovery::overlap::disjoint_assignment_view_v1 source{
        members.data(),
        assignments.data(),
        communities,
        members.size(),
        3,
        {2, 1},
        {3, 1},
        5};
    std::vector<std::uint64_t> offsets(members.size() + 1);
    std::vector<discovery::overlap::overlap_membership_v1> memberships(
        members.size());
    discovery::overlap::disjoint_baseline_buffers_v1 buffers{
        offsets.data(),
        offsets.size(),
        memberships.data(),
        memberships.size()};
    const auto result =
        discovery::overlap::build_disjoint_community_baseline_v1(source,
                                                                  buffers);
    assert(result.built());
    assert(result.view.maximum_memberships_per_member == 1);
    for (std::uint64_t index = 0; index < members.size(); ++index) {
        assert(result.view.member_offsets[index] == index);
        assert(result.view.memberships[index].local_community_index
               == assignments[index]);
    }

    assignments[800] = 3;
    assert(discovery::overlap::build_disjoint_community_baseline_v1(source,
                                                                     buffers)
               .code
           == discovery::overlap::disjoint_baseline_build_code_v1::
               assignment_out_of_range);
}
