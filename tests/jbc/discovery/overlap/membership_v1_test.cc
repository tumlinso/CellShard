#include <CellShard/compiler/discovery/overlap/membership_v1.hh>

#include <cassert>
#include <cstdint>

using namespace cellshard::compiler;

int main() {
    const std::uint64_t members[]{UINT64_C(1) << 40,
                                  (UINT64_C(1) << 40) + 2,
                                  (UINT64_C(1) << 40) + 4};
    const std::uint64_t offsets[]{0, 1, 3, 4};
    const evidence::evidence_identity_v1 communities[]{{10, 1}, {10, 2}};
    discovery::overlap::overlap_membership_v1 memberships[]{
        {0, 0, 1, 1}, {0, 0, 1, 2}, {1, 0, 1, 2}, {1, 0, 1, 1}};
    discovery::overlap::bounded_overlap_membership_view_v1 view{
        members,
        offsets,
        memberships,
        communities,
        3,
        4,
        2,
        2,
        0,
        {20, 1},
        {21, 1},
        7};
    assert(discovery::overlap::validate_bounded_overlap_membership_v1(view)
               .valid());
    assert(!discovery::overlap::authorizes_execution(view));

    memberships[2].local_community_index = 0;
    assert(discovery::overlap::validate_bounded_overlap_membership_v1(view).code
           == discovery::overlap::bounded_overlap_validation_code_v1::
               unordered_or_duplicate_member_community);
    memberships[2].local_community_index = 1;

    view.maximum_memberships_per_member = 1;
    assert(discovery::overlap::validate_bounded_overlap_membership_v1(view).code
           == discovery::overlap::bounded_overlap_validation_code_v1::
               overlap_bound_exceeded);

    view.maximum_memberships_per_member = 2;
    memberships[1].weight_denominator = 0;
    assert(discovery::overlap::validate_bounded_overlap_membership_v1(view).code
           == discovery::overlap::bounded_overlap_validation_code_v1::
               invalid_weight);
}
