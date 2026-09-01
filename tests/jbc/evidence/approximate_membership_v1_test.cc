#include <CellShard/compiler/evidence/approximate_membership_v1.hh>

#include <array>
#include <cassert>

namespace evidence = cellshard::compiler::evidence;

int main() {
    std::array<evidence::approximate_member_v1, 2> members{{
        {{1, 1}, 1, 2},
        {{1, 2}, 3, 4},
    }};
    evidence::approximate_membership_view_v1 view{
        members.data(), members.size(), members.size(), {2, 1}};
    assert(evidence::validate_approximate_membership_v1(view).valid());
    assert(!evidence::is_exact_membership(view));

    auto malformed = view;
    malformed.member_capacity = 1;
    assert(evidence::validate_approximate_membership_v1(malformed).code
           == evidence::approximate_membership_validation_code_v1::capacity_overflow);
    malformed = view;
    members[1].member_identity = members[0].member_identity;
    assert(evidence::validate_approximate_membership_v1(malformed).code
           == evidence::approximate_membership_validation_code_v1::unordered_or_duplicate_member);
    members[1].member_identity = {1, 2};
    members[1].weight_numerator = 5;
    assert(evidence::validate_approximate_membership_v1(view).code
           == evidence::approximate_membership_validation_code_v1::invalid_weight);
}
