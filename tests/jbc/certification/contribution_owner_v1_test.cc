#include <CellShard/compiler/certification/contribution_owner_v1.hh>

#include <cassert>
#include <cstdint>

using namespace cellshard::compiler;

int main() {
    certification::certification_member_key_v1 members[]{
        {{1, 1}, 100, 4, certification::certification_member_kind_v1::entity, {}},
        {{1, 1}, 200, 7, certification::certification_member_kind_v1::entity, {}},
        {{2, 1},
         UINT64_C(1) << 44,
         9,
         certification::certification_member_kind_v1::relation_edge,
         {}}};
    certification::exact_contribution_owner_v1 owners[3]{};
    const auto result = certification::assign_exact_contribution_owners_v1(
        members, 3, owners, 3);
    assert(result.assigned());
    assert(owners[1].owner_atom_index == 7);
    assert(owners[2].global_identity == (UINT64_C(1) << 44));

    members[1].global_identity = members[0].global_identity;
    assert(certification::assign_exact_contribution_owners_v1(
               members, 3, owners, 3)
               .code
           == certification::contribution_owner_assignment_code_v1::
               duplicate_contribution_owner);

    members[1].global_identity = 50;
    assert(certification::assign_exact_contribution_owners_v1(
               members, 3, owners, 3)
               .code
           == certification::contribution_owner_assignment_code_v1::
               unordered_member);

    assert(certification::assign_exact_contribution_owners_v1(
               members, 3, owners, 2)
               .code
           == certification::contribution_owner_assignment_code_v1::
               insufficient_output);
}
