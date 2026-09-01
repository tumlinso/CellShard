#include <CellShard/compiler/atom/overlap_role_v1.hh>

#include <array>
#include <cassert>

namespace {

using namespace cellshard::compiler::atom;

atom_overlap_role_record_v1 make_record(
    std::uint64_t identity,
    atom_overlap_role_v1 role,
    bool overlaps) {
    atom_overlap_role_record_v1 record{};
    record.member_identity = {1, identity};
    record.membership_identity = {2, identity};
    record.role = role;
    record.overlaps_other_members = overlaps ? 1 : 0;
    if (overlaps) {
        record.overlap_group_identity = {3, 1};
    }
    if (role == atom_overlap_role_v1::partial_contribution) {
        record.partial_algebra_identity = {4, 1};
    }
    return record;
}

void test_categories_remain_distinct() {
    std::array<atom_overlap_role_record_v1, 5> records{
        make_record(1, atom_overlap_role_v1::proposal_membership, true),
        make_record(2, atom_overlap_role_v1::physical_replica, true),
        make_record(3, atom_overlap_role_v1::read_halo, true),
        make_record(4, atom_overlap_role_v1::partial_contribution, true),
        make_record(5, atom_overlap_role_v1::exclusive_contribution_owner,
                    false)};
    const atom_overlap_role_table_v1 table{
        records.data(), records.size(), {5, 1}};
    const auto result = validate_atom_overlap_role_table_v1(table);
    assert(result.valid());
    assert(result.index == records.size());
    assert(atom_overlap_permitted_v1(atom_overlap_role_v1::read_halo));
    assert(!atom_overlap_permitted_v1(
        atom_overlap_role_v1::exclusive_contribution_owner));
}

void test_deterministic_rejections() {
    auto record = make_record(
        1, atom_overlap_role_v1::exclusive_contribution_owner, true);
    atom_overlap_role_table_v1 table{&record, 1, {5, 1}};
    assert(validate_atom_overlap_role_table_v1(table).code
           == atom_overlap_role_validation_code_v1::exclusive_owner_overlap);

    record = make_record(1, atom_overlap_role_v1::partial_contribution, true);
    record.partial_algebra_identity = {};
    assert(validate_atom_overlap_role_table_v1(table).code
           == atom_overlap_role_validation_code_v1::missing_partial_algebra);

    record = make_record(1, atom_overlap_role_v1::physical_replica, false);
    record.overlap_group_identity = {3, 1};
    assert(validate_atom_overlap_role_table_v1(table).code
           == atom_overlap_role_validation_code_v1::
                  unexpected_overlap_group);

    record = make_record(1, atom_overlap_role_v1::read_halo, true);
    record.partial_algebra_identity = {4, 1};
    assert(validate_atom_overlap_role_table_v1(table).code
           == atom_overlap_role_validation_code_v1::
                  unexpected_partial_algebra);

    std::array<atom_overlap_role_record_v1, 2> duplicate{
        make_record(1, atom_overlap_role_v1::proposal_membership, false),
        make_record(1, atom_overlap_role_v1::proposal_membership, true)};
    table = {duplicate.data(), duplicate.size(), {5, 1}};
    assert(validate_atom_overlap_role_table_v1(table).code
           == atom_overlap_role_validation_code_v1::
                  unordered_or_duplicate_role);
}

} // namespace

int main() {
    test_categories_remain_distinct();
    test_deterministic_rejections();
    return 0;
}
