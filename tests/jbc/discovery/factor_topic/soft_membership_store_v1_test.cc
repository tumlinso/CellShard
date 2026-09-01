#include <CellShard/compiler/discovery/factor_topic/soft_membership_store_v1.hh>

#include <cassert>

namespace factor_topic = cellshard::compiler::discovery::factor_topic;
namespace evidence = cellshard::compiler::evidence;

int main() {
    factor_topic::external_factor_topic_evidence_v1 external{};
    external.evidence_identity = {10, 20};
    external.subject_atom_identity = {30, 40};
    external.source_identity = {50, 60};
    external.observation_generation = 2;
    external.observation_count = 3;

    const evidence::approximate_member_v1 input_members[] = {
        {{7, 1}, 1, 4},
        {{7, 2}, 3, 4},
        {{7, 3}, 1, 1},
    };
    const evidence::approximate_membership_view_v1 input{
        input_members, 3, 3, external.evidence_identity};
    evidence::approximate_member_v1 storage[3]{};
    factor_topic::soft_membership_store_v1 stored{};
    auto result = factor_topic::store_soft_membership_evidence_v1(
        external, input, storage, 3, &stored);
    assert(result.stored());
    assert(stored.membership.members == storage);
    assert(stored.membership.member_count == 3);
    assert(storage[1].weight_numerator == 3);
    assert(evidence::validate_approximate_membership_v1(stored.membership).valid());
    assert(!evidence::is_exact_membership(stored.membership));

    result = factor_topic::store_soft_membership_evidence_v1(
        external, input, storage, 2, &stored);
    assert(result.code
           == factor_topic::soft_membership_store_code_v1::insufficient_capacity);
    assert(result.required_capacity == 3);

    auto mismatched = input;
    mismatched.evidence_identity = {10, 21};
    result = factor_topic::store_soft_membership_evidence_v1(
        external, mismatched, storage, 3, &stored);
    assert(result.code
           == factor_topic::soft_membership_store_code_v1::
               evidence_identity_mismatch);

    const evidence::approximate_member_v1 invalid_members[] = {
        {{7, 1}, 2, 1},
    };
    const evidence::approximate_membership_view_v1 invalid_view{
        invalid_members, 1, 1, external.evidence_identity};
    result = factor_topic::store_soft_membership_evidence_v1(
        external,
        invalid_view,
        storage,
        3,
        &stored);
    assert(result.code
           == factor_topic::soft_membership_store_code_v1::invalid_membership);
    return 0;
}
