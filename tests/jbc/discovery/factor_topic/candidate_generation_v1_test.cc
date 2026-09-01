#include <CellShard/compiler/discovery/factor_topic/candidate_generation_v1.hh>

#include <cassert>

namespace factor_topic = cellshard::compiler::discovery::factor_topic;
namespace evidence = cellshard::compiler::evidence;

factor_topic::soft_membership_store_v1 make_store(
    evidence::evidence_identity_v1 identity,
    const evidence::approximate_member_v1 *members,
    std::uint64_t count) {
    factor_topic::soft_membership_store_v1 store{};
    store.evidence_record.evidence_identity = identity;
    store.evidence_record.subject_atom_identity = {99, identity.local_identity};
    store.evidence_record.source_identity = {77, 1};
    store.evidence_record.observation_generation = 1;
    store.evidence_record.observation_count = count;
    store.evidence_record.kind = evidence::evidence_kind::factor_membership;
    store.membership = {members, count, count, identity};
    return store;
}

int main() {
    const evidence::approximate_member_v1 first[] = {
        {{1, 1}, 3, 4}, {{1, 2}, 1, 2}, {{1, 3}, 1, 4}};
    const evidence::approximate_member_v1 second[] = {
        {{1, 1}, 7, 8}, {{1, 3}, 3, 4}, {{1, 4}, 1, 2}};
    const factor_topic::soft_membership_store_v1 stores[] = {
        make_store({8, 1}, first, 3), make_store({8, 2}, second, 3)};
    factor_topic::factor_candidate_v1 candidates[2]{};
    factor_topic::factor_candidate_member_v1 members[6]{};
    const factor_topic::factor_candidate_generation_config_v1 config{
        1, 2, 2, 1, 64};
    auto result = factor_topic::generate_factor_candidates_v1(
        stores, 2, config, candidates, 2, members, 6);
    assert(result.generated());
    assert(result.candidate_count == 2);
    assert(result.member_count == 4);
    assert(candidates[0].member_count == 2);
    assert(candidates[1].member_count == 2);
    assert((members[0].member_identity == evidence::evidence_identity_v1{1, 1}));
    assert((members[2].member_identity == evidence::evidence_identity_v1{1, 3}));
    assert((members[3].member_identity == evidence::evidence_identity_v1{1, 4}));

    auto limited = config;
    limited.maximum_pair_checks = 1;
    result = factor_topic::generate_factor_candidates_v1(
        stores, 2, limited, candidates, 2, members, 6);
    assert(result.code
           == factor_topic::factor_candidate_generation_code_v1::work_limit_exceeded);
    assert(result.candidate_count == 1);
    assert(result.member_count == 2);

    auto invalid = config;
    invalid.threshold_denominator = 0;
    result = factor_topic::generate_factor_candidates_v1(
        stores, 2, invalid, candidates, 2, members, 6);
    assert(result.code
           == factor_topic::factor_candidate_generation_code_v1::invalid_config);

    assert(factor_topic::fraction_at_least_v1(UINT64_MAX, UINT64_MAX, 1, 1));
    assert(!factor_topic::fraction_at_least_v1(1, UINT64_MAX, 1, 2));
    return 0;
}
