#include <CellShard/compiler/discovery/factor_topic/exact_coverage_v1.hh>

#include <cassert>

namespace factor_topic = cellshard::compiler::discovery::factor_topic;
namespace evidence = cellshard::compiler::evidence;

int main() {
    const evidence::evidence_identity_v1 canonical_members[] = {
        {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}};
    const factor_topic::factor_candidate_member_v1 proposed[] = {
        {{1, 1}, 1, 1}, {{1, 2}, 1, 1}, {{1, 3}, 1, 1},
        {{1, 3}, 1, 1}, {{1, 4}, 1, 1}};
    const factor_topic::factor_candidate_v1 candidates[] = {
        {{8, 1}, 0, 3, factor_topic::external_factor_topic_kind_v1::factor, 0},
        {{8, 2}, 3, 2, factor_topic::external_factor_topic_kind_v1::topic, 0}};
    const evidence::evidence_identity_v1 exact_first[] = {{1, 1}, {1, 3}};
    const evidence::evidence_identity_v1 exact_second[] = {{1, 3}, {1, 4}};
    const factor_topic::factor_exact_membership_view_v1 exact[] = {
        {exact_first, 2, {8, 1}}, {exact_second, 2, {8, 2}}};
    factor_topic::factor_exact_coverage_span_v1 spans[2]{};
    factor_topic::factor_exact_owner_v1 owners[4]{};
    evidence::evidence_identity_v1 residual[5]{};

    auto result = factor_topic::construct_factor_exact_coverage_v1(
        {canonical_members, 5, {9, 1}, 7},
        candidates,
        2,
        proposed,
        5,
        exact,
        spans,
        2,
        owners,
        4,
        residual,
        5);
    assert(result.constructed());
    assert(result.owner_count == 3);
    assert(result.residual_count == 2);
    assert(spans[0].owner_count == 2);
    assert(spans[1].owner_count == 1);
    assert((owners[2].member_identity == evidence::evidence_identity_v1{1, 4}));
    assert((residual[0] == evidence::evidence_identity_v1{1, 2}));
    assert((residual[1] == evidence::evidence_identity_v1{1, 5}));
    assert(!factor_topic::authorizes_execution(result));

    result = factor_topic::construct_factor_exact_coverage_v1(
        {canonical_members, 5, {9, 1}, 7}, candidates, 2, proposed, 5, exact,
        spans, 2, owners, 2, residual, 5);
    assert(result.code
           == factor_topic::factor_exact_coverage_code_v1::
               insufficient_owner_capacity);
    assert(result.owner_count == 3);

    const evidence::evidence_identity_v1 unproposed[] = {{1, 5}};
    auto bad_exact = exact[1];
    bad_exact.members = unproposed;
    bad_exact.member_count = 1;
    const factor_topic::factor_exact_membership_view_v1 bad_views[] = {
        exact[0], bad_exact};
    result = factor_topic::construct_factor_exact_coverage_v1(
        {canonical_members, 5, {9, 1}, 7}, candidates, 2, proposed, 5, bad_views,
        spans, 2, owners, 4, residual, 5);
    assert(result.code
           == factor_topic::factor_exact_coverage_code_v1::
               exact_member_not_proposed);
    return 0;
}
