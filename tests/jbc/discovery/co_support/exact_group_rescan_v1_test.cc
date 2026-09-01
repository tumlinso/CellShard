#include <CellShard/compiler/discovery/co_support/exact_group_rescan_v1.hh>

#include <cassert>

namespace co_support = cellshard::compiler::discovery::co_support;

int main() {
    const std::uint64_t offsets[] = {0, 3, 5, 8};
    const std::uint32_t sources[] = {0, 1, 2, 0, 2, 0, 1, 3};
    const co_support::support_relation_view_v1 relation{
        offsets, sources, nullptr, nullptr, 8, 4, 3, 77, 5};
    const std::uint64_t group_offsets[] = {0, 2, 4};
    const std::uint32_t group_sources[] = {0, 1, 0, 2};
    const std::uint64_t identities[] = {901, 902};
    const co_support::source_group_proposals_view_v1 proposals{
        group_offsets, group_sources, identities, 4, 2, 0, 77, 5};
    co_support::destination_group_member_v1 members[4]{};
    co_support::exact_group_rescan_summary_v1 summaries[2]{};
    auto result = co_support::exact_rescan_group_proposals_v1(
        relation, proposals, members, 4, summaries, 2, 64);
    assert(result.rescanned());
    assert(result.summary_count == 2);
    assert(result.member_count == 4);
    assert(summaries[0].destination_count == 2);
    assert(summaries[0].assigned_edge_count == 4);
    assert(summaries[0].residual_edge_count == 4);
    assert(summaries[1].destination_count == 2);
    assert(members[0].proposal_identity == 901);
    assert(members[0].destination_id == 0);

    result = co_support::exact_rescan_group_proposals_v1(
        relation, proposals, members, 3, summaries, 2, 64);
    assert(result.code == co_support::exact_group_rescan_code_v1::
        insufficient_member_capacity);
    return 0;
}
