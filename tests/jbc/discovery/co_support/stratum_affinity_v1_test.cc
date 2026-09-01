#include <CellShard/compiler/discovery/co_support/stratum_affinity_v1.hh>

#include <cassert>

namespace co_support = cellshard::compiler::discovery::co_support;

int main() {
    const std::uint64_t offsets[] = {0, 2, 4, 6};
    const std::uint32_t sources[] = {0, 1, 0, 1, 0, 1};
    const std::uint64_t weights[] = {7, 4, 2, 8, 5, 6};
    const std::uint32_t strata[] = {1, 0, 1};
    const co_support::support_relation_view_v1 relation{
        offsets, sources, weights, strata, 6, 2, 3, 91, 4};
    const co_support::sampled_source_pair_v1 pairs[] = {
        {0, 1, 0, 0, 1, 1}, {0, 1, 1, 0, 1, 1},
        {0, 1, 2, 0, 1, 1},
    };
    co_support::stratum_affinity_record_v1 records[2]{};
    auto result = co_support::accumulate_stratum_affinity_v1(
        relation, 2, pairs, 3, records, 2, 32);
    assert(result.accumulated());
    assert(result.record_count == 2);
    assert(records[0].stratum_id == 0);
    assert(records[0].weighted_support == 2);
    assert(records[1].stratum_id == 1);
    assert(records[1].sampled_destination_count == 2);
    assert(records[1].weighted_support == 9);

    result = co_support::accumulate_stratum_affinity_v1(
        relation, 1, pairs, 3, records, 2, 32);
    assert(result.code
           == co_support::stratum_affinity_code_v1::stratum_out_of_range);
    return 0;
}
