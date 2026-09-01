#include <CellShard/compiler/discovery/co_support/weighted_co_support_v1.hh>

#include <cassert>

namespace co_support = cellshard::compiler::discovery::co_support;

int main() {
    const std::uint64_t offsets[] = {0, 3, 6};
    const std::uint32_t sources[] = {0, 1, 2, 0, 1, 2};
    const std::uint64_t weights[] = {9, 4, 7, 3, 8, 5};
    const co_support::support_relation_view_v1 relation{
        offsets, sources, weights, nullptr, 6, 3, 2, 41, 2};
    const co_support::sampled_source_pair_v1 pairs[] = {
        {0, 1, 0, 0, 1, 1}, {0, 1, 1, 0, 1, 1},
        {1, 2, 0, 0, 1, 1},
    };
    co_support::weighted_co_support_record_v1 records[2]{};
    auto result = co_support::aggregate_weighted_co_support_v1(
        relation, pairs, 3, records, 2, 32);
    assert(result.aggregated());
    assert(result.record_count == 2);
    assert(records[0].source_a == 0 && records[0].source_b == 1);
    assert(records[0].sampled_destination_count == 2);
    assert(records[0].weighted_support == 7);
    assert(records[1].weighted_support == 4);

    const co_support::sampled_source_pair_v1 missing[] = {
        {0, 2, 2, 0, 1, 1},
    };
    result = co_support::aggregate_weighted_co_support_v1(
        relation, missing, 1, records, 2, 32);
    assert(result.code
           == co_support::weighted_co_support_code_v1::invalid_pair);

    result = co_support::aggregate_weighted_co_support_v1(
        relation, pairs, 3, records, 2, 1);
    assert(result.code
           == co_support::weighted_co_support_code_v1::work_limit_exceeded);
    return 0;
}
