#include <CellShard/compiler/discovery/co_support/raw_co_support_v1.hh>

#include <cassert>

namespace co_support = cellshard::compiler::discovery::co_support;

int main() {
    const co_support::sampled_source_pair_v1 pairs[] = {
        {2, 3, 0, 0, 1, 1},
        {0, 1, 1, 0, 1, 1},
        {2, 3, 2, 0, 1, 1},
        {0, 2, 2, 0, 1, 1},
    };
    co_support::raw_co_support_record_v1 records[3]{};
    auto result = co_support::aggregate_raw_co_support_v1(
        pairs, 4, records, 3, 32);
    assert(result.aggregated());
    assert(result.record_count == 3);
    assert(result.consumed_pair_count == 4);
    assert(records[0].source_a == 0 && records[0].source_b == 1);
    assert(records[1].source_a == 0 && records[1].source_b == 2);
    assert(records[2].source_a == 2 && records[2].source_b == 3);
    assert(records[2].sampled_destination_count == 2);

    result = co_support::aggregate_raw_co_support_v1(
        pairs, 4, records, 2, 32);
    assert(result.code
           == co_support::raw_co_support_code_v1::insufficient_capacity);

    const co_support::sampled_source_pair_v1 invalid[] = {
        {1, 1, 0, 0, 1, 1},
    };
    result = co_support::aggregate_raw_co_support_v1(
        invalid, 1, records, 3, 32);
    assert(result.code == co_support::raw_co_support_code_v1::invalid_pair);

    result = co_support::aggregate_raw_co_support_v1(
        pairs, 4, records, 3, 1);
    assert(result.code
           == co_support::raw_co_support_code_v1::work_limit_exceeded);
    return 0;
}
