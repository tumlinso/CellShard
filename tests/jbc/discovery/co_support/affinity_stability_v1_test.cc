#include <CellShard/compiler/discovery/co_support/affinity_stability_v1.hh>

#include <cassert>

namespace co_support = cellshard::compiler::discovery::co_support;

int main() {
    const co_support::affinity_observation_v1 observations[] = {
        {0, 0, 0, 1}, {0, 1, 0, 1}, {1, 0, 0, 1},
        {2, 1, 0, 2},
    };
    co_support::affinity_stability_record_v1 records[2]{};
    auto result = co_support::compute_affinity_stability_v1(
        observations, 4, 3, 4, 2, records, 2, 32);
    assert(result.computed());
    assert(result.record_count == 2);
    assert(records[0].source_id == 0 && records[0].neighbor_source_id == 1);
    assert(records[0].resample_presence_count == 2);
    assert(records[0].resample_numerator == 1);
    assert(records[0].resample_denominator == 2);
    assert(records[0].stratum_presence_count == 2);
    assert(records[0].stratum_numerator == 1);
    assert(records[0].stratum_denominator == 1);

    const co_support::affinity_observation_v1 duplicate[] = {
        {0, 0, 0, 1}, {0, 0, 0, 1},
    };
    result = co_support::compute_affinity_stability_v1(
        duplicate, 2, 3, 1, 1, records, 2, 8);
    assert(result.code
           == co_support::affinity_stability_code_v1::duplicate_observation);
    return 0;
}
