#include <CellShard/compiler/discovery/co_support/normalized_association_v1.hh>

#include <cassert>

namespace co_support = cellshard::compiler::discovery::co_support;

int main() {
    const co_support::weighted_co_support_record_v1 records[] = {
        {0, 1, 4, 10}, {0, 2, 3, 8},
    };
    const std::uint64_t prevalence[] = {6, 4, 9};
    co_support::normalized_association_record_v1 output[2]{};
    auto result = co_support::compute_normalized_association_v1(
        records, 2, prevalence, 3, output, 2);
    assert(result.computed());
    assert(output[0].raw_numerator == 1);
    assert(output[0].raw_denominator == 1);
    assert(output[0].weighted_numerator == 5);
    assert(output[0].weighted_denominator == 2);
    assert(output[1].raw_numerator == 1);
    assert(output[1].raw_denominator == 2);
    assert(output[1].weighted_numerator == 4);
    assert(output[1].weighted_denominator == 3);

    const co_support::weighted_co_support_record_v1 excessive[] = {
        {0, 1, 5, 5},
    };
    result = co_support::compute_normalized_association_v1(
        excessive, 1, prevalence, 3, output, 2);
    assert(result.code == co_support::normalized_association_code_v1::
        support_exceeds_prevalence);
    return 0;
}
