#include <CellShard/compiler/discovery/co_support/pair_sampling_v1.hh>

#include <cassert>

namespace co_support = cellshard::compiler::discovery::co_support;

int main() {
    const std::uint64_t offsets[] = {0, 4, 6};
    const std::uint32_t sources[] = {0, 1, 2, 3, 0, 1};
    const co_support::support_relation_view_v1 relation{
        offsets, sources, nullptr, nullptr, 6, 4, 2, 10, 1};
    co_support::sampled_source_pair_v1 pairs[4]{};
    auto result = co_support::sample_high_degree_pairs_v1(
        relation, {3, 3, 64}, pairs, 4);
    assert(result.sampled());
    assert(result.sampled_destination_count == 1);
    assert(result.pair_count == 3);
    assert(pairs[0].source_a < pairs[0].source_b);
    assert(pairs[0].inclusion_numerator == 3);
    assert(pairs[0].inclusion_denominator == 6);

    result = co_support::sample_high_degree_pairs_v1(
        relation, {3, 3, 1}, pairs, 4);
    assert(result.code == co_support::pair_sampling_code_v1::work_limit_exceeded);
    return 0;
}
