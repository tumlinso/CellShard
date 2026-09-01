#include <CellShard/compiler/discovery/co_support/raw_co_support_v1.hh>

#include <cassert>
#include <chrono>
#include <cstdint>

namespace co_support = cellshard::compiler::discovery::co_support;

struct benchmark_evidence_v1 {
    std::uint64_t iterations = 0;
    std::uint64_t observations = 0;
    std::uint64_t checksum = 0;
    std::uint64_t elapsed_nanoseconds = 0;
};

benchmark_evidence_v1 run_exact_oracle_fixture(bool null_fixture) {
    const std::uint64_t shared_offsets[] = {0, 3, 5, 8};
    const std::uint64_t null_offsets[] = {0, 1, 2, 3};
    const std::uint32_t shared_sources[] = {0, 1, 2, 0, 1, 0, 1, 2};
    const std::uint32_t null_sources[] = {0, 1, 2};
    const co_support::support_relation_view_v1 relation{
        null_fixture ? null_offsets : shared_offsets,
        null_fixture ? null_sources : shared_sources,
        nullptr, nullptr, null_fixture ? 3u : 8u, 4, 3,
        null_fixture ? 92u : 91u, 1};
    co_support::sampled_source_pair_v1 pairs[9]{};
    co_support::raw_co_support_record_v1 records[9]{};
    benchmark_evidence_v1 evidence{1000, 0, 0, 0};
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t iteration = 0; iteration < evidence.iterations;
         ++iteration) {
        const auto sampled = co_support::sample_high_degree_pairs_v1(
            relation, {2, 3, 64}, pairs, 9);
        assert(sampled.sampled());
        const auto aggregated = co_support::aggregate_raw_co_support_v1(
            pairs, sampled.pair_count, records, 9, 64);
        assert(aggregated.aggregated());
        evidence.observations += sampled.pair_count;
        for (std::uint64_t index = 0; index < aggregated.record_count; ++index)
            evidence.checksum += records[index].sampled_destination_count
                * (records[index].source_a + 1u)
                * (records[index].source_b + 1u);
    }
    evidence.elapsed_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
    return evidence;
}

int main() {
    const auto oracle = run_exact_oracle_fixture(false);
    const auto null = run_exact_oracle_fixture(true);
    assert(oracle.iterations == 1000);
    assert(oracle.observations == 7'000);
    assert(oracle.checksum == 24'000);
    assert(null.observations == 0);
    assert(null.checksum == 0);
    assert(oracle.elapsed_nanoseconds > 0);
    assert(null.elapsed_nanoseconds > 0);
    return 0;
}
