#include <CellShard/compiler/discovery/bicluster/benchmark_cases_v1.hh>

#include <cstdint>
#include <iostream>

namespace bicluster = cellshard::compiler::discovery::bicluster;

bicluster::bicluster_benchmark_case_v1 fixture(
    bicluster::bicluster_benchmark_scenario_v1 scenario,
    std::uint64_t candidates,
    std::uint64_t proposals) {
    return {scenario, 1, 128, 64, 2, 1024, 4, 16, candidates, proposals,
            1000, 16384, 500, 4096, 250, 1200, 700, 100, 50, 8,
            proposals == 0 ? bicluster::bicluster_promotion_v1::no_promotion
                           : bicluster::bicluster_promotion_v1::promote_proposal,
            0};
}

int main() {
    const bicluster::bicluster_benchmark_case_v1 cases[] = {
        fixture(bicluster::bicluster_benchmark_scenario_v1::planted, 3, 1),
        fixture(bicluster::bicluster_benchmark_scenario_v1::overlapping, 5, 2),
        fixture(bicluster::bicluster_benchmark_scenario_v1::condition_specific, 4, 1),
        fixture(bicluster::bicluster_benchmark_scenario_v1::null_case, 2, 0),
    };
    bicluster::bicluster_benchmark_summary_v1 summary{};
    const auto result = bicluster::summarize_bicluster_benchmark_v1(
        cases, 4, &summary);
    if (!result.summarized()) return 1;
    std::cout << "{\"fixture_only\":true,\"cases\":" << summary.case_count
              << ",\"recovered_expected\":" << summary.recovered_expected_cases
              << ",\"null_no_promotion\":" << summary.null_no_promotion_cases
              << ",\"candidates\":" << summary.total_candidates
              << ",\"proposals\":" << summary.total_proposals << "}\n";
    return 0;
}
