#include <CellShard/compiler/discovery/bicluster/benchmark_cases_v1.hh>

#include <cassert>

namespace bicluster = cellshard::compiler::discovery::bicluster;

bicluster::bicluster_benchmark_case_v1 make_case(
    bicluster::bicluster_benchmark_scenario_v1 scenario,
    std::uint64_t proposals) {
    bicluster::bicluster_benchmark_case_v1 value{};
    value.scenario = scenario;
    value.exact_reconstruction = 1;
    value.source_count = 64;
    value.destination_count = 32;
    value.condition_count = 2;
    value.edge_count = 256;
    value.warmup_count = 2;
    value.repeat_count = 8;
    value.candidate_count = proposals + 2;
    value.proposal_count = proposals;
    value.cold_build_time_ns = 100;
    value.peak_temporary_bytes = 4096;
    value.exact_census_time_ns = 50;
    value.artifact_bytes = 1024;
    value.acquisition_transfer_time_ns = 10;
    value.baseline_runtime_ns = 200;
    value.candidate_runtime_ns = 100;
    value.output_transform_time_ns = 5;
    value.synchronization_time_ns = 5;
    value.expected_reuse = 4;
    value.disposition = proposals == 0
        ? bicluster::bicluster_promotion_v1::no_promotion
        : bicluster::bicluster_promotion_v1::promote_proposal;
    return value;
}

int main() {
    const bicluster::bicluster_benchmark_case_v1 cases[] = {
        make_case(bicluster::bicluster_benchmark_scenario_v1::planted, 1),
        make_case(bicluster::bicluster_benchmark_scenario_v1::overlapping, 2),
        make_case(bicluster::bicluster_benchmark_scenario_v1::condition_specific, 1),
        make_case(bicluster::bicluster_benchmark_scenario_v1::null_case, 0),
    };
    bicluster::bicluster_benchmark_summary_v1 summary{};
    auto result = bicluster::summarize_bicluster_benchmark_v1(
        cases, 4, &summary);
    assert(result.summarized());
    assert(summary.case_count == 4);
    assert(summary.recovered_expected_cases == 3);
    assert(summary.null_no_promotion_cases == 1);
    assert(summary.total_proposals == 4);

    auto invalid_null = cases[3];
    invalid_null.proposal_count = 1;
    invalid_null.disposition = bicluster::bicluster_promotion_v1::promote_proposal;
    assert(bicluster::validate_bicluster_benchmark_case_v1(invalid_null).code
           == bicluster::bicluster_benchmark_case_code_v1::null_case_promoted);
    return 0;
}
