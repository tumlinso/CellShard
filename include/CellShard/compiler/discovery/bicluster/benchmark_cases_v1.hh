#pragma once

#include <CellShard/compiler/discovery/bicluster/spectral_coclustering_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::bicluster {

enum class bicluster_benchmark_scenario_v1 : std::uint32_t {
    planted = 1,
    overlapping = 2,
    condition_specific = 3,
    null_case = 4,
};

struct bicluster_benchmark_case_v1 {
    bicluster_benchmark_scenario_v1 scenario =
        bicluster_benchmark_scenario_v1::null_case;
    std::uint32_t exact_reconstruction = 0;
    std::uint64_t source_count = 0;
    std::uint64_t destination_count = 0;
    std::uint64_t condition_count = 0;
    std::uint64_t edge_count = 0;
    std::uint64_t warmup_count = 0;
    std::uint64_t repeat_count = 0;
    std::uint64_t candidate_count = 0;
    std::uint64_t proposal_count = 0;
    std::uint64_t cold_build_time_ns = 0;
    std::uint64_t peak_temporary_bytes = 0;
    std::uint64_t exact_census_time_ns = 0;
    std::uint64_t artifact_bytes = 0;
    std::uint64_t acquisition_transfer_time_ns = 0;
    std::uint64_t baseline_runtime_ns = 0;
    std::uint64_t candidate_runtime_ns = 0;
    std::uint64_t output_transform_time_ns = 0;
    std::uint64_t synchronization_time_ns = 0;
    std::uint64_t expected_reuse = 0;
    bicluster_promotion_v1 disposition = bicluster_promotion_v1::no_promotion;
    std::uint32_t reserved = 0;
};

enum class bicluster_benchmark_case_code_v1 : std::uint32_t {
    valid = 0,
    invalid_scenario,
    empty_shape,
    empty_repeats,
    missing_cost_measurement,
    invalid_exact_result,
    expected_case_not_recovered,
    null_case_promoted,
    nonzero_reserved,
};

struct bicluster_benchmark_case_validation_v1 {
    bicluster_benchmark_case_code_v1 code = bicluster_benchmark_case_code_v1::valid;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == bicluster_benchmark_case_code_v1::valid;
    }
};

[[nodiscard]] constexpr bool valid_benchmark_scenario_v1(
    bicluster_benchmark_scenario_v1 scenario) noexcept {
    const auto value = static_cast<std::uint32_t>(scenario);
    return value >= 1 && value <= 4;
}

[[nodiscard]] constexpr bicluster_benchmark_case_validation_v1
validate_bicluster_benchmark_case_v1(
    const bicluster_benchmark_case_v1 &record) noexcept {
    if (!valid_benchmark_scenario_v1(record.scenario))
        return {bicluster_benchmark_case_code_v1::invalid_scenario};
    if (record.source_count == 0 || record.destination_count == 0
        || record.condition_count == 0 || record.edge_count == 0)
        return {bicluster_benchmark_case_code_v1::empty_shape};
    if (record.warmup_count == 0 || record.repeat_count == 0)
        return {bicluster_benchmark_case_code_v1::empty_repeats};
    if (record.cold_build_time_ns == 0 || record.peak_temporary_bytes == 0
        || record.exact_census_time_ns == 0 || record.artifact_bytes == 0
        || record.baseline_runtime_ns == 0 || record.candidate_runtime_ns == 0
        || record.expected_reuse == 0)
        return {bicluster_benchmark_case_code_v1::missing_cost_measurement};
    if (record.exact_reconstruction > 1)
        return {bicluster_benchmark_case_code_v1::invalid_exact_result};
    if (record.scenario == bicluster_benchmark_scenario_v1::null_case) {
        if (record.proposal_count != 0
            || record.disposition != bicluster_promotion_v1::no_promotion)
            return {bicluster_benchmark_case_code_v1::null_case_promoted};
    } else if (record.exact_reconstruction == 0 || record.proposal_count == 0) {
        return {bicluster_benchmark_case_code_v1::expected_case_not_recovered};
    }
    if (record.reserved != 0)
        return {bicluster_benchmark_case_code_v1::nonzero_reserved};
    return {};
}

struct bicluster_benchmark_summary_v1 {
    std::uint64_t case_count = 0;
    std::uint64_t recovered_expected_cases = 0;
    std::uint64_t null_no_promotion_cases = 0;
    std::uint64_t total_candidates = 0;
    std::uint64_t total_proposals = 0;
};

enum class bicluster_benchmark_summary_code_v1 : std::uint32_t {
    summarized = 0,
    missing_cases,
    invalid_case,
    duplicate_scenario,
    incomplete_scenarios,
    null_destination,
};

struct bicluster_benchmark_summary_result_v1 {
    bicluster_benchmark_summary_code_v1 code =
        bicluster_benchmark_summary_code_v1::summarized;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool summarized() const noexcept {
        return code == bicluster_benchmark_summary_code_v1::summarized;
    }
};

[[nodiscard]] inline bicluster_benchmark_summary_result_v1
summarize_bicluster_benchmark_v1(
    const bicluster_benchmark_case_v1 *cases,
    std::uint64_t case_count,
    bicluster_benchmark_summary_v1 *destination) noexcept {
    if (destination == nullptr)
        return {bicluster_benchmark_summary_code_v1::null_destination};
    *destination = {};
    if (cases == nullptr || case_count == 0)
        return {bicluster_benchmark_summary_code_v1::missing_cases};
    std::uint32_t seen = 0;
    for (std::uint64_t index = 0; index < case_count; ++index) {
        if (!validate_bicluster_benchmark_case_v1(cases[index]).valid())
            return {bicluster_benchmark_summary_code_v1::invalid_case, index};
        const auto bit = UINT32_C(1)
            << (static_cast<std::uint32_t>(cases[index].scenario) - 1);
        if ((seen & bit) != 0)
            return {bicluster_benchmark_summary_code_v1::duplicate_scenario, index};
        seen |= bit;
        destination->total_candidates += cases[index].candidate_count;
        destination->total_proposals += cases[index].proposal_count;
        if (cases[index].scenario == bicluster_benchmark_scenario_v1::null_case)
            ++destination->null_no_promotion_cases;
        else
            ++destination->recovered_expected_cases;
    }
    if (seen != UINT32_C(0x0f)) {
        *destination = {};
        return {bicluster_benchmark_summary_code_v1::incomplete_scenarios,
                case_count};
    }
    destination->case_count = case_count;
    return {};
}

static_assert(std::is_standard_layout<bicluster_benchmark_case_v1>::value);
static_assert(std::is_trivially_copyable<bicluster_benchmark_case_v1>::value);

} // namespace cellshard::compiler::discovery::bicluster
