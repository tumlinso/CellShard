#pragma once

#include <CellShard/compiler/discovery/factor_topic/execution_utility_v1.hh>
#include <CellShard/compiler/evidence/confidence_stability_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::factor_topic {

struct factor_experiment_observation_v1 {
    evidence::evidence_identity_v1 candidate_identity{};
    std::uint64_t net_utility = 0;
    std::uint64_t trial_index = 0;
    std::uint32_t exact_reconstruction = 0;
    std::uint32_t reserved = 0;
};

struct factor_null_gate_config_v1 {
    std::uint64_t maximum_observations = 0;
    std::uint64_t minimum_actual_trials = 0;
    std::uint64_t minimum_trial_utility = 0;
    std::uint64_t maximum_null_exceedances = 0;
};

struct factor_null_gate_outcome_v1 {
    evidence::evidence_identity_v1 candidate_identity{};
    std::uint64_t actual_trial_count = 0;
    std::uint64_t qualified_actual_count = 0;
    std::uint64_t null_trial_count = 0;
    std::uint64_t null_exceedance_count = 0;
    std::uint64_t minimum_actual_utility = 0;
    std::uint64_t maximum_null_utility = 0;
    evidence::confidence_stability_v1 assessment{};
    factor_utility_disposition_v1 disposition =
        factor_utility_disposition_v1::no_promotion;
    std::uint32_t reserved = 0;
};

enum class factor_null_gate_code_v1 : std::uint32_t {
    evaluated = 0,
    invalid_config,
    invalid_utility,
    insufficient_actual_trials,
    observation_limit_exceeded,
    missing_actual_observations,
    missing_null_observations,
    invalid_actual_observation,
    invalid_null_observation,
    null_destination,
};

struct factor_null_gate_result_v1 {
    factor_null_gate_code_v1 code = factor_null_gate_code_v1::evaluated;
    std::uint64_t observation_index = 0;

    [[nodiscard]] constexpr bool evaluated() const noexcept {
        return code == factor_null_gate_code_v1::evaluated;
    }
};

[[nodiscard]] inline factor_null_gate_result_v1
run_bounded_factor_null_gate_v1(
    const factor_execution_utility_v1 &utility,
    const factor_experiment_observation_v1 *actual,
    std::uint64_t actual_count,
    const factor_experiment_observation_v1 *null_observations,
    std::uint64_t null_count,
    factor_null_gate_config_v1 config,
    factor_null_gate_outcome_v1 *destination) noexcept {
    if (destination == nullptr) {
        return {factor_null_gate_code_v1::null_destination};
    }
    *destination = {};
    if (config.maximum_observations == 0
        || config.minimum_actual_trials == 0
        || config.minimum_trial_utility == 0) {
        return {factor_null_gate_code_v1::invalid_config};
    }
    if (!evidence::valid_evidence_identity_v1(utility.candidate_identity)
        || utility.exact_owner_count == 0 || utility.reserved != 0) {
        return {factor_null_gate_code_v1::invalid_utility};
    }
    if (actual_count < config.minimum_actual_trials) {
        return {factor_null_gate_code_v1::insufficient_actual_trials};
    }
    if (actual_count > config.maximum_observations
        || null_count > config.maximum_observations - actual_count) {
        return {factor_null_gate_code_v1::observation_limit_exceeded};
    }
    if (actual == nullptr) {
        return {factor_null_gate_code_v1::missing_actual_observations};
    }
    if (null_count != 0 && null_observations == nullptr) {
        return {factor_null_gate_code_v1::missing_null_observations};
    }

    std::uint64_t minimum_actual = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t qualified_actual = 0;
    bool all_exact = true;
    for (std::uint64_t index = 0; index < actual_count; ++index) {
        const auto &observation = actual[index];
        if (!(observation.candidate_identity == utility.candidate_identity)
            || observation.trial_index != index
            || observation.exact_reconstruction > 1
            || observation.reserved != 0) {
            return {factor_null_gate_code_v1::invalid_actual_observation, index};
        }
        if (observation.exact_reconstruction == 0) {
            all_exact = false;
        }
        if (observation.net_utility >= config.minimum_trial_utility) {
            ++qualified_actual;
        }
        if (observation.net_utility < minimum_actual) {
            minimum_actual = observation.net_utility;
        }
    }

    std::uint64_t maximum_null = 0;
    std::uint64_t null_exceedances = 0;
    for (std::uint64_t index = 0; index < null_count; ++index) {
        const auto &observation = null_observations[index];
        if (!(observation.candidate_identity == utility.candidate_identity)
            || observation.trial_index != index
            || observation.exact_reconstruction != 1
            || observation.reserved != 0) {
            return {factor_null_gate_code_v1::invalid_null_observation, index};
        }
        if (observation.net_utility > maximum_null) {
            maximum_null = observation.net_utility;
        }
        if (observation.net_utility >= minimum_actual) {
            ++null_exceedances;
        }
    }

    const bool stable = qualified_actual >= config.minimum_actual_trials;
    const bool null_passed =
        null_exceedances <= config.maximum_null_exceedances;
    const bool utility_promoted = utility.disposition
        == factor_utility_disposition_v1::promote_proposal;
    const bool passed = all_exact && stable && null_passed && utility_promoted;
    auto assessment = evidence::evidence_assessment_v1::no_promotion;
    if (!all_exact || !stable) {
        assessment = evidence::evidence_assessment_v1::unstable_evidence;
    } else if (!null_passed) {
        assessment = evidence::evidence_assessment_v1::null_result;
    } else if (passed) {
        assessment = evidence::evidence_assessment_v1::candidate_supported;
    }
    destination->candidate_identity = utility.candidate_identity;
    destination->actual_trial_count = actual_count;
    destination->qualified_actual_count = qualified_actual;
    destination->null_trial_count = null_count;
    destination->null_exceedance_count = null_exceedances;
    destination->minimum_actual_utility = minimum_actual;
    destination->maximum_null_utility = maximum_null;
    destination->assessment = {utility.candidate_identity,
                               qualified_actual,
                               actual_count,
                               qualified_actual,
                               actual_count,
                               all_exact ? 1U : 0U,
                               1,
                               assessment,
                               0};
    destination->disposition = passed
        ? factor_utility_disposition_v1::promote_proposal
        : factor_utility_disposition_v1::no_promotion;
    return {};
}

[[nodiscard]] constexpr bool authorizes_execution(
    const factor_null_gate_outcome_v1 &) noexcept {
    return false;
}

static_assert(std::is_standard_layout<factor_experiment_observation_v1>::value);
static_assert(std::is_trivially_copyable<factor_experiment_observation_v1>::value);
static_assert(std::is_standard_layout<factor_null_gate_outcome_v1>::value);
static_assert(std::is_trivially_copyable<factor_null_gate_outcome_v1>::value);

} // namespace cellshard::compiler::discovery::factor_topic
