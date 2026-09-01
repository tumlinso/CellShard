#pragma once

#include <CellShard/compiler/discovery/multimodal/exact_certification_v1.hh>

#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>

namespace cellshard::compiler::discovery::multimodal {

struct modality_map_null_summary_v1 {
    std::uint64_t observed_score = 0;
    std::uint64_t trial_count = 0;
    std::uint64_t null_at_least_observed_count = 0;
    std::uint64_t p_value_numerator = 0;
    std::uint64_t p_value_denominator = 1;
};

struct multimodal_promotion_policy_v1 {
    std::uint64_t maximum_p_value_numerator = 1;
    std::uint64_t maximum_p_value_denominator = 20;
    std::uint64_t minimum_checked_subject_count = 1;
    std::uint64_t minimum_checked_element_count = 1;
    std::uint32_t minimum_payload_count = 2;
    std::uint32_t reserved = 0;
};

enum class multimodal_promotion_reason_v1 : std::uint32_t {
    promoted = 0,
    invalid_null_trials,
    invalid_policy,
    uncertified_atom,
    insufficient_payloads,
    insufficient_exact_coverage,
    null_not_rejected,
};

struct multimodal_promotion_result_v1 {
    multimodal_promotion_reason_v1 reason
        = multimodal_promotion_reason_v1::promoted;
    std::uint32_t promoted = 1;
    modality_map_null_summary_v1 null_summary{};
};

[[nodiscard]] inline bool positive_fraction_greater_v1(
    std::uint64_t left_numerator,
    std::uint64_t left_denominator,
    std::uint64_t right_numerator,
    std::uint64_t right_denominator) noexcept {
    bool reversed = false;
    while (true) {
        const auto left_q = left_numerator / left_denominator;
        const auto right_q = right_numerator / right_denominator;
        if (left_q != right_q)
            return reversed ? left_q < right_q : left_q > right_q;
        const auto left_r = left_numerator % left_denominator;
        const auto right_r = right_numerator % right_denominator;
        if (left_r == 0 || right_r == 0) {
            if (left_r == right_r) return false;
            return reversed ? left_r == 0 : right_r == 0;
        }
        left_numerator = left_denominator;
        left_denominator = left_r;
        right_numerator = right_denominator;
        right_denominator = right_r;
        reversed = !reversed;
    }
}

[[nodiscard]] inline multimodal_promotion_result_v1
run_modality_map_null_promotion_gate_v1(
    std::uint64_t observed_score,
    const std::uint64_t *null_scores,
    std::uint64_t trial_count,
    const multimodal_exact_certificate_v1 &certificate,
    const multi_payload_atom_v1 &atom,
    multimodal_promotion_policy_v1 policy) noexcept {
    multimodal_promotion_result_v1 result{};
    result.promoted = 0;
    result.null_summary.observed_score = observed_score;
    result.null_summary.trial_count = trial_count;
    if (observed_score == 0 || trial_count == 0 || null_scores == nullptr
        || trial_count == std::numeric_limits<std::uint64_t>::max()) {
        result.reason = multimodal_promotion_reason_v1::invalid_null_trials;
        return result;
    }
    if (policy.maximum_p_value_numerator == 0
        || policy.maximum_p_value_denominator == 0
        || policy.minimum_payload_count < 2) {
        result.reason = multimodal_promotion_reason_v1::invalid_policy;
        return result;
    }
    for (std::uint64_t trial = 0; trial < trial_count; ++trial)
        if (null_scores[trial] >= observed_score)
            ++result.null_summary.null_at_least_observed_count;
    auto numerator = result.null_summary.null_at_least_observed_count + 1;
    auto denominator = trial_count + 1;
    const auto divisor = std::gcd(numerator, denominator);
    result.null_summary.p_value_numerator = numerator / divisor;
    result.null_summary.p_value_denominator = denominator / divisor;
    if (certificate.certified != 1
        || certificate.atom_identity != atom.atom_identity
        || certificate.evidence_identity != atom.evidence_identity
        || certificate.spine_identity != atom.spine_identity
        || certificate.structure_epoch != atom.structure_epoch) {
        result.reason = multimodal_promotion_reason_v1::uncertified_atom;
        return result;
    }
    if (certificate.checked_payload_count < policy.minimum_payload_count
        || atom.payload_count < policy.minimum_payload_count) {
        result.reason = multimodal_promotion_reason_v1::insufficient_payloads;
        return result;
    }
    if (certificate.checked_subject_count
            < policy.minimum_checked_subject_count
        || certificate.checked_element_count
            < policy.minimum_checked_element_count) {
        result.reason
            = multimodal_promotion_reason_v1::insufficient_exact_coverage;
        return result;
    }
    if (positive_fraction_greater_v1(
            result.null_summary.p_value_numerator,
            result.null_summary.p_value_denominator,
            policy.maximum_p_value_numerator,
            policy.maximum_p_value_denominator)) {
        result.reason = multimodal_promotion_reason_v1::null_not_rejected;
        return result;
    }
    result.reason = multimodal_promotion_reason_v1::promoted;
    result.promoted = 1;
    return result;
}

static_assert(std::is_standard_layout<modality_map_null_summary_v1>::value);
static_assert(std::is_trivially_copyable<modality_map_null_summary_v1>::value);
static_assert(std::is_standard_layout<multimodal_promotion_policy_v1>::value);
static_assert(std::is_trivially_copyable<multimodal_promotion_policy_v1>::value);

} // namespace cellshard::compiler::discovery::multimodal
