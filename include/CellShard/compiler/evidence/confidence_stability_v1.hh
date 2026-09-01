#pragma once

#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::evidence {

enum class evidence_assessment_v1 : std::uint32_t {
    candidate_supported = 1,
    weak_evidence = 2,
    unstable_evidence = 3,
    null_result = 4,
    no_promotion = 5,
};

struct confidence_stability_v1 {
    evidence_identity_v1 evidence_identity{};
    std::uint64_t confidence_numerator = 0;
    std::uint64_t confidence_denominator = 0;
    std::uint64_t stable_resamples = 0;
    std::uint64_t total_resamples = 0;
    std::uint64_t supporting_strata = 0;
    std::uint64_t observed_strata = 0;
    evidence_assessment_v1 assessment = evidence_assessment_v1::null_result;
    std::uint32_t reserved = 0;
};

enum class confidence_stability_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_evidence_identity,
    invalid_confidence,
    invalid_stability,
    invalid_strata,
    invalid_assessment,
    nonzero_reserved,
};

struct confidence_stability_validation_v1 {
    confidence_stability_validation_code_v1 code =
        confidence_stability_validation_code_v1::valid;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == confidence_stability_validation_code_v1::valid;
    }
};

[[nodiscard]] constexpr bool valid_evidence_assessment_v1(
    evidence_assessment_v1 assessment) noexcept {
    const auto value = static_cast<std::uint32_t>(assessment);
    return value >= 1 && value <= 5;
}

[[nodiscard]] constexpr confidence_stability_validation_v1
validate_confidence_stability_v1(
    const confidence_stability_v1 &record) noexcept {
    if (!valid_evidence_identity_v1(record.evidence_identity)) {
        return {confidence_stability_validation_code_v1::
                    invalid_evidence_identity};
    }
    if (record.confidence_denominator == 0
        || record.confidence_numerator > record.confidence_denominator) {
        return {confidence_stability_validation_code_v1::invalid_confidence};
    }
    if (record.total_resamples == 0
        || record.stable_resamples > record.total_resamples) {
        return {confidence_stability_validation_code_v1::invalid_stability};
    }
    if (record.observed_strata == 0
        || record.supporting_strata > record.observed_strata) {
        return {confidence_stability_validation_code_v1::invalid_strata};
    }
    if (!valid_evidence_assessment_v1(record.assessment)) {
        return {confidence_stability_validation_code_v1::invalid_assessment};
    }
    if (record.reserved != 0) {
        return {confidence_stability_validation_code_v1::nonzero_reserved};
    }
    return {};
}

// Assessments are descriptive proposal outcomes only. In particular,
// candidate_supported is not exact coverage, contribution ownership,
// placement, or execution authorization.
[[nodiscard]] constexpr bool authorizes_execution(
    const confidence_stability_v1 &) noexcept {
    return false;
}

static_assert(std::is_standard_layout<confidence_stability_v1>::value);
static_assert(std::is_trivially_copyable<confidence_stability_v1>::value);

} // namespace cellshard::compiler::evidence
