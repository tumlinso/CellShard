#pragma once

#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::evidence {

enum class negative_evidence_reason_v1 : std::uint32_t {
    not_observed = 1,
    contradicted = 2,
    unstable = 3,
    bounded_search_exhausted = 4,
    candidate_cap_reached = 5,
    complete_cost_nonpromotion = 6,
};

struct negative_evidence_v1 {
    evidence_identity_v1 evidence_identity{};
    evidence_identity_v1 subject_identity{};
    evidence_identity_v1 observation_scope_identity{};
    std::uint64_t observation_generation = 0;
    std::uint64_t attempted_observations = 0;
    std::uint64_t contradictory_observations = 0;
    negative_evidence_reason_v1 reason =
        negative_evidence_reason_v1::not_observed;
    std::uint32_t reserved = 0;
};

enum class negative_evidence_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_evidence_identity,
    invalid_subject_identity,
    invalid_observation_scope,
    missing_observation_generation,
    empty_attempts,
    contradiction_overflow,
    invalid_reason,
    inconsistent_contradiction,
    nonzero_reserved,
};

struct negative_evidence_validation_v1 {
    negative_evidence_validation_code_v1 code =
        negative_evidence_validation_code_v1::valid;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == negative_evidence_validation_code_v1::valid;
    }
};

[[nodiscard]] constexpr bool valid_negative_evidence_reason_v1(
    negative_evidence_reason_v1 reason) noexcept {
    const auto value = static_cast<std::uint32_t>(reason);
    return value >= 1 && value <= 6;
}

[[nodiscard]] constexpr negative_evidence_validation_v1
validate_negative_evidence_v1(const negative_evidence_v1 &record) noexcept {
    if (!valid_evidence_identity_v1(record.evidence_identity))
        return {negative_evidence_validation_code_v1::invalid_evidence_identity};
    if (!valid_evidence_identity_v1(record.subject_identity))
        return {negative_evidence_validation_code_v1::invalid_subject_identity};
    if (!valid_evidence_identity_v1(record.observation_scope_identity))
        return {negative_evidence_validation_code_v1::invalid_observation_scope};
    if (record.observation_generation == 0)
        return {negative_evidence_validation_code_v1::missing_observation_generation};
    if (record.attempted_observations == 0)
        return {negative_evidence_validation_code_v1::empty_attempts};
    if (record.contradictory_observations > record.attempted_observations)
        return {negative_evidence_validation_code_v1::contradiction_overflow};
    if (!valid_negative_evidence_reason_v1(record.reason))
        return {negative_evidence_validation_code_v1::invalid_reason};
    if (record.reason == negative_evidence_reason_v1::contradicted
        && record.contradictory_observations == 0)
        return {negative_evidence_validation_code_v1::inconsistent_contradiction};
    if (record.reserved != 0)
        return {negative_evidence_validation_code_v1::nonzero_reserved};
    return {};
}

// A bounded failure to observe recurrence is not proof of exact absence.
[[nodiscard]] constexpr bool certifies_absence(const negative_evidence_v1 &) noexcept {
    return false;
}

static_assert(std::is_standard_layout<negative_evidence_v1>::value);
static_assert(std::is_trivially_copyable<negative_evidence_v1>::value);

} // namespace cellshard::compiler::evidence
