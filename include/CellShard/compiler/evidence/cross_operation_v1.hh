#pragma once

#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::evidence {

enum class cross_operation_relation_v1 : std::uint32_t {
    co_access = 1,
    producer_consumer = 2,
    persistent_order = 3,
    shared_structure_proposal = 4,
};

struct cross_operation_evidence_v1 {
    evidence_identity_v1 evidence_identity{};
    evidence_identity_v1 subject_atom_identity{};
    evidence_identity_v1 first_operation_identity{};
    evidence_identity_v1 second_operation_identity{};
    evidence_identity_v1 workload_scope_identity{};
    std::uint64_t observation_generation = 0;
    std::uint64_t joint_observations = 0;
    std::uint64_t total_observations = 0;
    cross_operation_relation_v1 relation =
        cross_operation_relation_v1::co_access;
    evidence_disposition_v1 disposition = evidence_disposition_v1::proposal_only;
};

enum class cross_operation_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_identity,
    identical_or_unordered_operations,
    missing_observation_generation,
    invalid_observation_count,
    invalid_relation,
    non_proposal_disposition,
};

struct cross_operation_validation_v1 {
    cross_operation_validation_code_v1 code = cross_operation_validation_code_v1::valid;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == cross_operation_validation_code_v1::valid;
    }
};

[[nodiscard]] constexpr cross_operation_validation_v1
validate_cross_operation_evidence_v1(
    const cross_operation_evidence_v1 &record) noexcept {
    if (!valid_evidence_identity_v1(record.evidence_identity)
        || !valid_evidence_identity_v1(record.subject_atom_identity)
        || !valid_evidence_identity_v1(record.first_operation_identity)
        || !valid_evidence_identity_v1(record.second_operation_identity)
        || !valid_evidence_identity_v1(record.workload_scope_identity))
        return {cross_operation_validation_code_v1::invalid_identity};
    if (!evidence_identity_less_v1(record.first_operation_identity,
                                   record.second_operation_identity))
        return {cross_operation_validation_code_v1::identical_or_unordered_operations};
    if (record.observation_generation == 0)
        return {cross_operation_validation_code_v1::missing_observation_generation};
    if (record.total_observations == 0
        || record.joint_observations > record.total_observations)
        return {cross_operation_validation_code_v1::invalid_observation_count};
    const auto relation = static_cast<std::uint32_t>(record.relation);
    if (relation < 1 || relation > 4)
        return {cross_operation_validation_code_v1::invalid_relation};
    if (record.disposition != evidence_disposition_v1::proposal_only)
        return {cross_operation_validation_code_v1::non_proposal_disposition};
    return {};
}

// Cross-operation recurrence says nothing about physical representation
// overlap or exact execution-contribution ownership.
[[nodiscard]] constexpr bool implies_representation_overlap(
    const cross_operation_evidence_v1 &) noexcept { return false; }
[[nodiscard]] constexpr bool implies_contribution_overlap(
    const cross_operation_evidence_v1 &) noexcept { return false; }

static_assert(std::is_standard_layout<cross_operation_evidence_v1>::value);
static_assert(std::is_trivially_copyable<cross_operation_evidence_v1>::value);

} // namespace cellshard::compiler::evidence
