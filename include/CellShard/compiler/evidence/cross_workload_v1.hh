#pragma once

#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::evidence {

struct cross_workload_evidence_view_v1 {
    const evidence_identity_v1 *workload_identities = nullptr;
    std::uint64_t workload_count = 0;
    std::uint64_t workload_capacity = 0;
    evidence_identity_v1 evidence_identity{};
    evidence_identity_v1 subject_atom_identity{};
    evidence_identity_v1 graph_family_identity{};
    std::uint64_t observation_generation = 0;
    evidence_disposition_v1 disposition = evidence_disposition_v1::proposal_only;
    std::uint32_t reserved = 0;
};

enum class cross_workload_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_identity,
    insufficient_workloads,
    missing_workloads,
    capacity_overflow,
    invalid_workload_identity,
    unordered_or_duplicate_workload,
    missing_observation_generation,
    non_proposal_disposition,
    nonzero_reserved,
};

struct cross_workload_validation_v1 {
    cross_workload_validation_code_v1 code = cross_workload_validation_code_v1::valid;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == cross_workload_validation_code_v1::valid;
    }
};

[[nodiscard]] constexpr cross_workload_validation_v1
validate_cross_workload_evidence_v1(
    cross_workload_evidence_view_v1 view) noexcept {
    if (!valid_evidence_identity_v1(view.evidence_identity)
        || !valid_evidence_identity_v1(view.subject_atom_identity)
        || !valid_evidence_identity_v1(view.graph_family_identity))
        return {cross_workload_validation_code_v1::invalid_identity, 0};
    if (view.workload_count < 2)
        return {cross_workload_validation_code_v1::insufficient_workloads, 0};
    if (view.workload_identities == nullptr)
        return {cross_workload_validation_code_v1::missing_workloads, 0};
    if (view.workload_count > view.workload_capacity)
        return {cross_workload_validation_code_v1::capacity_overflow, 0};
    for (std::uint64_t index = 0; index < view.workload_count; ++index) {
        if (!valid_evidence_identity_v1(view.workload_identities[index]))
            return {cross_workload_validation_code_v1::invalid_workload_identity, index};
        if (index != 0 && !evidence_identity_less_v1(
                view.workload_identities[index - 1],
                view.workload_identities[index]))
            return {cross_workload_validation_code_v1::unordered_or_duplicate_workload, index};
    }
    if (view.observation_generation == 0)
        return {cross_workload_validation_code_v1::missing_observation_generation, 0};
    if (view.disposition != evidence_disposition_v1::proposal_only)
        return {cross_workload_validation_code_v1::non_proposal_disposition, 0};
    if (view.reserved != 0)
        return {cross_workload_validation_code_v1::nonzero_reserved, 0};
    return {cross_workload_validation_code_v1::valid, view.workload_count};
}

[[nodiscard]] constexpr bool certifies_graph_equivalence(
    cross_workload_evidence_view_v1) noexcept { return false; }

static_assert(offsetof(cross_workload_evidence_view_v1, workload_identities) == 0);
static_assert(std::is_standard_layout<cross_workload_evidence_view_v1>::value);
static_assert(std::is_trivially_copyable<cross_workload_evidence_view_v1>::value);

} // namespace cellshard::compiler::evidence
