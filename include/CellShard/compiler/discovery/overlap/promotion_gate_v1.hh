#pragma once

#include <CellShard/compiler/discovery/overlap/certification_candidate_v1.hh>
#include <CellShard/compiler/discovery/overlap/expansion_v1.hh>
#include <CellShard/compiler/discovery/overlap/stability_cost_v1.hh>

#include <cstdint>
#include <limits>

namespace cellshard::compiler::discovery::overlap {

inline constexpr std::uint32_t overlap_promotion_gate_contract_version_v1 = 1;

enum class zero_overlap_equivalence_code_v1 : std::uint32_t {
    equivalent = 0,
    invalid_baseline,
    baseline_not_disjoint,
    invalid_candidate,
    member_spine_mismatch,
    community_spine_mismatch,
    source_mismatch,
    candidate_not_disjoint,
    assignment_mismatch,
    nonexact_weight,
};

struct zero_overlap_equivalence_v1 {
    zero_overlap_equivalence_code_v1 code =
        zero_overlap_equivalence_code_v1::equivalent;
    std::uint64_t member_index = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool equivalent() const noexcept {
        return code == zero_overlap_equivalence_code_v1::equivalent;
    }
};

struct overlap_promotion_policy_v1 {
    std::uint64_t minimum_stability_numerator = 0;
    std::uint64_t minimum_stability_denominator = 0;
    std::uint64_t maximum_additional_memberships_per_member = 0;
    std::uint64_t maximum_complete_duplication_cost_bytes = 0;
};

enum class overlap_promotion_decision_v1 : std::uint32_t {
    baseline_equivalent = 1,
    promote_to_independent_certification = 2,
    reject_unstable = 3,
    reject_overlap_bound = 4,
    reject_duplication_cost = 5,
};

enum class overlap_promotion_gate_code_v1 : std::uint32_t {
    evaluated = 0,
    invalid_baseline,
    invalid_candidate,
    spine_mismatch,
    invalid_score,
    invalid_policy,
    invalid_exact_candidates,
    provider_self_certification,
    exact_member_count_overflow,
    exact_member_count_mismatch,
};

struct overlap_promotion_gate_result_v1 {
    overlap_promotion_gate_code_v1 code =
        overlap_promotion_gate_code_v1::evaluated;
    overlap_promotion_decision_v1 decision =
        overlap_promotion_decision_v1::reject_unstable;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool evaluated() const noexcept {
        return code == overlap_promotion_gate_code_v1::evaluated;
    }
    [[nodiscard]] constexpr bool promoted() const noexcept {
        return evaluated()
            && (decision == overlap_promotion_decision_v1::baseline_equivalent
                || decision
                    == overlap_promotion_decision_v1::
                        promote_to_independent_certification);
    }
};

// Linear exact proof that the common overlap representation degenerates to the
// disjoint baseline without changing member, community, source, assignment, or
// unit weight identity.
[[nodiscard]] inline zero_overlap_equivalence_v1
prove_zero_overlap_equivalence_v1(
    bounded_overlap_membership_view_v1 baseline,
    bounded_overlap_membership_view_v1 candidate) noexcept {
    const auto baseline_validation =
        validate_bounded_overlap_membership_v1(baseline);
    if (!baseline_validation.valid()) {
        return {zero_overlap_equivalence_code_v1::invalid_baseline,
                baseline_validation.member_index,
                static_cast<std::uint32_t>(baseline_validation.code)};
    }
    if (baseline.maximum_memberships_per_member != 1
        || baseline.membership_count != baseline.member_count) {
        return {zero_overlap_equivalence_code_v1::baseline_not_disjoint};
    }
    const auto candidate_validation =
        validate_bounded_overlap_membership_v1(candidate);
    if (!candidate_validation.valid()) {
        return {zero_overlap_equivalence_code_v1::invalid_candidate,
                candidate_validation.member_index,
                static_cast<std::uint32_t>(candidate_validation.code)};
    }
    std::uint64_t mismatch = 0;
    bool community_mismatch = false;
    if (!same_overlap_spines_v1(
            baseline, candidate, &mismatch, &community_mismatch)) {
        return {community_mismatch
                    ? zero_overlap_equivalence_code_v1::
                          community_spine_mismatch
                    : zero_overlap_equivalence_code_v1::member_spine_mismatch,
                mismatch};
    }
    if (!(baseline.source_identity == candidate.source_identity)) {
        return {zero_overlap_equivalence_code_v1::source_mismatch};
    }
    if (candidate.membership_count != candidate.member_count) {
        return {zero_overlap_equivalence_code_v1::candidate_not_disjoint};
    }
    for (std::uint64_t member_index = 0;
         member_index < baseline.member_count;
         ++member_index) {
        const auto &expected =
            baseline.memberships[baseline.member_offsets[member_index]];
        const auto &actual =
            candidate.memberships[candidate.member_offsets[member_index]];
        if (actual.local_community_index != expected.local_community_index) {
            return {zero_overlap_equivalence_code_v1::assignment_mismatch,
                    member_index};
        }
        if (actual.weight_numerator != actual.weight_denominator) {
            return {zero_overlap_equivalence_code_v1::nonexact_weight,
                    member_index};
        }
    }
    return {zero_overlap_equivalence_code_v1::equivalent,
            baseline.member_count,
            0};
}

[[nodiscard]] inline overlap_promotion_gate_result_v1 gate_overlap_promotion_v1(
    bounded_overlap_membership_view_v1 baseline,
    bounded_overlap_membership_view_v1 candidate,
    overlap_stability_cost_v1 score,
    exact_overlap_candidate_table_v1 exact_candidates,
    overlap_promotion_policy_v1 policy) noexcept {
    const auto baseline_validation =
        validate_bounded_overlap_membership_v1(baseline);
    if (!baseline_validation.valid()) {
        return {overlap_promotion_gate_code_v1::invalid_baseline};
    }
    const auto candidate_validation =
        validate_bounded_overlap_membership_v1(candidate);
    if (!candidate_validation.valid()) {
        return {overlap_promotion_gate_code_v1::invalid_candidate};
    }
    std::uint64_t mismatch = 0;
    bool community_mismatch = false;
    if (!same_overlap_spines_v1(
            baseline, candidate, &mismatch, &community_mismatch)
        || !(baseline.source_identity == candidate.source_identity)) {
        return {overlap_promotion_gate_code_v1::spine_mismatch,
                overlap_promotion_decision_v1::reject_unstable,
                mismatch};
    }
    const auto equivalence =
        prove_zero_overlap_equivalence_v1(baseline, candidate);
    if (equivalence.equivalent()) {
        return {overlap_promotion_gate_code_v1::evaluated,
                overlap_promotion_decision_v1::baseline_equivalent,
                candidate.member_count};
    }
    if (score.stability_union_count == 0
        || score.stability_intersection_count > score.stability_union_count
        || score.additional_membership_count
               != candidate.membership_count - candidate.member_count) {
        return {overlap_promotion_gate_code_v1::invalid_score};
    }
    if (policy.minimum_stability_denominator == 0
        || policy.minimum_stability_numerator
               > policy.minimum_stability_denominator
        || policy.maximum_additional_memberships_per_member == 0) {
        return {overlap_promotion_gate_code_v1::invalid_policy};
    }
    if (exact_candidates.candidate_count == 0
        || exact_candidates.candidates == nullptr
        || !atom::validate_atom_persistent_identity_v1(
                exact_candidates.proposal_provider_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(
                exact_candidates.certification_authority_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(
                exact_candidates.canonical_source_identity)
                .valid()
        || exact_candidates.canonical_source_generation == 0) {
        return {overlap_promotion_gate_code_v1::invalid_exact_candidates};
    }
    if (exact_candidates.proposal_provider_identity
        == exact_candidates.certification_authority_identity) {
        return {overlap_promotion_gate_code_v1::provider_self_certification};
    }
    std::uint64_t exact_member_count = 0;
    for (std::uint64_t index = 0;
         index < exact_candidates.candidate_count;
         ++index) {
        const auto &exact = exact_candidates.candidates[index];
        if (exact.member_count == 0 || exact.global_member_ids == nullptr
            || !atom::validate_atom_persistent_identity_v1(
                    exact.candidate_identity)
                    .valid()
            || !evidence::valid_evidence_identity_v1(
                exact.community_identity)) {
            return {overlap_promotion_gate_code_v1::invalid_exact_candidates,
                    overlap_promotion_decision_v1::reject_unstable,
                    index};
        }
        for (std::uint64_t member_index = 0;
             member_index < exact.member_count;
             ++member_index) {
            if (exact.global_member_ids[member_index] == 0
                || (member_index != 0
                    && exact.global_member_ids[member_index - 1]
                        >= exact.global_member_ids[member_index])) {
                return {overlap_promotion_gate_code_v1::
                            invalid_exact_candidates,
                        overlap_promotion_decision_v1::reject_unstable,
                        index};
            }
        }
        if (exact.member_count > UINT64_MAX - exact_member_count) {
            return {overlap_promotion_gate_code_v1::
                        exact_member_count_overflow};
        }
        exact_member_count += exact.member_count;
    }
    if (exact_member_count != candidate.membership_count) {
        return {overlap_promotion_gate_code_v1::exact_member_count_mismatch,
                overlap_promotion_decision_v1::reject_unstable,
                exact_member_count};
    }
    const bool below_stability = overlap_fraction_greater_v1(
        policy.minimum_stability_numerator,
        policy.minimum_stability_denominator,
        score.stability_intersection_count,
        score.stability_union_count);
    if (below_stability) {
        return {overlap_promotion_gate_code_v1::evaluated,
                overlap_promotion_decision_v1::reject_unstable};
    }
    if (candidate.member_count
        > UINT64_MAX / policy.maximum_additional_memberships_per_member) {
        return {overlap_promotion_gate_code_v1::invalid_policy};
    }
    if (score.additional_membership_count
        > candidate.member_count
              * policy.maximum_additional_memberships_per_member) {
        return {overlap_promotion_gate_code_v1::evaluated,
                overlap_promotion_decision_v1::reject_overlap_bound};
    }
    if (score.complete_duplication_cost_bytes
        > policy.maximum_complete_duplication_cost_bytes) {
        return {overlap_promotion_gate_code_v1::evaluated,
                overlap_promotion_decision_v1::reject_duplication_cost};
    }
    return {overlap_promotion_gate_code_v1::evaluated,
            overlap_promotion_decision_v1::
                promote_to_independent_certification,
            exact_member_count};
}

[[nodiscard]] constexpr bool authorizes_execution(
    overlap_promotion_gate_result_v1) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::overlap
