#pragma once

#include <CellShard/compiler/discovery/overlap/disjoint_baseline_v1.hh>

#include <cstdint>
#include <limits>

namespace cellshard::compiler::discovery::overlap {

inline constexpr std::uint32_t overlap_expansion_contract_version_v1 = 1;

struct overlap_expansion_candidate_v1 {
    std::uint64_t global_member_id = 0;
    std::uint32_t local_community_index = 0;
    std::uint32_t reserved = 0;
    std::uint64_t score_numerator = 0;
    std::uint64_t score_denominator = 0;
};

struct overlap_expansion_buffers_v1 {
    std::uint64_t *member_offsets = nullptr;
    std::uint64_t offset_capacity = 0;
    overlap_membership_v1 *memberships = nullptr;
    std::uint64_t membership_capacity = 0;
    overlap_membership_v1 *top_l_scratch = nullptr;
    std::uint64_t scratch_capacity = 0;
};

enum class overlap_expansion_code_v1 : std::uint32_t {
    built = 0,
    invalid_baseline,
    baseline_not_disjoint,
    invalid_evidence_identity,
    missing_generation,
    invalid_overlap_bound,
    missing_candidates,
    invalid_candidate_member,
    invalid_candidate_community,
    invalid_candidate_score,
    nonzero_reserved,
    unordered_or_duplicate_candidate,
    candidate_member_not_in_baseline,
    capacity_overflow,
    missing_output,
    insufficient_output,
    missing_scratch,
    insufficient_scratch,
    invalid_built_view,
};

struct overlap_expansion_result_v1 {
    overlap_expansion_code_v1 code = overlap_expansion_code_v1::built;
    bounded_overlap_membership_view_v1 view{};
    std::uint64_t index = 0;
    std::uint64_t required_offset_capacity = 0;
    std::uint64_t required_membership_capacity = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool built() const noexcept {
        return code == overlap_expansion_code_v1::built;
    }
};

// Exact fraction comparison without cross-product overflow. Continued-fraction
// quotients alternate ordering after each reciprocal step.
[[nodiscard]] constexpr bool overlap_fraction_greater_v1(
    std::uint64_t lhs_numerator,
    std::uint64_t lhs_denominator,
    std::uint64_t rhs_numerator,
    std::uint64_t rhs_denominator) noexcept {
    bool reversed = false;
    while (true) {
        const auto lhs_quotient = lhs_numerator / lhs_denominator;
        const auto rhs_quotient = rhs_numerator / rhs_denominator;
        if (lhs_quotient != rhs_quotient) {
            return reversed ? lhs_quotient < rhs_quotient
                            : lhs_quotient > rhs_quotient;
        }
        const auto lhs_remainder = lhs_numerator % lhs_denominator;
        const auto rhs_remainder = rhs_numerator % rhs_denominator;
        if (lhs_remainder == 0 || rhs_remainder == 0) {
            if (lhs_remainder == rhs_remainder) {
                return false;
            }
            return lhs_remainder == 0 ? reversed : !reversed;
        }
        lhs_numerator = lhs_denominator;
        lhs_denominator = lhs_remainder;
        rhs_numerator = rhs_denominator;
        rhs_denominator = rhs_remainder;
        reversed = !reversed;
    }
}

[[nodiscard]] constexpr bool overlap_membership_ranked_before_v1(
    const overlap_membership_v1 &lhs,
    const overlap_membership_v1 &rhs) noexcept {
    if (overlap_fraction_greater_v1(lhs.weight_numerator,
                                    lhs.weight_denominator,
                                    rhs.weight_numerator,
                                    rhs.weight_denominator)) {
        return true;
    }
    if (overlap_fraction_greater_v1(rhs.weight_numerator,
                                    rhs.weight_denominator,
                                    lhs.weight_numerator,
                                    lhs.weight_denominator)) {
        return false;
    }
    return lhs.local_community_index < rhs.local_community_index;
}

// Keeps the disjoint baseline assignment and at most L-1 strongest additional
// memberships. Work is O(members + candidates*L), with caller-owned O(L)
// scratch; L is an explicit hard bound and no all-pairs community scan occurs.
[[nodiscard]] inline overlap_expansion_result_v1
build_bounded_overlap_expansion_v1(
    bounded_overlap_membership_view_v1 baseline,
    const overlap_expansion_candidate_v1 *candidates,
    std::uint64_t candidate_count,
    std::uint32_t maximum_memberships_per_member,
    evidence::evidence_identity_v1 expanded_evidence_identity,
    std::uint64_t observation_generation,
    overlap_expansion_buffers_v1 buffers) noexcept {
    const auto baseline_validation =
        validate_bounded_overlap_membership_v1(baseline);
    if (!baseline_validation.valid()) {
        return {overlap_expansion_code_v1::invalid_baseline,
                {},
                baseline_validation.member_index,
                0,
                0,
                static_cast<std::uint32_t>(baseline_validation.code)};
    }
    if (baseline.maximum_memberships_per_member != 1
        || baseline.membership_count != baseline.member_count) {
        return {overlap_expansion_code_v1::baseline_not_disjoint};
    }
    if (!evidence::valid_evidence_identity_v1(expanded_evidence_identity)) {
        return {overlap_expansion_code_v1::invalid_evidence_identity};
    }
    if (observation_generation == 0) {
        return {overlap_expansion_code_v1::missing_generation};
    }
    if (maximum_memberships_per_member == 0) {
        return {overlap_expansion_code_v1::invalid_overlap_bound};
    }
    if (candidate_count != 0 && candidates == nullptr) {
        return {overlap_expansion_code_v1::missing_candidates};
    }
    for (std::uint64_t index = 0; index < candidate_count; ++index) {
        const auto &candidate = candidates[index];
        if (candidate.global_member_id == 0) {
            return {overlap_expansion_code_v1::invalid_candidate_member,
                    {},
                    index};
        }
        if (candidate.local_community_index >= baseline.community_count) {
            return {overlap_expansion_code_v1::invalid_candidate_community,
                    {},
                    index};
        }
        if (candidate.score_numerator == 0
            || candidate.score_denominator == 0
            || candidate.score_numerator > candidate.score_denominator) {
            return {overlap_expansion_code_v1::invalid_candidate_score,
                    {},
                    index};
        }
        if (candidate.reserved != 0) {
            return {overlap_expansion_code_v1::nonzero_reserved, {}, index};
        }
        if (index != 0
            && (candidates[index - 1].global_member_id
                    > candidate.global_member_id
                || (candidates[index - 1].global_member_id
                            == candidate.global_member_id
                    && candidates[index - 1].local_community_index
                           >= candidate.local_community_index))) {
            return {overlap_expansion_code_v1::
                        unordered_or_duplicate_candidate,
                    {},
                    index};
        }
    }
    if (baseline.member_count
        > std::numeric_limits<std::uint64_t>::max()
              / maximum_memberships_per_member) {
        return {overlap_expansion_code_v1::capacity_overflow};
    }
    const auto maximum_output =
        baseline.member_count * maximum_memberships_per_member;
    if (buffers.member_offsets == nullptr || buffers.memberships == nullptr) {
        return {overlap_expansion_code_v1::missing_output,
                {},
                0,
                baseline.member_count + 1,
                maximum_output};
    }
    if (buffers.offset_capacity < baseline.member_count + 1
        || buffers.membership_capacity < maximum_output) {
        return {overlap_expansion_code_v1::insufficient_output,
                {},
                0,
                baseline.member_count + 1,
                maximum_output};
    }
    if (buffers.top_l_scratch == nullptr) {
        return {overlap_expansion_code_v1::missing_scratch};
    }
    if (buffers.scratch_capacity < maximum_memberships_per_member) {
        return {overlap_expansion_code_v1::insufficient_scratch};
    }

    std::uint64_t candidate_index = 0;
    std::uint64_t output = 0;
    for (std::uint64_t member_index = 0;
         member_index < baseline.member_count;
         ++member_index) {
        const auto global_member = baseline.global_member_ids[member_index];
        if (candidate_index < candidate_count
            && candidates[candidate_index].global_member_id < global_member) {
            return {overlap_expansion_code_v1::
                        candidate_member_not_in_baseline,
                    {},
                    candidate_index};
        }
        buffers.member_offsets[member_index] = output;
        const auto baseline_membership =
            baseline.memberships[baseline.member_offsets[member_index]];
        buffers.top_l_scratch[0] = baseline_membership;
        std::uint64_t selected = 1;
        while (candidate_index < candidate_count
               && candidates[candidate_index].global_member_id
                      == global_member) {
            const auto &candidate = candidates[candidate_index++];
            if (candidate.local_community_index
                == baseline_membership.local_community_index) {
                continue;
            }
            const overlap_membership_v1 membership{
                candidate.local_community_index,
                0,
                candidate.score_numerator,
                candidate.score_denominator};
            std::uint64_t position = 1;
            while (position < selected
                   && overlap_membership_ranked_before_v1(
                       buffers.top_l_scratch[position], membership)) {
                ++position;
            }
            if (position < maximum_memberships_per_member) {
                const auto new_selected =
                    selected < maximum_memberships_per_member
                    ? selected + 1
                    : selected;
                for (std::uint64_t move = new_selected; move-- > position + 1;) {
                    buffers.top_l_scratch[move] =
                        buffers.top_l_scratch[move - 1];
                }
                buffers.top_l_scratch[position] = membership;
                selected = new_selected;
            }
        }
        for (std::uint64_t index = 1; index < selected; ++index) {
            const auto value = buffers.top_l_scratch[index];
            auto position = index;
            while (position != 0
                   && buffers.top_l_scratch[position - 1]
                          .local_community_index
                       > value.local_community_index) {
                buffers.top_l_scratch[position] =
                    buffers.top_l_scratch[position - 1];
                --position;
            }
            buffers.top_l_scratch[position] = value;
        }
        for (std::uint64_t index = 0; index < selected; ++index) {
            buffers.memberships[output++] = buffers.top_l_scratch[index];
        }
    }
    if (candidate_index != candidate_count) {
        return {overlap_expansion_code_v1::candidate_member_not_in_baseline,
                {},
                candidate_index};
    }
    buffers.member_offsets[baseline.member_count] = output;
    bounded_overlap_membership_view_v1 view{
        baseline.global_member_ids,
        buffers.member_offsets,
        buffers.memberships,
        baseline.community_identities,
        baseline.member_count,
        output,
        baseline.community_count,
        maximum_memberships_per_member,
        0,
        expanded_evidence_identity,
        baseline.source_identity,
        observation_generation};
    const auto validation = validate_bounded_overlap_membership_v1(view);
    if (!validation.valid()) {
        return {overlap_expansion_code_v1::invalid_built_view,
                {},
                validation.member_index,
                baseline.member_count + 1,
                maximum_output,
                static_cast<std::uint32_t>(validation.code)};
    }
    return {overlap_expansion_code_v1::built,
            view,
            output,
            baseline.member_count + 1,
            maximum_output,
            0};
}

} // namespace cellshard::compiler::discovery::overlap
