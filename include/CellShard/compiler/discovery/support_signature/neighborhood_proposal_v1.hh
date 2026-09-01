#pragma once

#include <CellShard/compiler/discovery/support_signature/exact_rescan_v1.hh>

#include <cstdint>

namespace cellshard::compiler::discovery::support_signature {

struct support_neighborhood_policy_v1 {
    std::uint64_t minimum_shared_support = 0;
    std::uint64_t minimum_jaccard_numerator = 0;
    std::uint64_t minimum_jaccard_denominator = 0;
    std::uint64_t minimum_bidirectional_containment_numerator = 0;
    std::uint64_t minimum_bidirectional_containment_denominator = 0;
};

struct destination_support_neighborhood_proposal_v1 {
    atom::atom_persistent_identity_v1 proposal_identity{};
    std::uint64_t first_global_destination_id = 0;
    std::uint64_t second_global_destination_id = 0;
    exact_support_pair_score_v1 exact_score{};
};

struct destination_support_neighborhood_view_v1 {
    const destination_support_neighborhood_proposal_v1 *proposals = nullptr;
    std::uint64_t proposal_count = 0;
    atom::atom_persistent_identity_v1 relation_identity{};
    std::uint64_t relation_generation = 0;
};

enum class support_neighborhood_code_v1 : std::uint32_t {
    built = 0,
    invalid_support,
    invalid_scores,
    context_mismatch,
    invalid_policy,
    missing_proposal_identities,
    insufficient_proposal_identities,
    invalid_proposal_identity,
    unordered_or_duplicate_proposal_identity,
    missing_output,
    insufficient_output,
};

struct support_neighborhood_result_v1 {
    support_neighborhood_code_v1 code = support_neighborhood_code_v1::built;
    destination_support_neighborhood_view_v1 view{};
    std::uint64_t index = 0;
    std::uint64_t required_proposals = 0;
    [[nodiscard]] constexpr bool built() const noexcept {
        return code == support_neighborhood_code_v1::built;
    }
};

// Overflow-free exact comparison lhs_num/lhs_den >= rhs_num/rhs_den using
// continued-fraction quotients. All denominators must be nonzero.
[[nodiscard]] constexpr bool ratio_at_least_v1(
    std::uint64_t lhs_num,
    std::uint64_t lhs_den,
    std::uint64_t rhs_num,
    std::uint64_t rhs_den) noexcept {
    bool reversed = false;
    while (true) {
        const auto lhs_q = lhs_num / lhs_den;
        const auto rhs_q = rhs_num / rhs_den;
        if (lhs_q != rhs_q) {
            return reversed ? lhs_q < rhs_q : lhs_q > rhs_q;
        }
        const auto lhs_r = lhs_num % lhs_den;
        const auto rhs_r = rhs_num % rhs_den;
        if (lhs_r == 0 || rhs_r == 0) {
            if (lhs_r == 0 && rhs_r == 0) return true;
            const bool lhs_greater = lhs_r != 0;
            return reversed ? !lhs_greater : lhs_greater;
        }
        lhs_num = lhs_den;
        lhs_den = lhs_r;
        rhs_num = rhs_den;
        rhs_den = rhs_r;
        reversed = !reversed;
    }
}

[[nodiscard]] constexpr bool score_passes_neighborhood_policy_v1(
    const exact_support_pair_score_v1 &score,
    const support_neighborhood_policy_v1 &policy) noexcept {
    return score.shared_support_count >= policy.minimum_shared_support
        && ratio_at_least_v1(
            score.shared_support_count, score.union_support_count,
            policy.minimum_jaccard_numerator,
            policy.minimum_jaccard_denominator)
        && ratio_at_least_v1(
            score.shared_support_count, score.first_support_count,
            policy.minimum_bidirectional_containment_numerator,
            policy.minimum_bidirectional_containment_denominator)
        && ratio_at_least_v1(
            score.shared_support_count, score.second_support_count,
            policy.minimum_bidirectional_containment_numerator,
            policy.minimum_bidirectional_containment_denominator);
}

[[nodiscard]] constexpr support_neighborhood_result_v1
build_support_neighborhood_proposals_v1(
    exact_destination_support_view_v1 support,
    exact_support_pair_score_view_v1 scores,
    support_neighborhood_policy_v1 policy,
    const atom::atom_persistent_identity_v1 *proposal_identities,
    std::uint64_t proposal_identity_count,
    destination_support_neighborhood_proposal_v1 *output,
    std::uint64_t output_capacity) noexcept {
    if (!validate_exact_destination_support_view_v1(support)) {
        return {support_neighborhood_code_v1::invalid_support};
    }
    if (scores.scores == nullptr || scores.score_count == 0
        || !atom::validate_atom_persistent_identity_v1(
                scores.relation_identity).valid()
        || scores.relation_generation == 0) {
        return {support_neighborhood_code_v1::invalid_scores};
    }
    if (scores.relation_identity != support.relation_identity
        || scores.relation_generation != support.relation_generation) {
        return {support_neighborhood_code_v1::context_mismatch};
    }
    if (policy.minimum_shared_support == 0
        || policy.minimum_jaccard_denominator == 0
        || policy.minimum_jaccard_numerator
               > policy.minimum_jaccard_denominator
        || policy.minimum_bidirectional_containment_denominator == 0
        || policy.minimum_bidirectional_containment_numerator
               > policy.minimum_bidirectional_containment_denominator) {
        return {support_neighborhood_code_v1::invalid_policy};
    }
    std::uint64_t required = 0;
    for (std::uint64_t index = 0; index < scores.score_count; ++index) {
        const auto &score = scores.scores[index];
        if (score.first_destination_index >= score.second_destination_index
            || score.second_destination_index >= support.destination_count
            || score.union_support_count == 0
            || score.first_support_count == 0
            || score.second_support_count == 0
            || score.shared_support_count > score.first_support_count
            || score.shared_support_count > score.second_support_count) {
            return {support_neighborhood_code_v1::invalid_scores, {}, index};
        }
        if (score_passes_neighborhood_policy_v1(score, policy)) ++required;
    }
    if (proposal_identities == nullptr) {
        return {support_neighborhood_code_v1::missing_proposal_identities, {},
                0, required};
    }
    if (proposal_identity_count < required) {
        return {support_neighborhood_code_v1::
                    insufficient_proposal_identities,
                {}, proposal_identity_count, required};
    }
    if (output == nullptr) {
        return {support_neighborhood_code_v1::missing_output, {}, 0,
                required};
    }
    if (output_capacity < required) {
        return {support_neighborhood_code_v1::insufficient_output, {}, 0,
                required};
    }
    for (std::uint64_t index = 0; index < required; ++index) {
        if (!atom::validate_atom_persistent_identity_v1(
                 proposal_identities[index]).valid()) {
            return {support_neighborhood_code_v1::invalid_proposal_identity,
                    {}, index, required};
        }
        if (index != 0
            && !atom::atom_persistent_identity_less_v1(
                proposal_identities[index - 1], proposal_identities[index])) {
            return {support_neighborhood_code_v1::
                        unordered_or_duplicate_proposal_identity,
                    {}, index, required};
        }
    }
    std::uint64_t cursor = 0;
    for (std::uint64_t index = 0; index < scores.score_count; ++index) {
        const auto &score = scores.scores[index];
        if (!score_passes_neighborhood_policy_v1(score, policy)) continue;
        output[cursor] = {
            proposal_identities[cursor],
            support.global_destination_ids[score.first_destination_index],
            support.global_destination_ids[score.second_destination_index],
            score};
        ++cursor;
    }
    return {support_neighborhood_code_v1::built,
            {output, cursor, support.relation_identity,
             support.relation_generation},
            scores.score_count, required};
}

[[nodiscard]] constexpr bool authorizes_execution(
    destination_support_neighborhood_view_v1) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::support_signature
