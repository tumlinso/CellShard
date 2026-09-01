#pragma once

#include <CellShard/compiler/discovery/bicluster/alternating_expansion_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::bicluster {

struct bicluster_complete_cost_v1 {
    std::uint64_t baseline_cost_per_interaction = 0;
    std::uint64_t candidate_cost_per_interaction = 0;
    std::uint64_t residual_cost_per_interaction = 0;
    std::uint64_t discovery_cost = 0;
    std::uint64_t exact_census_cost = 0;
    std::uint64_t projection_cost = 0;
    std::uint64_t acquisition_transfer_cost = 0;
    std::uint64_t output_transform_cost = 0;
    std::uint64_t synchronization_cost = 0;
    std::uint64_t expected_reuse = 0;
};

enum class bicluster_promotion_v1 : std::uint32_t {
    promote_proposal = 1,
    no_promotion = 2,
};

struct bicluster_marginal_utility_v1 {
    std::uint64_t exact_interaction_count = 0;
    std::uint64_t residual_interaction_count = 0;
    std::uint64_t gross_savings = 0;
    std::uint64_t complete_overhead = 0;
    std::uint64_t net_utility = 0;
    bicluster_promotion_v1 disposition = bicluster_promotion_v1::no_promotion;
    std::uint32_t reserved = 0;
};

enum class bicluster_marginal_cost_code_v1 : std::uint32_t {
    scored = 0,
    empty_exact_coverage,
    missing_reuse,
    candidate_not_faster,
    arithmetic_overflow,
    null_destination,
};

struct bicluster_marginal_cost_result_v1 {
    bicluster_marginal_cost_code_v1 code = bicluster_marginal_cost_code_v1::scored;
    [[nodiscard]] constexpr bool scored() const noexcept {
        return code == bicluster_marginal_cost_code_v1::scored
            || code == bicluster_marginal_cost_code_v1::candidate_not_faster;
    }
};

[[nodiscard]] constexpr bool bicluster_checked_add_v1(
    std::uint64_t lhs, std::uint64_t rhs, std::uint64_t *output) noexcept {
    if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs) return false;
    *output = lhs + rhs;
    return true;
}

[[nodiscard]] constexpr bool bicluster_checked_multiply_v1(
    std::uint64_t lhs, std::uint64_t rhs, std::uint64_t *output) noexcept {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        return false;
    *output = lhs * rhs;
    return true;
}

[[nodiscard]] inline bicluster_marginal_cost_result_v1
score_bicluster_marginal_cost_v1(
    std::uint64_t exact_interaction_count,
    std::uint64_t residual_interaction_count,
    bicluster_complete_cost_v1 cost,
    bicluster_marginal_utility_v1 *destination) noexcept {
    if (destination == nullptr)
        return {bicluster_marginal_cost_code_v1::null_destination};
    *destination = {};
    if (exact_interaction_count == 0)
        return {bicluster_marginal_cost_code_v1::empty_exact_coverage};
    if (cost.expected_reuse == 0)
        return {bicluster_marginal_cost_code_v1::missing_reuse};
    destination->exact_interaction_count = exact_interaction_count;
    destination->residual_interaction_count = residual_interaction_count;
    if (cost.baseline_cost_per_interaction
        <= cost.candidate_cost_per_interaction) {
        return {bicluster_marginal_cost_code_v1::candidate_not_faster};
    }
    std::uint64_t savings_per_use = 0;
    std::uint64_t gross = 0;
    std::uint64_t residual_per_use = 0;
    std::uint64_t residual_total = 0;
    std::uint64_t overhead = 0;
    if (!bicluster_checked_multiply_v1(
            exact_interaction_count,
            cost.baseline_cost_per_interaction
                - cost.candidate_cost_per_interaction,
            &savings_per_use)
        || !bicluster_checked_multiply_v1(
            savings_per_use, cost.expected_reuse, &gross)
        || !bicluster_checked_multiply_v1(
            residual_interaction_count, cost.residual_cost_per_interaction,
            &residual_per_use)
        || !bicluster_checked_multiply_v1(
            residual_per_use, cost.expected_reuse, &residual_total)) {
        *destination = {};
        return {bicluster_marginal_cost_code_v1::arithmetic_overflow};
    }
    const std::uint64_t fixed_costs[] = {
        cost.discovery_cost, cost.exact_census_cost, cost.projection_cost,
        cost.acquisition_transfer_cost, cost.output_transform_cost,
        cost.synchronization_cost, residual_total};
    for (const auto fixed : fixed_costs) {
        if (!bicluster_checked_add_v1(overhead, fixed, &overhead)) {
            *destination = {};
            return {bicluster_marginal_cost_code_v1::arithmetic_overflow};
        }
    }
    destination->gross_savings = gross;
    destination->complete_overhead = overhead;
    if (gross > overhead) {
        destination->net_utility = gross - overhead;
        destination->disposition = bicluster_promotion_v1::promote_proposal;
    }
    return {};
}

[[nodiscard]] constexpr bool authorizes_execution(
    const bicluster_marginal_utility_v1 &) noexcept { return false; }

static_assert(std::is_standard_layout<bicluster_complete_cost_v1>::value);
static_assert(std::is_trivially_copyable<bicluster_complete_cost_v1>::value);

} // namespace cellshard::compiler::discovery::bicluster
