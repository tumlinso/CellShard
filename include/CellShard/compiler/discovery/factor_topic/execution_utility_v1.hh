#pragma once

#include <CellShard/compiler/discovery/factor_topic/exact_coverage_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::factor_topic {

struct factor_execution_cost_v1 {
    evidence::evidence_identity_v1 candidate_identity{};
    std::uint64_t baseline_cost_per_use = 0;
    std::uint64_t candidate_cost_per_use = 0;
    std::uint64_t residual_cost_per_member_per_use = 0;
    std::uint64_t preparation_cost = 0;
    std::uint64_t acquisition_transfer_cost = 0;
    std::uint64_t reconstruction_cost = 0;
    std::uint64_t expected_reuse = 0;
};

enum class factor_utility_disposition_v1 : std::uint32_t {
    promote_proposal = 1,
    no_promotion = 2,
};

struct factor_execution_utility_v1 {
    evidence::evidence_identity_v1 candidate_identity{};
    std::uint64_t exact_owner_count = 0;
    std::uint64_t residual_count = 0;
    std::uint64_t gross_savings = 0;
    std::uint64_t total_overhead = 0;
    std::uint64_t net_utility = 0;
    factor_utility_disposition_v1 disposition =
        factor_utility_disposition_v1::no_promotion;
    std::uint32_t reserved = 0;
};

enum class factor_execution_utility_code_v1 : std::uint32_t {
    scored = 0,
    invalid_identity,
    identity_mismatch,
    empty_exact_coverage,
    missing_reuse,
    candidate_not_faster,
    arithmetic_overflow,
    null_destination,
};

struct factor_execution_utility_result_v1 {
    factor_execution_utility_code_v1 code =
        factor_execution_utility_code_v1::scored;

    [[nodiscard]] constexpr bool scored() const noexcept {
        return code == factor_execution_utility_code_v1::scored
            || code == factor_execution_utility_code_v1::candidate_not_faster;
    }
};

[[nodiscard]] constexpr bool checked_add_u64_v1(
    std::uint64_t lhs, std::uint64_t rhs, std::uint64_t *result) noexcept {
    if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

[[nodiscard]] constexpr bool checked_multiply_u64_v1(
    std::uint64_t lhs, std::uint64_t rhs, std::uint64_t *result) noexcept {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

// Costs are caller-measured units from a comparable complete-cost experiment.
// The scorer does no timing and cannot turn an unevaluated regularity into a
// promotion. Residual execution, preparation, acquisition, transfer, and
// reconstruction are all charged explicitly.
[[nodiscard]] inline factor_execution_utility_result_v1
score_factor_execution_utility_v1(
    const factor_exact_coverage_span_v1 &coverage,
    std::uint64_t residual_count,
    factor_execution_cost_v1 cost,
    factor_execution_utility_v1 *destination) noexcept {
    if (destination == nullptr) {
        return {factor_execution_utility_code_v1::null_destination};
    }
    *destination = {};
    if (!evidence::valid_evidence_identity_v1(cost.candidate_identity)) {
        return {factor_execution_utility_code_v1::invalid_identity};
    }
    if (!(coverage.candidate_identity == cost.candidate_identity)) {
        return {factor_execution_utility_code_v1::identity_mismatch};
    }
    if (coverage.owner_count == 0) {
        return {factor_execution_utility_code_v1::empty_exact_coverage};
    }
    if (cost.expected_reuse == 0) {
        return {factor_execution_utility_code_v1::missing_reuse};
    }

    destination->candidate_identity = cost.candidate_identity;
    destination->exact_owner_count = coverage.owner_count;
    destination->residual_count = residual_count;
    if (cost.baseline_cost_per_use <= cost.candidate_cost_per_use) {
        destination->disposition = factor_utility_disposition_v1::no_promotion;
        return {factor_execution_utility_code_v1::candidate_not_faster};
    }

    const auto per_use_savings =
        cost.baseline_cost_per_use - cost.candidate_cost_per_use;
    std::uint64_t gross_savings = 0;
    std::uint64_t residual_per_use = 0;
    std::uint64_t residual_total = 0;
    std::uint64_t overhead = 0;
    if (!checked_multiply_u64_v1(
            per_use_savings, cost.expected_reuse, &gross_savings)
        || !checked_multiply_u64_v1(
            residual_count, cost.residual_cost_per_member_per_use,
            &residual_per_use)
        || !checked_multiply_u64_v1(
            residual_per_use, cost.expected_reuse, &residual_total)
        || !checked_add_u64_v1(
            cost.preparation_cost, cost.acquisition_transfer_cost, &overhead)
        || !checked_add_u64_v1(overhead, cost.reconstruction_cost, &overhead)
        || !checked_add_u64_v1(overhead, residual_total, &overhead)) {
        *destination = {};
        return {factor_execution_utility_code_v1::arithmetic_overflow};
    }
    destination->gross_savings = gross_savings;
    destination->total_overhead = overhead;
    if (gross_savings > overhead) {
        destination->net_utility = gross_savings - overhead;
        destination->disposition =
            factor_utility_disposition_v1::promote_proposal;
    } else {
        destination->net_utility = 0;
        destination->disposition = factor_utility_disposition_v1::no_promotion;
    }
    return {};
}

[[nodiscard]] constexpr bool authorizes_execution(
    const factor_execution_utility_v1 &) noexcept {
    return false;
}

static_assert(std::is_standard_layout<factor_execution_cost_v1>::value);
static_assert(std::is_trivially_copyable<factor_execution_cost_v1>::value);
static_assert(std::is_standard_layout<factor_execution_utility_v1>::value);
static_assert(std::is_trivially_copyable<factor_execution_utility_v1>::value);

} // namespace cellshard::compiler::discovery::factor_topic
