#pragma once

#include <CellShard/compiler/partial/partial_atom_v1.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::partial {

inline constexpr std::uint32_t log_sum_exp_state_schema_version_v1 = 1;

struct log_sum_exp_state_v1 {
    double maximum = -std::numeric_limits<double>::infinity();
    double scaled_exponential_sum = 0.0;
    std::uint64_t contribution_count = 0;
    atom_persistent_identity_v1 algebra_identity{};
    atom_persistent_identity_v1 numerical_policy_identity{};
    atom_persistent_identity_v1 merge_order_identity{};
    std::uint64_t value_generation = 0;
    std::uint32_t schema_version = log_sum_exp_state_schema_version_v1;
    std::uint32_t reserved = 0;
};

enum class log_sum_exp_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_identity,
    missing_generation,
    empty_state,
    invalid_numeric_state,
    nonzero_reserved,
    incompatible_contract,
    count_overflow,
};

struct log_sum_exp_result_v1 {
    log_sum_exp_code_v1 code = log_sum_exp_code_v1::valid;
    log_sum_exp_state_v1 state{};
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == log_sum_exp_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<log_sum_exp_state_v1>::value);
static_assert(std::is_trivially_copyable<log_sum_exp_state_v1>::value);

[[nodiscard]] inline log_sum_exp_code_v1 validate_log_sum_exp_state_v1(
    const log_sum_exp_state_v1 &state) noexcept {
    if (state.schema_version != log_sum_exp_state_schema_version_v1) {
        return log_sum_exp_code_v1::unsupported_schema;
    }
    if (!atom::validate_atom_persistent_identity_v1(state.algebra_identity)
             .valid()
        || !atom::validate_atom_persistent_identity_v1(
                state.numerical_policy_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(
                state.merge_order_identity)
                .valid()) {
        return log_sum_exp_code_v1::invalid_identity;
    }
    if (state.value_generation == 0) return log_sum_exp_code_v1::missing_generation;
    if (state.contribution_count == 0) return log_sum_exp_code_v1::empty_state;
    const bool all_negative_infinity =
        state.maximum == -std::numeric_limits<double>::infinity()
        && state.scaled_exponential_sum == 0.0;
    if ((!std::isfinite(state.maximum) && !all_negative_infinity)
        || !std::isfinite(state.scaled_exponential_sum)
        || state.scaled_exponential_sum < 0.0
        || (!all_negative_infinity && state.scaled_exponential_sum < 1.0)) {
        return log_sum_exp_code_v1::invalid_numeric_state;
    }
    return state.reserved == 0 ? log_sum_exp_code_v1::valid
                               : log_sum_exp_code_v1::nonzero_reserved;
}

[[nodiscard]] inline log_sum_exp_result_v1 make_log_sum_exp_state_v1(
    const double *values, std::uint64_t count,
    atom_persistent_identity_v1 algebra_identity,
    atom_persistent_identity_v1 numerical_policy_identity,
    atom_persistent_identity_v1 merge_order_identity,
    std::uint64_t value_generation) noexcept {
    if (values == nullptr || count == 0) {
        return {log_sum_exp_code_v1::empty_state, {}};
    }
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::uint64_t index = 0; index < count; ++index) {
        if (std::isnan(values[index])
            || values[index] == std::numeric_limits<double>::infinity()) {
            return {log_sum_exp_code_v1::invalid_numeric_state, {}};
        }
        maximum = std::max(maximum, values[index]);
    }
    double scaled_sum = 0.0;
    if (std::isfinite(maximum)) {
        for (std::uint64_t index = 0; index < count; ++index) {
            scaled_sum += std::exp(values[index] - maximum);
        }
    }
    log_sum_exp_state_v1 state{maximum, scaled_sum, count, algebra_identity,
                               numerical_policy_identity, merge_order_identity,
                               value_generation,
                               log_sum_exp_state_schema_version_v1, 0};
    const auto code = validate_log_sum_exp_state_v1(state);
    return {code, code == log_sum_exp_code_v1::valid ? state
                                                     : log_sum_exp_state_v1{}};
}

[[nodiscard]] inline log_sum_exp_result_v1 merge_log_sum_exp_states_v1(
    const log_sum_exp_state_v1 &left,
    const log_sum_exp_state_v1 &right) noexcept {
    const auto left_code = validate_log_sum_exp_state_v1(left);
    if (left_code != log_sum_exp_code_v1::valid) return {left_code, {}};
    const auto right_code = validate_log_sum_exp_state_v1(right);
    if (right_code != log_sum_exp_code_v1::valid) return {right_code, {}};
    if (left.algebra_identity != right.algebra_identity
        || left.numerical_policy_identity != right.numerical_policy_identity
        || left.merge_order_identity != right.merge_order_identity
        || left.value_generation != right.value_generation) {
        return {log_sum_exp_code_v1::incompatible_contract, {}};
    }
    if (left.contribution_count
        > std::numeric_limits<std::uint64_t>::max()
              - right.contribution_count) {
        return {log_sum_exp_code_v1::count_overflow, {}};
    }
    auto merged = left;
    merged.contribution_count += right.contribution_count;
    if (left.maximum == -std::numeric_limits<double>::infinity()) {
        merged.maximum = right.maximum;
        merged.scaled_exponential_sum = right.scaled_exponential_sum;
    } else if (right.maximum != -std::numeric_limits<double>::infinity()) {
        merged.maximum = std::max(left.maximum, right.maximum);
        merged.scaled_exponential_sum =
            left.scaled_exponential_sum * std::exp(left.maximum - merged.maximum)
            + right.scaled_exponential_sum
                * std::exp(right.maximum - merged.maximum);
    }
    return {validate_log_sum_exp_state_v1(merged), merged};
}

[[nodiscard]] inline double finalize_log_sum_exp_state_v1(
    const log_sum_exp_state_v1 &state) noexcept {
    return state.maximum == -std::numeric_limits<double>::infinity()
        ? state.maximum
        : state.maximum + std::log(state.scaled_exponential_sum);
}

} // namespace cellshard::compiler::partial
