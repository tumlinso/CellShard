#pragma once

#include <CellShard/compiler/partial/partial_atom_v1.hh>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::partial {

inline constexpr std::uint32_t moments_state_schema_version_v1 = 1;

struct moments_state_v1 {
    std::uint64_t contribution_count = 0;
    double mean = 0.0;
    double centered_sum_squares = 0.0;
    atom_persistent_identity_v1 algebra_identity{};
    atom_persistent_identity_v1 numerical_policy_identity{};
    atom_persistent_identity_v1 merge_order_identity{};
    std::uint64_t value_generation = 0;
    std::uint32_t schema_version = moments_state_schema_version_v1;
    std::uint32_t reserved = 0;
};

enum class moments_state_code_v1 : std::uint32_t {
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

struct moments_state_result_v1 {
    moments_state_code_v1 code = moments_state_code_v1::valid;
    moments_state_v1 state{};
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == moments_state_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<moments_state_v1>::value);
static_assert(std::is_trivially_copyable<moments_state_v1>::value);

[[nodiscard]] inline moments_state_code_v1 validate_moments_state_v1(
    const moments_state_v1 &state) noexcept {
    if (state.schema_version != moments_state_schema_version_v1) {
        return moments_state_code_v1::unsupported_schema;
    }
    if (!atom::validate_atom_persistent_identity_v1(state.algebra_identity)
             .valid()
        || !atom::validate_atom_persistent_identity_v1(
                state.numerical_policy_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(
                state.merge_order_identity)
                .valid()) {
        return moments_state_code_v1::invalid_identity;
    }
    if (state.value_generation == 0) return moments_state_code_v1::missing_generation;
    if (state.contribution_count == 0) return moments_state_code_v1::empty_state;
    if (!std::isfinite(state.mean)
        || !std::isfinite(state.centered_sum_squares)
        || state.centered_sum_squares < 0.0) {
        return moments_state_code_v1::invalid_numeric_state;
    }
    return state.reserved == 0 ? moments_state_code_v1::valid
                               : moments_state_code_v1::nonzero_reserved;
}

[[nodiscard]] inline moments_state_result_v1 make_moments_state_v1(
    const double *values, std::uint64_t count,
    atom_persistent_identity_v1 algebra_identity,
    atom_persistent_identity_v1 numerical_policy_identity,
    atom_persistent_identity_v1 merge_order_identity,
    std::uint64_t value_generation) noexcept {
    if (values == nullptr || count == 0) {
        return {moments_state_code_v1::empty_state, {}};
    }
    moments_state_v1 state{0, 0.0, 0.0, algebra_identity,
                           numerical_policy_identity, merge_order_identity,
                           value_generation, moments_state_schema_version_v1,
                           0};
    for (std::uint64_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) {
            return {moments_state_code_v1::invalid_numeric_state, {}};
        }
        ++state.contribution_count;
        const double delta = values[index] - state.mean;
        state.mean += delta / static_cast<double>(state.contribution_count);
        const double delta2 = values[index] - state.mean;
        state.centered_sum_squares += delta * delta2;
    }
    const auto code = validate_moments_state_v1(state);
    return {code, code == moments_state_code_v1::valid ? state
                                                       : moments_state_v1{}};
}

[[nodiscard]] inline moments_state_result_v1 merge_moments_states_v1(
    const moments_state_v1 &left, const moments_state_v1 &right) noexcept {
    const auto left_code = validate_moments_state_v1(left);
    if (left_code != moments_state_code_v1::valid) return {left_code, {}};
    const auto right_code = validate_moments_state_v1(right);
    if (right_code != moments_state_code_v1::valid) return {right_code, {}};
    if (left.algebra_identity != right.algebra_identity
        || left.numerical_policy_identity != right.numerical_policy_identity
        || left.merge_order_identity != right.merge_order_identity
        || left.value_generation != right.value_generation) {
        return {moments_state_code_v1::incompatible_contract, {}};
    }
    if (left.contribution_count
        > std::numeric_limits<std::uint64_t>::max()
              - right.contribution_count) {
        return {moments_state_code_v1::count_overflow, {}};
    }
    auto merged = left;
    const std::uint64_t count = left.contribution_count + right.contribution_count;
    const double delta = right.mean - left.mean;
    merged.mean = left.mean
        + delta * static_cast<double>(right.contribution_count)
            / static_cast<double>(count);
    merged.centered_sum_squares = left.centered_sum_squares
        + right.centered_sum_squares
        + delta * delta * static_cast<double>(left.contribution_count)
            * static_cast<double>(right.contribution_count)
            / static_cast<double>(count);
    merged.contribution_count = count;
    const auto code = validate_moments_state_v1(merged);
    return {code, code == moments_state_code_v1::valid ? merged
                                                       : moments_state_v1{}};
}

[[nodiscard]] constexpr double population_variance_v1(
    const moments_state_v1 &state) noexcept {
    return state.centered_sum_squares
        / static_cast<double>(state.contribution_count);
}

} // namespace cellshard::compiler::partial
