#pragma once

#include <CellShard/compiler/partial/partial_atom_v1.hh>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::partial {

inline constexpr std::uint32_t additive_state_schema_version_v1 = 1;

// FP64 Neumaier sufficient state. Algebra, numerical policy, and merge order
// are persistent identities because changing any of them changes the result.
struct additive_state_v1 {
    double sum = 0.0;
    double correction = 0.0;
    std::uint64_t contribution_count = 0;
    atom_persistent_identity_v1 algebra_identity{};
    atom_persistent_identity_v1 numerical_policy_identity{};
    atom_persistent_identity_v1 merge_order_identity{};
    std::uint64_t value_generation = 0;
    std::uint32_t schema_version = additive_state_schema_version_v1;
    std::uint32_t reserved = 0;
};

enum class additive_state_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_identity,
    missing_generation,
    empty_state,
    nonfinite_state,
    nonzero_reserved,
    incompatible_contract,
    count_overflow,
};

struct additive_state_result_v1 {
    additive_state_code_v1 code = additive_state_code_v1::valid;
    additive_state_v1 state{};
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == additive_state_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<additive_state_v1>::value);
static_assert(std::is_trivially_copyable<additive_state_v1>::value);

[[nodiscard]] inline additive_state_code_v1 validate_additive_state_v1(
    const additive_state_v1 &state) noexcept {
    if (state.schema_version != additive_state_schema_version_v1) {
        return additive_state_code_v1::unsupported_schema;
    }
    if (!atom::validate_atom_persistent_identity_v1(state.algebra_identity)
             .valid()
        || !atom::validate_atom_persistent_identity_v1(
                state.numerical_policy_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(
                state.merge_order_identity)
                .valid()) {
        return additive_state_code_v1::invalid_identity;
    }
    if (state.value_generation == 0) {
        return additive_state_code_v1::missing_generation;
    }
    if (state.contribution_count == 0) {
        return additive_state_code_v1::empty_state;
    }
    if (!std::isfinite(state.sum) || !std::isfinite(state.correction)) {
        return additive_state_code_v1::nonfinite_state;
    }
    return state.reserved == 0 ? additive_state_code_v1::valid
                               : additive_state_code_v1::nonzero_reserved;
}

inline void neumaier_add_v1(double value, double *sum,
                            double *correction) noexcept {
    const double updated = *sum + value;
    if (std::fabs(*sum) >= std::fabs(value)) {
        *correction += (*sum - updated) + value;
    } else {
        *correction += (value - updated) + *sum;
    }
    *sum = updated;
}

[[nodiscard]] inline additive_state_result_v1 make_additive_state_v1(
    const double *values, std::uint64_t count,
    atom_persistent_identity_v1 algebra_identity,
    atom_persistent_identity_v1 numerical_policy_identity,
    atom_persistent_identity_v1 merge_order_identity,
    std::uint64_t value_generation) noexcept {
    additive_state_v1 state{0.0, 0.0, count, algebra_identity,
                            numerical_policy_identity, merge_order_identity,
                            value_generation, additive_state_schema_version_v1,
                            0};
    if (values == nullptr || count == 0) {
        return {additive_state_code_v1::empty_state, {}};
    }
    for (std::uint64_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) {
            return {additive_state_code_v1::nonfinite_state, {}};
        }
        neumaier_add_v1(values[index], &state.sum, &state.correction);
    }
    const auto code = validate_additive_state_v1(state);
    return {code, code == additive_state_code_v1::valid ? state
                                                        : additive_state_v1{}};
}

[[nodiscard]] inline additive_state_result_v1 merge_additive_states_v1(
    const additive_state_v1 &left, const additive_state_v1 &right) noexcept {
    const auto left_code = validate_additive_state_v1(left);
    if (left_code != additive_state_code_v1::valid) {
        return {left_code, {}};
    }
    const auto right_code = validate_additive_state_v1(right);
    if (right_code != additive_state_code_v1::valid) {
        return {right_code, {}};
    }
    if (left.algebra_identity != right.algebra_identity
        || left.numerical_policy_identity != right.numerical_policy_identity
        || left.merge_order_identity != right.merge_order_identity
        || left.value_generation != right.value_generation) {
        return {additive_state_code_v1::incompatible_contract, {}};
    }
    if (left.contribution_count
        > std::numeric_limits<std::uint64_t>::max()
              - right.contribution_count) {
        return {additive_state_code_v1::count_overflow, {}};
    }
    auto merged = left;
    neumaier_add_v1(right.sum, &merged.sum, &merged.correction);
    neumaier_add_v1(right.correction, &merged.sum, &merged.correction);
    merged.contribution_count += right.contribution_count;
    if (!std::isfinite(merged.sum) || !std::isfinite(merged.correction)) {
        return {additive_state_code_v1::nonfinite_state, {}};
    }
    return {additive_state_code_v1::valid, merged};
}

[[nodiscard]] constexpr double finalize_additive_state_v1(
    const additive_state_v1 &state) noexcept {
    return state.sum + state.correction;
}

} // namespace cellshard::compiler::partial
