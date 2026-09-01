#pragma once

#include <CellShard/compiler/partial/partial_atom_v1.hh>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::partial {

inline constexpr std::uint32_t extrema_state_schema_version_v1 = 1;

struct extrema_witness_v1 {
    double value = 0.0;
    atom_persistent_identity_v1 biological_identity{};
    std::uint64_t canonical_ordinal = 0;
};

struct extrema_state_v1 {
    extrema_witness_v1 minimum{};
    extrema_witness_v1 maximum{};
    std::uint64_t contribution_count = 0;
    atom_persistent_identity_v1 algebra_identity{};
    atom_persistent_identity_v1 numerical_policy_identity{};
    atom_persistent_identity_v1 persistent_order_identity{};
    std::uint64_t value_generation = 0;
    std::uint32_t schema_version = extrema_state_schema_version_v1;
    std::uint32_t reserved = 0;
};

enum class extrema_state_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_identity,
    missing_generation,
    empty_state,
    nonfinite_value,
    inverted_extrema,
    nonzero_reserved,
    incompatible_contract,
    count_overflow,
};

struct extrema_state_result_v1 {
    extrema_state_code_v1 code = extrema_state_code_v1::valid;
    extrema_state_v1 state{};
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == extrema_state_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<extrema_witness_v1>::value);
static_assert(std::is_trivially_copyable<extrema_witness_v1>::value);
static_assert(std::is_standard_layout<extrema_state_v1>::value);
static_assert(std::is_trivially_copyable<extrema_state_v1>::value);

[[nodiscard]] constexpr bool extrema_witness_less_v1(
    const extrema_witness_v1 &lhs, const extrema_witness_v1 &rhs) noexcept {
    return atom::atom_persistent_identity_less_v1(
               lhs.biological_identity, rhs.biological_identity)
        || (lhs.biological_identity == rhs.biological_identity
            && lhs.canonical_ordinal < rhs.canonical_ordinal);
}

[[nodiscard]] inline extrema_state_code_v1 validate_extrema_state_v1(
    const extrema_state_v1 &state) noexcept {
    if (state.schema_version != extrema_state_schema_version_v1) {
        return extrema_state_code_v1::unsupported_schema;
    }
    if (!atom::validate_atom_persistent_identity_v1(state.algebra_identity)
             .valid()
        || !atom::validate_atom_persistent_identity_v1(
                state.numerical_policy_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(
                state.persistent_order_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(
                state.minimum.biological_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(
                state.maximum.biological_identity)
                .valid()) {
        return extrema_state_code_v1::invalid_identity;
    }
    if (state.value_generation == 0) {
        return extrema_state_code_v1::missing_generation;
    }
    if (state.contribution_count == 0) {
        return extrema_state_code_v1::empty_state;
    }
    if (!std::isfinite(state.minimum.value)
        || !std::isfinite(state.maximum.value)) {
        return extrema_state_code_v1::nonfinite_value;
    }
    if (state.minimum.value > state.maximum.value) {
        return extrema_state_code_v1::inverted_extrema;
    }
    return state.reserved == 0 ? extrema_state_code_v1::valid
                               : extrema_state_code_v1::nonzero_reserved;
}

[[nodiscard]] constexpr bool prefer_minimum_v1(
    const extrema_witness_v1 &lhs, const extrema_witness_v1 &rhs) noexcept {
    return lhs.value < rhs.value
        || (lhs.value == rhs.value && extrema_witness_less_v1(lhs, rhs));
}

[[nodiscard]] constexpr bool prefer_maximum_v1(
    const extrema_witness_v1 &lhs, const extrema_witness_v1 &rhs) noexcept {
    return lhs.value > rhs.value
        || (lhs.value == rhs.value && extrema_witness_less_v1(lhs, rhs));
}

[[nodiscard]] inline extrema_state_result_v1 merge_extrema_states_v1(
    const extrema_state_v1 &left, const extrema_state_v1 &right) noexcept {
    const auto left_code = validate_extrema_state_v1(left);
    if (left_code != extrema_state_code_v1::valid) return {left_code, {}};
    const auto right_code = validate_extrema_state_v1(right);
    if (right_code != extrema_state_code_v1::valid) return {right_code, {}};
    if (left.algebra_identity != right.algebra_identity
        || left.numerical_policy_identity != right.numerical_policy_identity
        || left.persistent_order_identity != right.persistent_order_identity
        || left.value_generation != right.value_generation) {
        return {extrema_state_code_v1::incompatible_contract, {}};
    }
    if (left.contribution_count
        > std::numeric_limits<std::uint64_t>::max()
              - right.contribution_count) {
        return {extrema_state_code_v1::count_overflow, {}};
    }
    auto merged = left;
    if (prefer_minimum_v1(right.minimum, merged.minimum)) {
        merged.minimum = right.minimum;
    }
    if (prefer_maximum_v1(right.maximum, merged.maximum)) {
        merged.maximum = right.maximum;
    }
    merged.contribution_count += right.contribution_count;
    return {extrema_state_code_v1::valid, merged};
}

} // namespace cellshard::compiler::partial
