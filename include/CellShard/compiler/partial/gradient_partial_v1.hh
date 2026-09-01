#pragma once

#include <CellShard/compiler/atom/gradient_plane_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::partial {

using atom::atom_gradient_plane_v1;
using atom::atom_gradient_ownership_v1;
using atom::atom_persistent_identity_v1;

inline constexpr std::uint32_t gradient_partial_schema_version_v1 = 1;

enum class gradient_legality_v1 : std::uint32_t {
    exact_declared_vjp = 1,
};

struct gradient_partial_view_v1 {
    const atom_gradient_plane_v1 *gradient = nullptr;
    atom_persistent_identity_v1 forward_partial_identity{};
    atom_persistent_identity_v1 derivative_rule_identity{};
    atom_persistent_identity_v1 dependency_closure_identity{};
    atom_persistent_identity_v1 numerical_policy_identity{};
    std::uint64_t forward_structure_generation = 0;
    std::uint64_t forward_value_generation = 0;
    std::uint64_t forward_state_generation = 0;
    std::uint64_t parameter_generation = 0;
    std::uint64_t adjoint_generation = 0;
    gradient_legality_v1 legality = gradient_legality_v1::exact_declared_vjp;
    std::uint32_t schema_version = gradient_partial_schema_version_v1;
    std::uint32_t reserved = 0;
    std::uint32_t trailing_reserved = 0;
};

enum class gradient_partial_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    missing_gradient,
    invalid_gradient,
    mirror_cannot_own_contribution,
    invalid_identity,
    missing_generation,
    unproven_derivative,
    nonzero_reserved,
};

struct gradient_partial_result_v1 {
    gradient_partial_code_v1 code = gradient_partial_code_v1::valid;
    std::uint32_t nested_code = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == gradient_partial_code_v1::valid;
    }
};

static_assert(offsetof(gradient_partial_view_v1, gradient) == 0);
static_assert(std::is_standard_layout<gradient_partial_view_v1>::value);
static_assert(std::is_trivially_copyable<gradient_partial_view_v1>::value);

[[nodiscard]] inline gradient_partial_result_v1 validate_gradient_partial_v1(
    const gradient_partial_view_v1 &partial,
    std::uint32_t coverage_source_validation,
    std::uint8_t *canonical_marks,
    std::uint64_t mark_capacity) noexcept {
    if (partial.schema_version != gradient_partial_schema_version_v1) {
        return {gradient_partial_code_v1::unsupported_schema, 0};
    }
    if (partial.gradient == nullptr) {
        return {gradient_partial_code_v1::missing_gradient, 0};
    }
    const auto gradient_result = atom::validate_atom_gradient_plane_v1(
        *partial.gradient, coverage_source_validation, canonical_marks,
        mark_capacity);
    if (!gradient_result.valid()) {
        return {gradient_partial_code_v1::invalid_gradient,
                static_cast<std::uint32_t>(gradient_result.code)};
    }
    if (partial.gradient->ownership != atom_gradient_ownership_v1::primary) {
        return {gradient_partial_code_v1::mirror_cannot_own_contribution, 0};
    }
#define CELLSHARD_GRADIENT_PARTIAL_ID(field) \
    if (!atom::validate_atom_persistent_identity_v1(partial.field).valid()) \
        return {gradient_partial_code_v1::invalid_identity, 0}
    CELLSHARD_GRADIENT_PARTIAL_ID(forward_partial_identity);
    CELLSHARD_GRADIENT_PARTIAL_ID(derivative_rule_identity);
    CELLSHARD_GRADIENT_PARTIAL_ID(dependency_closure_identity);
    CELLSHARD_GRADIENT_PARTIAL_ID(numerical_policy_identity);
#undef CELLSHARD_GRADIENT_PARTIAL_ID
    if (partial.forward_structure_generation == 0
        || partial.forward_value_generation == 0
        || partial.forward_state_generation == 0
        || partial.parameter_generation == 0
        || partial.adjoint_generation == 0) {
        return {gradient_partial_code_v1::missing_generation, 0};
    }
    if (partial.legality != gradient_legality_v1::exact_declared_vjp) {
        return {gradient_partial_code_v1::unproven_derivative, 0};
    }
    return partial.reserved == 0 && partial.trailing_reserved == 0
        ? gradient_partial_result_v1{}
        : gradient_partial_result_v1{gradient_partial_code_v1::nonzero_reserved,
                                     0};
}

} // namespace cellshard::compiler::partial
