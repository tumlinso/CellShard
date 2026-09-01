#pragma once

#include <CellShard/compiler/partial/partial_atom_v1.hh>
#include <CellShard/compiler/atom/dependency_invalidation_plane_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::partial {

using atom::atom_dependency_generation_kind_v1;

inline constexpr std::uint32_t partial_dependency_closure_schema_version_v1 = 1;

enum class partial_dependency_role_v1 : std::uint32_t {
    direct = 1,
    transitive = 2,
};

struct partial_dependency_requirement_v1 {
    atom_persistent_identity_v1 dependency_identity{};
    atom_persistent_identity_v1 generation_namespace{};
    std::uint64_t captured_generation = 0;
    atom_dependency_generation_kind_v1 generation_kind =
        atom_dependency_generation_kind_v1::structure;
    partial_dependency_role_v1 role = partial_dependency_role_v1::direct;
};

// This is the complete, certified correctness dependency closure captured when
// the partial was materialized. Preference-only and performance-only inputs do
// not belong here and cannot authorize reuse.
struct partial_dependency_closure_view_v1 {
    const partial_dependency_requirement_v1 *dependencies = nullptr;
    std::uint64_t dependency_count = 0;
    atom_persistent_identity_v1 closure_identity{};
    atom_persistent_identity_v1 exact_certification_identity{};
    std::uint32_t schema_version = partial_dependency_closure_schema_version_v1;
    std::uint32_t reserved = 0;
};

struct partial_dependency_observation_v1 {
    atom_persistent_identity_v1 dependency_identity{};
    atom_persistent_identity_v1 generation_namespace{};
    std::uint64_t current_generation = 0;
    atom_dependency_generation_kind_v1 generation_kind =
        atom_dependency_generation_kind_v1::structure;
    std::uint32_t reserved = 0;
};

struct partial_dependency_observation_view_v1 {
    const partial_dependency_observation_v1 *observations = nullptr;
    std::uint64_t observation_count = 0;
};

enum class partial_dependency_validation_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_closure_identity,
    invalid_exact_certification,
    missing_dependencies,
    invalid_dependency_identity,
    self_dependency,
    invalid_generation_namespace,
    missing_captured_generation,
    invalid_generation_kind,
    invalid_role,
    unordered_or_duplicate_dependency,
    nonzero_reserved,
};

struct partial_dependency_validation_v1 {
    partial_dependency_validation_code_v1 code =
        partial_dependency_validation_code_v1::valid;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == partial_dependency_validation_code_v1::valid;
    }
};

enum class partial_freshness_v1 : std::uint32_t {
    current = 1,
    stale = 2,
    unproven = 3,
    invalid = 4,
};

enum class partial_freshness_reason_v1 : std::uint32_t {
    all_generations_match = 0,
    invalid_closure,
    closure_binding_mismatch,
    inconsistent_observation_pointer,
    invalid_observation,
    unordered_or_duplicate_observation,
    missing_observation,
    generation_mismatch,
};

struct partial_freshness_result_v1 {
    partial_freshness_v1 freshness = partial_freshness_v1::invalid;
    partial_freshness_reason_v1 reason =
        partial_freshness_reason_v1::invalid_closure;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool reusable() const noexcept {
        return freshness == partial_freshness_v1::current;
    }
};

static_assert(offsetof(partial_dependency_closure_view_v1, dependencies) == 0);
static_assert(std::is_standard_layout<partial_dependency_requirement_v1>::value);
static_assert(std::is_trivially_copyable<partial_dependency_requirement_v1>::value);
static_assert(std::is_standard_layout<partial_dependency_closure_view_v1>::value);
static_assert(std::is_trivially_copyable<partial_dependency_closure_view_v1>::value);
static_assert(std::is_standard_layout<partial_dependency_observation_v1>::value);
static_assert(std::is_trivially_copyable<partial_dependency_observation_v1>::value);

[[nodiscard]] constexpr bool valid_partial_dependency_role_v1(
    partial_dependency_role_v1 role) noexcept {
    return role == partial_dependency_role_v1::direct
        || role == partial_dependency_role_v1::transitive;
}

[[nodiscard]] constexpr bool partial_dependency_key_less_v1(
    const partial_dependency_requirement_v1 &lhs,
    const partial_dependency_requirement_v1 &rhs) noexcept {
    return atom::atom_persistent_identity_less_v1(
               lhs.dependency_identity, rhs.dependency_identity)
        || (lhs.dependency_identity == rhs.dependency_identity
            && (atom::atom_persistent_identity_less_v1(
                    lhs.generation_namespace, rhs.generation_namespace)
                || (lhs.generation_namespace == rhs.generation_namespace
                    && static_cast<std::uint32_t>(lhs.generation_kind)
                        < static_cast<std::uint32_t>(rhs.generation_kind))));
}

[[nodiscard]] constexpr bool partial_observation_key_less_v1(
    const partial_dependency_observation_v1 &lhs,
    const partial_dependency_observation_v1 &rhs) noexcept {
    return atom::atom_persistent_identity_less_v1(
               lhs.dependency_identity, rhs.dependency_identity)
        || (lhs.dependency_identity == rhs.dependency_identity
            && (atom::atom_persistent_identity_less_v1(
                    lhs.generation_namespace, rhs.generation_namespace)
                || (lhs.generation_namespace == rhs.generation_namespace
                    && static_cast<std::uint32_t>(lhs.generation_kind)
                        < static_cast<std::uint32_t>(rhs.generation_kind))));
}

[[nodiscard]] constexpr bool partial_dependency_key_equal_v1(
    const partial_dependency_requirement_v1 &dependency,
    const partial_dependency_observation_v1 &observation) noexcept {
    return dependency.dependency_identity == observation.dependency_identity
        && dependency.generation_namespace == observation.generation_namespace
        && dependency.generation_kind == observation.generation_kind;
}

[[nodiscard]] inline partial_dependency_validation_v1
validate_partial_dependency_closure_v1(
    const partial_dependency_closure_view_v1 &closure,
    atom_persistent_identity_v1 partial_identity) noexcept {
    if (closure.schema_version != partial_dependency_closure_schema_version_v1) {
        return {partial_dependency_validation_code_v1::unsupported_schema, 0};
    }
    if (!atom::validate_atom_persistent_identity_v1(closure.closure_identity)
             .valid()) {
        return {partial_dependency_validation_code_v1::invalid_closure_identity,
                0};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             closure.exact_certification_identity)
             .valid()) {
        return {partial_dependency_validation_code_v1::
                    invalid_exact_certification,
                0};
    }
    if (closure.dependency_count == 0 || closure.dependencies == nullptr) {
        return {partial_dependency_validation_code_v1::missing_dependencies, 0};
    }
    for (std::uint64_t index = 0; index < closure.dependency_count; ++index) {
        const auto &dependency = closure.dependencies[index];
        if (!atom::validate_atom_persistent_identity_v1(
                 dependency.dependency_identity)
                 .valid()) {
            return {partial_dependency_validation_code_v1::
                        invalid_dependency_identity,
                    index};
        }
        if (dependency.dependency_identity == partial_identity) {
            return {partial_dependency_validation_code_v1::self_dependency,
                    index};
        }
        if (!atom::validate_atom_persistent_identity_v1(
                 dependency.generation_namespace)
                 .valid()) {
            return {partial_dependency_validation_code_v1::
                        invalid_generation_namespace,
                    index};
        }
        if (dependency.captured_generation == 0) {
            return {partial_dependency_validation_code_v1::
                        missing_captured_generation,
                    index};
        }
        if (!atom::valid_atom_dependency_generation_kind_v1(
                dependency.generation_kind)) {
            return {partial_dependency_validation_code_v1::
                        invalid_generation_kind,
                    index};
        }
        if (!valid_partial_dependency_role_v1(dependency.role)) {
            return {partial_dependency_validation_code_v1::invalid_role, index};
        }
        if (index != 0
            && !partial_dependency_key_less_v1(
                closure.dependencies[index - 1], dependency)) {
            return {partial_dependency_validation_code_v1::
                        unordered_or_duplicate_dependency,
                    index};
        }
    }
    if (closure.reserved != 0) {
        return {partial_dependency_validation_code_v1::nonzero_reserved, 0};
    }
    return {partial_dependency_validation_code_v1::valid,
            closure.dependency_count};
}

// O(dependencies + observations), allocation-free merge comparison over two
// canonical sorted tables. Any invalid, missing, or unproven generation fails
// closed: only an exact all-generation match permits reuse.
[[nodiscard]] inline partial_freshness_result_v1 evaluate_partial_freshness_v1(
    const partial_atom_view_v1 &partial,
    const partial_dependency_closure_view_v1 &closure,
    partial_dependency_observation_view_v1 observations) noexcept {
    const auto closure_result = validate_partial_dependency_closure_v1(
        closure, partial.header.partial_identity);
    if (!closure_result.valid()) {
        return {partial_freshness_v1::invalid,
                partial_freshness_reason_v1::invalid_closure,
                closure_result.index};
    }
    if (closure.closure_identity
        != partial.header.dependency_closure_identity) {
        return {partial_freshness_v1::invalid,
                partial_freshness_reason_v1::closure_binding_mismatch, 0};
    }
    if ((observations.observation_count == 0)
        != (observations.observations == nullptr)) {
        return {partial_freshness_v1::invalid,
                partial_freshness_reason_v1::
                    inconsistent_observation_pointer,
                0};
    }
    for (std::uint64_t index = 0; index < observations.observation_count;
         ++index) {
        const auto &observation = observations.observations[index];
        if (!atom::validate_atom_persistent_identity_v1(
                 observation.dependency_identity)
                 .valid()
            || !atom::validate_atom_persistent_identity_v1(
                    observation.generation_namespace)
                    .valid()
            || observation.current_generation == 0
            || !atom::valid_atom_dependency_generation_kind_v1(
                observation.generation_kind)
            || observation.reserved != 0) {
            return {partial_freshness_v1::invalid,
                    partial_freshness_reason_v1::invalid_observation, index};
        }
        if (index != 0
            && !partial_observation_key_less_v1(
                observations.observations[index - 1], observation)) {
            return {partial_freshness_v1::invalid,
                    partial_freshness_reason_v1::
                        unordered_or_duplicate_observation,
                    index};
        }
    }
    std::uint64_t observation_index = 0;
    for (std::uint64_t index = 0; index < closure.dependency_count; ++index) {
        const auto &dependency = closure.dependencies[index];
        while (observation_index < observations.observation_count) {
            const auto &observation = observations.observations[observation_index];
            partial_dependency_requirement_v1 observation_key{
                observation.dependency_identity,
                observation.generation_namespace,
                observation.current_generation,
                observation.generation_kind,
                partial_dependency_role_v1::direct};
            if (!partial_dependency_key_less_v1(observation_key, dependency)) {
                break;
            }
            ++observation_index;
        }
        if (observation_index == observations.observation_count
            || !partial_dependency_key_equal_v1(
                dependency, observations.observations[observation_index])) {
            return {partial_freshness_v1::unproven,
                    partial_freshness_reason_v1::missing_observation, index};
        }
        if (dependency.captured_generation
            != observations.observations[observation_index].current_generation) {
            return {partial_freshness_v1::stale,
                    partial_freshness_reason_v1::generation_mismatch, index};
        }
        ++observation_index;
    }
    return {partial_freshness_v1::current,
            partial_freshness_reason_v1::all_generations_match,
            closure.dependency_count};
}

} // namespace cellshard::compiler::partial
