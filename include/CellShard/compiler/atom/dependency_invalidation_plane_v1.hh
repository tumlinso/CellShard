#pragma once

#include <CellShard/compiler/atom/persistent_identity_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t
    atom_dependency_invalidation_plane_schema_version_v1 = 1;

enum class atom_dependency_generation_kind_v1 : std::uint32_t {
    structure = 1,
    value = 2,
    state = 3,
    graph = 4,
    topology = 5,
    build = 6,
    provider_defined = 7,
};

enum class atom_dependency_effect_v1 : std::uint32_t {
    correctness = 1,
    preference = 2,
    performance = 3,
};

struct atom_dependency_requirement_v1 {
    atom_persistent_identity_v1 dependency_identity{};
    atom_persistent_identity_v1 generation_namespace{};
    std::uint64_t required_generation = 0;
    std::uint64_t observed_generation = 0;
    atom_dependency_generation_kind_v1 generation_kind =
        atom_dependency_generation_kind_v1::structure;
    atom_dependency_effect_v1 effect = atom_dependency_effect_v1::correctness;
};

struct atom_dependency_invalidation_plane_v1 {
    const atom_dependency_requirement_v1 *dependencies = nullptr;
    std::uint64_t dependency_count = 0;
    atom_persistent_identity_v1 plane_identity{};
    std::uint64_t validation_generation = 0;
};

enum class atom_dependency_invalidation_code_v1 : std::uint32_t {
    valid = 0,
    empty_dependencies,
    missing_dependencies,
    invalid_plane_identity,
    missing_validation_generation,
    invalid_dependency_identity,
    invalid_generation_namespace,
    missing_required_generation,
    missing_observed_generation,
    invalid_generation_kind,
    invalid_effect,
    unordered_or_duplicate_dependency,
    stale_correctness_dependency,
};

struct atom_dependency_invalidation_validation_v1 {
    atom_dependency_invalidation_code_v1 code =
        atom_dependency_invalidation_code_v1::valid;
    std::uint64_t index = 0;
    std::uint64_t stale_preference_count = 0;
    std::uint64_t stale_performance_count = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_dependency_invalidation_code_v1::valid;
    }

    [[nodiscard]] constexpr bool preference_fresh() const noexcept {
        return valid() && stale_preference_count == 0;
    }

    [[nodiscard]] constexpr bool performance_fresh() const noexcept {
        return valid() && stale_performance_count == 0;
    }
};

static_assert(std::is_standard_layout<atom_dependency_requirement_v1>::value);
static_assert(
    std::is_trivially_copyable<atom_dependency_requirement_v1>::value);
static_assert(
    offsetof(atom_dependency_invalidation_plane_v1, dependencies) == 0,
    "dependency-invalidation planes must remain pointer-first");
static_assert(
    std::is_standard_layout<atom_dependency_invalidation_plane_v1>::value);
static_assert(std::is_trivially_copyable<
              atom_dependency_invalidation_plane_v1>::value);

[[nodiscard]] constexpr bool valid_atom_dependency_generation_kind_v1(
    atom_dependency_generation_kind_v1 kind) noexcept {
    const auto value = static_cast<std::uint32_t>(kind);
    return value >= 1 && value <= 7;
}

[[nodiscard]] constexpr bool valid_atom_dependency_effect_v1(
    atom_dependency_effect_v1 effect) noexcept {
    const auto value = static_cast<std::uint32_t>(effect);
    return value >= 1 && value <= 3;
}

[[nodiscard]] constexpr bool atom_dependency_requirement_less_v1(
    const atom_dependency_requirement_v1 &lhs,
    const atom_dependency_requirement_v1 &rhs) noexcept {
    return atom_persistent_identity_less_v1(
               lhs.dependency_identity, rhs.dependency_identity)
        || (lhs.dependency_identity == rhs.dependency_identity
            && (static_cast<std::uint32_t>(lhs.generation_kind)
                    < static_cast<std::uint32_t>(rhs.generation_kind)
                || (lhs.generation_kind == rhs.generation_kind
                    && atom_persistent_identity_less_v1(
                        lhs.generation_namespace,
                        rhs.generation_namespace))));
}

// O(dependency_count), O(1) storage, and allocation-free. Stale correctness
// dependencies invalidate the plane. Stale preferences and measurements remain
// structurally valid but are reported independently for replanning/remeasurement.
[[nodiscard]] constexpr atom_dependency_invalidation_validation_v1
validate_atom_dependency_invalidation_plane_v1(
    const atom_dependency_invalidation_plane_v1 &plane) noexcept {
    if (plane.dependency_count == 0) {
        return {atom_dependency_invalidation_code_v1::empty_dependencies,
                0, 0, 0};
    }
    if (plane.dependencies == nullptr) {
        return {atom_dependency_invalidation_code_v1::missing_dependencies,
                0, 0, 0};
    }
    if (!validate_atom_persistent_identity_v1(plane.plane_identity).valid()) {
        return {atom_dependency_invalidation_code_v1::invalid_plane_identity,
                0, 0, 0};
    }
    if (plane.validation_generation == 0) {
        return {atom_dependency_invalidation_code_v1::
                    missing_validation_generation,
                0, 0, 0};
    }
    std::uint64_t stale_preference_count = 0;
    std::uint64_t stale_performance_count = 0;
    for (std::uint64_t index = 0; index < plane.dependency_count; ++index) {
        const auto &dependency = plane.dependencies[index];
        if (!validate_atom_persistent_identity_v1(
                 dependency.dependency_identity)
                 .valid()) {
            return {atom_dependency_invalidation_code_v1::
                        invalid_dependency_identity,
                    index, stale_preference_count, stale_performance_count};
        }
        if (!validate_atom_persistent_identity_v1(
                 dependency.generation_namespace)
                 .valid()) {
            return {atom_dependency_invalidation_code_v1::
                        invalid_generation_namespace,
                    index, stale_preference_count, stale_performance_count};
        }
        if (dependency.required_generation == 0) {
            return {atom_dependency_invalidation_code_v1::
                        missing_required_generation,
                    index, stale_preference_count, stale_performance_count};
        }
        if (dependency.observed_generation == 0) {
            return {atom_dependency_invalidation_code_v1::
                        missing_observed_generation,
                    index, stale_preference_count, stale_performance_count};
        }
        if (!valid_atom_dependency_generation_kind_v1(
                dependency.generation_kind)) {
            return {atom_dependency_invalidation_code_v1::
                        invalid_generation_kind,
                    index, stale_preference_count, stale_performance_count};
        }
        if (!valid_atom_dependency_effect_v1(dependency.effect)) {
            return {atom_dependency_invalidation_code_v1::invalid_effect,
                    index, stale_preference_count, stale_performance_count};
        }
        if (index != 0
            && !atom_dependency_requirement_less_v1(
                plane.dependencies[index - 1], dependency)) {
            return {atom_dependency_invalidation_code_v1::
                        unordered_or_duplicate_dependency,
                    index, stale_preference_count, stale_performance_count};
        }
        if (dependency.required_generation == dependency.observed_generation) {
            continue;
        }
        if (dependency.effect == atom_dependency_effect_v1::correctness) {
            return {atom_dependency_invalidation_code_v1::
                        stale_correctness_dependency,
                    index, stale_preference_count, stale_performance_count};
        }
        if (dependency.effect == atom_dependency_effect_v1::preference) {
            ++stale_preference_count;
        } else {
            ++stale_performance_count;
        }
    }
    return {atom_dependency_invalidation_code_v1::valid,
            plane.dependency_count,
            stale_preference_count,
            stale_performance_count};
}

} // namespace cellshard::compiler::atom
