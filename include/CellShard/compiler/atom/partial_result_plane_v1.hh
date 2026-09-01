#pragma once

#include <CellShard/compiler/atom/logical_coverage_v1.hh>
#include <CellShard/compiler/atom/value_plane_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_partial_result_plane_schema_version_v1 = 1;

enum class atom_partial_result_status_v1 : std::uint32_t {
    accumulating = 1,
    ready_to_merge = 2,
    merged = 3,
    finalized = 4,
};

struct atom_dependency_generation_v1 {
    atom_persistent_identity_v1 dependency_identity{};
    atom_persistent_identity_v1 generation_kind{};
    std::uint64_t generation = 0;
};

// A partial result has exact contribution coverage and a separately versioned
// Cellerator reconstruction algebra. Physical overlap never implies that two
// records own the same logical contribution.
struct atom_partial_result_plane_v1 {
    atom_value_plane_v1 partial_layout{};
    atom_logical_coverage_ref_v1 exact_contribution_coverage{};
    atom_persistent_identity_v1 contribution_owner_identity{};
    atom_persistent_identity_v1 reconstruction_algebra_identity{};
    atom_persistent_identity_v1 numerical_policy_identity{};
    atom_persistent_identity_v1 finalized_output_identity{};
    const atom_dependency_generation_v1 *dependencies = nullptr;
    std::uint64_t dependency_count = 0;
    std::uint64_t merge_generation = 0;
    atom_partial_result_status_v1 status =
        atom_partial_result_status_v1::accumulating;
    std::uint32_t reserved = 0;
};

enum class atom_partial_result_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_partial_layout,
    invalid_contribution_coverage,
    contribution_count_mismatch,
    invalid_contribution_owner,
    invalid_reconstruction_algebra,
    invalid_numerical_policy,
    missing_dependencies,
    invalid_dependency_identity,
    invalid_generation_kind,
    missing_dependency_generation,
    unordered_or_duplicate_dependency,
    invalid_status,
    premature_merge_generation,
    missing_merge_generation,
    unexpected_finalized_output,
    missing_finalized_output,
    nonzero_reserved,
};

struct atom_partial_result_validation_v1 {
    atom_partial_result_validation_code_v1 code =
        atom_partial_result_validation_code_v1::valid;
    std::uint64_t index = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_partial_result_validation_code_v1::valid;
    }
};

static_assert(offsetof(atom_partial_result_plane_v1, partial_layout) == 0);
static_assert(offsetof(atom_value_plane_v1, values) == 0);
static_assert(std::is_standard_layout<atom_dependency_generation_v1>::value);
static_assert(std::is_trivially_copyable<atom_dependency_generation_v1>::value);
static_assert(std::is_standard_layout<atom_partial_result_plane_v1>::value);
static_assert(std::is_trivially_copyable<atom_partial_result_plane_v1>::value);

[[nodiscard]] constexpr bool valid_atom_partial_result_status_v1(
    atom_partial_result_status_v1 status) noexcept {
    const auto value = static_cast<std::uint32_t>(status);
    return value >= 1 && value <= 4;
}

// O(partial elements + dirty elements + dependencies), using the caller-owned
// canonical marks required by partial_layout. No merge/finalize is executed.
[[nodiscard]] inline atom_partial_result_validation_v1
validate_atom_partial_result_plane_v1(
    const atom_partial_result_plane_v1 &plane,
    std::uint32_t coverage_source_validation,
    std::uint8_t *canonical_marks,
    std::uint64_t mark_capacity) noexcept {
    const auto layout_result = validate_atom_value_plane_v1(
        plane.partial_layout, canonical_marks, mark_capacity);
    if (!layout_result.valid()) {
        return {atom_partial_result_validation_code_v1::invalid_partial_layout,
                layout_result.index,
                static_cast<std::uint32_t>(layout_result.code)};
    }
    const auto coverage_result = validate_atom_logical_coverage_ref_v1(
        plane.exact_contribution_coverage, coverage_source_validation);
    if (!coverage_result.valid()) {
        return {atom_partial_result_validation_code_v1::
                    invalid_contribution_coverage,
                0, static_cast<std::uint32_t>(coverage_result.code)};
    }
    if (plane.exact_contribution_coverage.logical_count
        != plane.partial_layout.canonical_element_count) {
        return {atom_partial_result_validation_code_v1::
                    contribution_count_mismatch,
                plane.exact_contribution_coverage.logical_count, 0};
    }
#define CELLSHARD_ATOM_PARTIAL_CHECK_ID(field, code) \
    if (!validate_atom_persistent_identity_v1(plane.field).valid()) { \
        return {atom_partial_result_validation_code_v1::code, 0, 0}; \
    }
    CELLSHARD_ATOM_PARTIAL_CHECK_ID(contribution_owner_identity,
                                    invalid_contribution_owner)
    CELLSHARD_ATOM_PARTIAL_CHECK_ID(reconstruction_algebra_identity,
                                    invalid_reconstruction_algebra)
    CELLSHARD_ATOM_PARTIAL_CHECK_ID(numerical_policy_identity,
                                    invalid_numerical_policy)
#undef CELLSHARD_ATOM_PARTIAL_CHECK_ID
    if (plane.dependency_count == 0 || plane.dependencies == nullptr) {
        return {atom_partial_result_validation_code_v1::missing_dependencies,
                0, 0};
    }
    for (std::uint64_t index = 0; index < plane.dependency_count; ++index) {
        const auto &dependency = plane.dependencies[index];
        if (!validate_atom_persistent_identity_v1(
                 dependency.dependency_identity)
                 .valid()) {
            return {atom_partial_result_validation_code_v1::
                        invalid_dependency_identity,
                    index, 0};
        }
        if (!validate_atom_persistent_identity_v1(dependency.generation_kind)
                 .valid()) {
            return {atom_partial_result_validation_code_v1::
                        invalid_generation_kind,
                    index, 0};
        }
        if (dependency.generation == 0) {
            return {atom_partial_result_validation_code_v1::
                        missing_dependency_generation,
                    index, 0};
        }
        if (index != 0
            && !atom_persistent_identity_less_v1(
                plane.dependencies[index - 1].dependency_identity,
                dependency.dependency_identity)) {
            return {atom_partial_result_validation_code_v1::
                        unordered_or_duplicate_dependency,
                    index, 0};
        }
    }
    if (!valid_atom_partial_result_status_v1(plane.status)) {
        return {atom_partial_result_validation_code_v1::invalid_status, 0, 0};
    }
    const bool completed = plane.status == atom_partial_result_status_v1::merged
        || plane.status == atom_partial_result_status_v1::finalized;
    if (!completed && plane.merge_generation != 0) {
        return {atom_partial_result_validation_code_v1::
                    premature_merge_generation,
                0, 0};
    }
    if (completed && plane.merge_generation == 0) {
        return {atom_partial_result_validation_code_v1::
                    missing_merge_generation,
                0, 0};
    }
    const bool output_valid = validate_atom_persistent_identity_v1(
        plane.finalized_output_identity).valid();
    if (plane.status != atom_partial_result_status_v1::finalized
        && output_valid) {
        return {atom_partial_result_validation_code_v1::
                    unexpected_finalized_output,
                0, 0};
    }
    if (plane.status == atom_partial_result_status_v1::finalized
        && !output_valid) {
        return {atom_partial_result_validation_code_v1::
                    missing_finalized_output,
                0, 0};
    }
    if (plane.reserved != 0) {
        return {atom_partial_result_validation_code_v1::nonzero_reserved,
                0, 0};
    }
    return {atom_partial_result_validation_code_v1::valid,
            plane.dependency_count, 0};
}

} // namespace cellshard::compiler::atom
