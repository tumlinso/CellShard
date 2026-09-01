#pragma once

#include <CellShard/compiler/atom/logical_coverage_v1.hh>
#include <CellShard/compiler/atom/value_plane_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_gradient_plane_schema_version_v1 = 1;

enum class atom_gradient_ownership_v1 : std::uint32_t {
    primary = 1,
    physical_mirror = 2,
};

// value_layout supplies explicit bytes, numeric types, generation, and
// logical/projection order mapping. The remaining fields define what is being
// differentiated and how independently produced contributions accumulate.
struct atom_gradient_plane_v1 {
    atom_value_plane_v1 value_layout{};
    atom_logical_coverage_ref_v1 exact_target_coverage{};
    atom_persistent_identity_v1 gradient_target_identity{};
    atom_persistent_identity_v1 accumulation_algebra_identity{};
    atom_persistent_identity_v1 primary_gradient_plane_identity{};
    atom_gradient_ownership_v1 ownership = atom_gradient_ownership_v1::primary;
    std::uint32_t reserved = 0;
};

enum class atom_gradient_plane_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_value_layout,
    invalid_target_coverage,
    target_count_mismatch,
    invalid_gradient_target,
    invalid_accumulation_algebra,
    invalid_ownership,
    unexpected_primary_reference,
    missing_primary_reference,
    mirror_self_reference,
    nonzero_reserved,
};

struct atom_gradient_plane_validation_v1 {
    atom_gradient_plane_validation_code_v1 code =
        atom_gradient_plane_validation_code_v1::valid;
    std::uint64_t index = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_gradient_plane_validation_code_v1::valid;
    }
};

static_assert(offsetof(atom_gradient_plane_v1, value_layout) == 0,
              "gradient value storage must remain first");
static_assert(offsetof(atom_value_plane_v1, values) == 0,
              "nested gradient storage remains pointer-first");
static_assert(std::is_standard_layout<atom_gradient_plane_v1>::value);
static_assert(std::is_trivially_copyable<atom_gradient_plane_v1>::value);

[[nodiscard]] constexpr bool valid_atom_gradient_ownership_v1(
    atom_gradient_ownership_v1 ownership) noexcept {
    return ownership == atom_gradient_ownership_v1::primary
        || ownership == atom_gradient_ownership_v1::physical_mirror;
}

// O(target_count + dirty_count), with caller-owned canonical marks inherited
// from the value layout. No gradient storage, reduction buffer, or callback is
// allocated or discovered here.
[[nodiscard]] inline atom_gradient_plane_validation_v1
validate_atom_gradient_plane_v1(
    const atom_gradient_plane_v1 &plane,
    std::uint32_t coverage_source_validation,
    std::uint8_t *canonical_marks,
    std::uint64_t mark_capacity) noexcept {
    const auto value_result = validate_atom_value_plane_v1(
        plane.value_layout, canonical_marks, mark_capacity);
    if (!value_result.valid()) {
        return {atom_gradient_plane_validation_code_v1::invalid_value_layout,
                value_result.index,
                static_cast<std::uint32_t>(value_result.code)};
    }
    const auto coverage_result = validate_atom_logical_coverage_ref_v1(
        plane.exact_target_coverage, coverage_source_validation);
    if (!coverage_result.valid()) {
        return {atom_gradient_plane_validation_code_v1::
                    invalid_target_coverage,
                0, static_cast<std::uint32_t>(coverage_result.code)};
    }
    if (plane.exact_target_coverage.logical_count
        != plane.value_layout.canonical_element_count) {
        return {atom_gradient_plane_validation_code_v1::
                    target_count_mismatch,
                plane.exact_target_coverage.logical_count, 0};
    }
    if (!validate_atom_persistent_identity_v1(
             plane.gradient_target_identity)
             .valid()) {
        return {atom_gradient_plane_validation_code_v1::
                    invalid_gradient_target,
                0, 0};
    }
    if (!validate_atom_persistent_identity_v1(
             plane.accumulation_algebra_identity)
             .valid()) {
        return {atom_gradient_plane_validation_code_v1::
                    invalid_accumulation_algebra,
                0, 0};
    }
    if (!valid_atom_gradient_ownership_v1(plane.ownership)) {
        return {atom_gradient_plane_validation_code_v1::invalid_ownership,
                0, 0};
    }
    const bool valid_primary = validate_atom_persistent_identity_v1(
        plane.primary_gradient_plane_identity).valid();
    if (plane.ownership == atom_gradient_ownership_v1::primary
        && valid_primary) {
        return {atom_gradient_plane_validation_code_v1::
                    unexpected_primary_reference,
                0, 0};
    }
    if (plane.ownership == atom_gradient_ownership_v1::physical_mirror
        && !valid_primary) {
        return {atom_gradient_plane_validation_code_v1::
                    missing_primary_reference,
                0, 0};
    }
    if (plane.primary_gradient_plane_identity
        == plane.value_layout.plane_identity) {
        return {atom_gradient_plane_validation_code_v1::mirror_self_reference,
                0, 0};
    }
    if (plane.reserved != 0) {
        return {atom_gradient_plane_validation_code_v1::nonzero_reserved,
                0, 0};
    }
    return {atom_gradient_plane_validation_code_v1::valid,
            plane.value_layout.element_count, 0};
}

} // namespace cellshard::compiler::atom
