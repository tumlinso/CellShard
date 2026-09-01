#pragma once

#include <CellShard/compiler/atom/logical_coverage_v1.hh>
#include <CellShard/compiler/atom/relation_edge_spine_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_structural_plane_schema_version_v1 = 1;

enum class atom_structural_component_kind_v1 : std::uint32_t {
    support = 1,
    hierarchy = 2,
    relation_map = 3,
    segment_definition = 4,
    provider_defined = 5,
};

// Every component is immutable and source-linked. There is deliberately no
// mutable scientific-value pointer or value generation in this schema.
struct atom_structural_component_ref_v1 {
    const void *descriptor = nullptr;
    std::uint64_t descriptor_bytes = 0;
    atom_persistent_identity_v1 component_identity{};
    atom_persistent_identity_v1 descriptor_schema{};
    std::uint64_t structure_epoch = 0;
    atom_structural_component_kind_v1 kind =
        atom_structural_component_kind_v1::support;
    std::uint32_t descriptor_alignment = 0;
};

struct atom_structural_plane_v1 {
    const atom_structural_component_ref_v1 *components = nullptr;
    std::uint64_t component_count = 0;
    atom_persistent_identity_v1 plane_identity{};
    atom_persistent_identity_v1 persistent_order_identity{};
    std::uint64_t structure_epoch = 0;
    relation_edge_spine_view_v1 edge_spine{};
    atom_logical_coverage_ref_v1 exact_coverage{};
};

enum class atom_structural_plane_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_plane_identity,
    invalid_persistent_order,
    missing_structure_epoch,
    invalid_edge_spine,
    edge_spine_epoch_mismatch,
    invalid_exact_coverage,
    empty_components,
    missing_components,
    missing_descriptor,
    empty_descriptor,
    invalid_descriptor_alignment,
    misaligned_descriptor,
    invalid_component_identity,
    invalid_descriptor_schema,
    invalid_component_kind,
    stale_component_epoch,
    unordered_or_duplicate_component,
    missing_support,
    missing_relation_map,
};

struct atom_structural_plane_validation_v1 {
    atom_structural_plane_validation_code_v1 code =
        atom_structural_plane_validation_code_v1::valid;
    std::uint64_t index = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_structural_plane_validation_code_v1::valid;
    }
};

static_assert(offsetof(atom_structural_component_ref_v1, descriptor) == 0,
              "structural component references must remain pointer-first");
static_assert(std::is_standard_layout<atom_structural_component_ref_v1>::value);
static_assert(
    std::is_trivially_copyable<atom_structural_component_ref_v1>::value);
static_assert(offsetof(atom_structural_plane_v1, components) == 0,
              "structural planes must remain pointer-first");
static_assert(std::is_standard_layout<atom_structural_plane_v1>::value);
static_assert(std::is_trivially_copyable<atom_structural_plane_v1>::value);

[[nodiscard]] constexpr bool valid_atom_structural_component_kind_v1(
    atom_structural_component_kind_v1 kind) noexcept {
    const auto value = static_cast<std::uint32_t>(kind);
    return value >= 1 && value <= 5;
}

[[nodiscard]] constexpr bool atom_structural_component_less_v1(
    const atom_structural_component_ref_v1 &lhs,
    const atom_structural_component_ref_v1 &rhs) noexcept {
    return static_cast<std::uint32_t>(lhs.kind)
            < static_cast<std::uint32_t>(rhs.kind)
        || (lhs.kind == rhs.kind
            && atom_persistent_identity_less_v1(
                lhs.component_identity, rhs.component_identity));
}

// Structural validation is O(component_count + edge_count), O(1) storage and
// allocation-free. Cellerator exact-membership validation is consumed as an
// explicit diagnostic rather than reimplemented or inferred here.
[[nodiscard]] inline atom_structural_plane_validation_v1
validate_atom_structural_plane_v1(
    const atom_structural_plane_v1 &plane,
    std::uint32_t coverage_source_validation) noexcept {
    if (!validate_atom_persistent_identity_v1(plane.plane_identity).valid()) {
        return {atom_structural_plane_validation_code_v1::
                    invalid_plane_identity,
                0, 0};
    }
    if (!validate_atom_persistent_identity_v1(
             plane.persistent_order_identity)
             .valid()) {
        return {atom_structural_plane_validation_code_v1::
                    invalid_persistent_order,
                0, 0};
    }
    if (plane.structure_epoch == 0) {
        return {atom_structural_plane_validation_code_v1::
                    missing_structure_epoch,
                0, 0};
    }
    const auto spine_result = validate_relation_edge_spine_v1(plane.edge_spine);
    if (!spine_result.valid()) {
        return {atom_structural_plane_validation_code_v1::invalid_edge_spine,
                spine_result.index,
                static_cast<std::uint32_t>(spine_result.code)};
    }
    if (plane.edge_spine.structure_epoch != plane.structure_epoch) {
        return {atom_structural_plane_validation_code_v1::
                    edge_spine_epoch_mismatch,
                0, 0};
    }
    const auto coverage_result = validate_atom_logical_coverage_ref_v1(
        plane.exact_coverage, coverage_source_validation);
    if (!coverage_result.valid()) {
        return {atom_structural_plane_validation_code_v1::
                    invalid_exact_coverage,
                0, static_cast<std::uint32_t>(coverage_result.code)};
    }
    if (plane.component_count == 0) {
        return {atom_structural_plane_validation_code_v1::empty_components,
                0, 0};
    }
    if (plane.components == nullptr) {
        return {atom_structural_plane_validation_code_v1::missing_components,
                0, 0};
    }
    bool has_support = false;
    bool has_relation_map = false;
    for (std::uint64_t index = 0; index < plane.component_count; ++index) {
        const auto &component = plane.components[index];
        if (component.descriptor == nullptr) {
            return {atom_structural_plane_validation_code_v1::missing_descriptor,
                    index, 0};
        }
        if (component.descriptor_bytes == 0) {
            return {atom_structural_plane_validation_code_v1::empty_descriptor,
                    index, 0};
        }
        if (component.descriptor_alignment == 0
            || (component.descriptor_alignment
                & (component.descriptor_alignment - 1)) != 0) {
            return {atom_structural_plane_validation_code_v1::
                        invalid_descriptor_alignment,
                    index, 0};
        }
        if (reinterpret_cast<std::uintptr_t>(component.descriptor)
            % component.descriptor_alignment != 0) {
            return {atom_structural_plane_validation_code_v1::
                        misaligned_descriptor,
                    index, 0};
        }
        if (!validate_atom_persistent_identity_v1(
                 component.component_identity)
                 .valid()) {
            return {atom_structural_plane_validation_code_v1::
                        invalid_component_identity,
                    index, 0};
        }
        if (!validate_atom_persistent_identity_v1(component.descriptor_schema)
                 .valid()) {
            return {atom_structural_plane_validation_code_v1::
                        invalid_descriptor_schema,
                    index, 0};
        }
        if (!valid_atom_structural_component_kind_v1(component.kind)) {
            return {atom_structural_plane_validation_code_v1::
                        invalid_component_kind,
                    index, 0};
        }
        if (component.structure_epoch != plane.structure_epoch) {
            return {atom_structural_plane_validation_code_v1::
                        stale_component_epoch,
                    index, 0};
        }
        if (index != 0
            && !atom_structural_component_less_v1(
                plane.components[index - 1], component)) {
            return {atom_structural_plane_validation_code_v1::
                        unordered_or_duplicate_component,
                    index, 0};
        }
        has_support = has_support
            || component.kind == atom_structural_component_kind_v1::support;
        has_relation_map = has_relation_map
            || component.kind
                == atom_structural_component_kind_v1::relation_map;
    }
    if (!has_support) {
        return {atom_structural_plane_validation_code_v1::missing_support,
                plane.component_count, 0};
    }
    if (!has_relation_map) {
        return {atom_structural_plane_validation_code_v1::missing_relation_map,
                plane.component_count, 0};
    }
    return {atom_structural_plane_validation_code_v1::valid,
            plane.component_count, 0};
}

} // namespace cellshard::compiler::atom
