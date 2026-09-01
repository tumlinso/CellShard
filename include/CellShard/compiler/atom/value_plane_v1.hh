#pragma once

#include <CellShard/compiler/atom/port_v1.hh>
#include <CellShard/compiler/atom/relation_edge_spine_v1.hh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_value_plane_schema_version_v1 = 1;

enum class atom_value_subject_kind_v1 : std::uint32_t {
    relation_edge = 1,
    feature = 2,
    provider_defined = 3,
};

enum class atom_value_ownership_v1 : std::uint32_t {
    logical_primary = 1,
    projection_primary = 2,
};

struct atom_compact_index_array_v1 {
    const std::byte *indices = nullptr;
    std::uint64_t index_count = 0;
    std::uint64_t index_bytes = 0;
    compact_edge_index_width_v1 width = compact_edge_index_width_v1::u8;
    std::uint8_t reserved[7]{};
};

// Values are mutable launch state. Structure identity/epoch, order, canonical
// mapping and value generation are explicit and have independent lifetimes.
struct atom_value_plane_v1 {
    void *values = nullptr;
    std::uint64_t value_bytes = 0;
    std::uint64_t element_count = 0;
    std::uint64_t element_stride_bytes = 0;
    std::uint32_t element_bytes = 0;
    std::uint32_t value_alignment = 0;
    atom_persistent_identity_v1 plane_identity{};
    atom_persistent_identity_v1 structure_plane_identity{};
    atom_persistent_identity_v1 subject_space_identity{};
    atom_persistent_identity_v1 persistent_order_identity{};
    atom_port_numeric_v1 numeric{};
    std::uint64_t structure_epoch = 0;
    std::uint64_t value_generation = 0;
    std::uint64_t canonical_element_count = 0;
    atom_compact_index_array_v1 local_to_canonical{};
    atom_compact_index_array_v1 dirty_local_indices{};
    atom_value_subject_kind_v1 subject_kind =
        atom_value_subject_kind_v1::relation_edge;
    atom_value_ownership_v1 ownership =
        atom_value_ownership_v1::logical_primary;
};

enum class atom_value_plane_validation_code_v1 : std::uint32_t {
    valid = 0,
    missing_values,
    empty_values,
    invalid_element_bytes,
    invalid_stride,
    value_bytes_overflow,
    insufficient_value_bytes,
    invalid_value_alignment,
    misaligned_values,
    invalid_plane_identity,
    invalid_structure_plane_identity,
    invalid_subject_space_identity,
    invalid_persistent_order,
    invalid_numeric,
    missing_structure_epoch,
    missing_value_generation,
    invalid_subject_kind,
    invalid_ownership,
    invalid_canonical_count,
    missing_canonical_map,
    invalid_canonical_map_bytes,
    canonical_map_overflow,
    canonical_index_out_of_range,
    duplicate_canonical_index,
    nonidentity_logical_primary,
    inconsistent_dirty_pointer,
    invalid_dirty_bytes,
    dirty_index_out_of_range,
    unordered_or_duplicate_dirty_index,
    nonzero_reserved,
    missing_marks,
    insufficient_marks,
};

struct atom_value_plane_validation_v1 {
    atom_value_plane_validation_code_v1 code =
        atom_value_plane_validation_code_v1::valid;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_value_plane_validation_code_v1::valid;
    }
};

static_assert(offsetof(atom_compact_index_array_v1, indices) == 0);
static_assert(std::is_standard_layout<atom_compact_index_array_v1>::value);
static_assert(std::is_trivially_copyable<atom_compact_index_array_v1>::value);
static_assert(offsetof(atom_value_plane_v1, values) == 0,
              "value planes must remain pointer-first");
static_assert(std::is_standard_layout<atom_value_plane_v1>::value);
static_assert(std::is_trivially_copyable<atom_value_plane_v1>::value);

[[nodiscard]] constexpr bool valid_atom_value_subject_kind_v1(
    atom_value_subject_kind_v1 kind) noexcept {
    const auto value = static_cast<std::uint32_t>(kind);
    return value >= 1 && value <= 3;
}

[[nodiscard]] constexpr bool valid_atom_value_ownership_v1(
    atom_value_ownership_v1 ownership) noexcept {
    return ownership == atom_value_ownership_v1::logical_primary
        || ownership == atom_value_ownership_v1::projection_primary;
}

[[nodiscard]] constexpr bool valid_atom_port_numeric_v1(
    const atom_port_numeric_v1 &numeric) noexcept {
    return validate_atom_persistent_identity_v1(numeric.storage_type).valid()
        && validate_atom_persistent_identity_v1(numeric.logical_type).valid()
        && validate_atom_persistent_identity_v1(numeric.accumulation_type)
               .valid();
}

namespace detail {

[[nodiscard]] inline atom_value_plane_validation_v1 validate_compact_array_v1(
    atom_compact_index_array_v1 array,
    bool empty_allowed,
    atom_value_plane_validation_code_v1 missing_code,
    atom_value_plane_validation_code_v1 bytes_code) noexcept {
    if (array.index_count == 0) {
        if (array.indices != nullptr || array.index_bytes != 0) {
            return {atom_value_plane_validation_code_v1::
                        inconsistent_dirty_pointer,
                    0};
        }
        return empty_allowed
            ? atom_value_plane_validation_v1{}
            : atom_value_plane_validation_v1{missing_code, 0};
    }
    if (array.indices == nullptr) {
        return {missing_code, 0};
    }
    if (!valid_compact_edge_index_width_v1(array.width)) {
        return {bytes_code, 0};
    }
    const auto width = static_cast<std::uint8_t>(array.width);
    if (array.index_count
        > std::numeric_limits<std::uint64_t>::max() / width) {
        return {atom_value_plane_validation_code_v1::canonical_map_overflow, 0};
    }
    if (array.index_bytes != array.index_count * width) {
        return {bytes_code, array.index_bytes};
    }
    for (const auto reserved : array.reserved) {
        if (reserved != 0) {
            return {atom_value_plane_validation_code_v1::nonzero_reserved, 0};
        }
    }
    return {};
}

} // namespace detail

// O(element_count + dirty_count) time and one caller-owned mark byte per
// canonical element. The function allocates nothing and never hashes values.
[[nodiscard]] inline atom_value_plane_validation_v1
validate_atom_value_plane_v1(
    const atom_value_plane_v1 &plane,
    std::uint8_t *canonical_marks,
    std::uint64_t mark_capacity) noexcept {
    if (plane.values == nullptr) {
        return {atom_value_plane_validation_code_v1::missing_values, 0};
    }
    if (plane.element_count == 0 || plane.value_bytes == 0) {
        return {atom_value_plane_validation_code_v1::empty_values, 0};
    }
    if (plane.element_bytes == 0) {
        return {atom_value_plane_validation_code_v1::invalid_element_bytes, 0};
    }
    if (plane.element_stride_bytes < plane.element_bytes) {
        return {atom_value_plane_validation_code_v1::invalid_stride, 0};
    }
    if (plane.element_count - 1
        > (std::numeric_limits<std::uint64_t>::max() - plane.element_bytes)
              / plane.element_stride_bytes) {
        return {atom_value_plane_validation_code_v1::value_bytes_overflow, 0};
    }
    const auto required_bytes = (plane.element_count - 1)
            * plane.element_stride_bytes
        + plane.element_bytes;
    if (plane.value_bytes < required_bytes) {
        return {atom_value_plane_validation_code_v1::insufficient_value_bytes,
                required_bytes};
    }
    if (plane.value_alignment == 0
        || (plane.value_alignment & (plane.value_alignment - 1)) != 0) {
        return {atom_value_plane_validation_code_v1::invalid_value_alignment,
                0};
    }
    if (reinterpret_cast<std::uintptr_t>(plane.values)
        % plane.value_alignment != 0) {
        return {atom_value_plane_validation_code_v1::misaligned_values, 0};
    }
    if (!validate_atom_persistent_identity_v1(plane.plane_identity).valid()) {
        return {atom_value_plane_validation_code_v1::invalid_plane_identity, 0};
    }
    if (!validate_atom_persistent_identity_v1(
             plane.structure_plane_identity)
             .valid()) {
        return {atom_value_plane_validation_code_v1::
                    invalid_structure_plane_identity,
                0};
    }
    if (!validate_atom_persistent_identity_v1(plane.subject_space_identity)
             .valid()) {
        return {atom_value_plane_validation_code_v1::
                    invalid_subject_space_identity,
                0};
    }
    if (!validate_atom_persistent_identity_v1(
             plane.persistent_order_identity)
             .valid()) {
        return {atom_value_plane_validation_code_v1::invalid_persistent_order,
                0};
    }
    if (!valid_atom_port_numeric_v1(plane.numeric)) {
        return {atom_value_plane_validation_code_v1::invalid_numeric, 0};
    }
    if (plane.structure_epoch == 0) {
        return {atom_value_plane_validation_code_v1::missing_structure_epoch,
                0};
    }
    if (plane.value_generation == 0) {
        return {atom_value_plane_validation_code_v1::missing_value_generation,
                0};
    }
    if (!valid_atom_value_subject_kind_v1(plane.subject_kind)) {
        return {atom_value_plane_validation_code_v1::invalid_subject_kind, 0};
    }
    if (!valid_atom_value_ownership_v1(plane.ownership)) {
        return {atom_value_plane_validation_code_v1::invalid_ownership, 0};
    }
    if (plane.canonical_element_count == 0
        || plane.element_count != plane.canonical_element_count) {
        return {atom_value_plane_validation_code_v1::invalid_canonical_count,
                plane.canonical_element_count};
    }
    auto result = detail::validate_compact_array_v1(
        plane.local_to_canonical, false,
        atom_value_plane_validation_code_v1::missing_canonical_map,
        atom_value_plane_validation_code_v1::invalid_canonical_map_bytes);
    if (!result.valid()) return result;
    if (plane.local_to_canonical.index_count != plane.element_count) {
        return {atom_value_plane_validation_code_v1::invalid_canonical_count,
                plane.local_to_canonical.index_count};
    }
    if (plane.canonical_element_count
        > compact_edge_index_capacity_v1(plane.local_to_canonical.width)) {
        return {atom_value_plane_validation_code_v1::
                    invalid_canonical_map_bytes,
                0};
    }
    if (canonical_marks == nullptr) {
        return {atom_value_plane_validation_code_v1::missing_marks, 0};
    }
    if (mark_capacity < plane.canonical_element_count) {
        return {atom_value_plane_validation_code_v1::insufficient_marks,
                mark_capacity};
    }
    for (std::uint64_t index = 0; index < plane.canonical_element_count;
         ++index) {
        canonical_marks[index] = 0;
    }
    for (std::uint64_t index = 0; index < plane.element_count; ++index) {
        const auto canonical = read_compact_edge_index_v1(
            plane.local_to_canonical.indices, index,
            plane.local_to_canonical.width);
        if (canonical >= plane.canonical_element_count) {
            return {atom_value_plane_validation_code_v1::
                        canonical_index_out_of_range,
                    index};
        }
        if (canonical_marks[canonical] != 0) {
            return {atom_value_plane_validation_code_v1::
                        duplicate_canonical_index,
                    index};
        }
        if (plane.ownership == atom_value_ownership_v1::logical_primary
            && canonical != index) {
            return {atom_value_plane_validation_code_v1::
                        nonidentity_logical_primary,
                    index};
        }
        canonical_marks[canonical] = 1;
    }

    result = detail::validate_compact_array_v1(
        plane.dirty_local_indices, true,
        atom_value_plane_validation_code_v1::inconsistent_dirty_pointer,
        atom_value_plane_validation_code_v1::invalid_dirty_bytes);
    if (!result.valid()) return result;
    std::uint64_t previous = 0;
    for (std::uint64_t index = 0;
         index < plane.dirty_local_indices.index_count; ++index) {
        const auto local = read_compact_edge_index_v1(
            plane.dirty_local_indices.indices, index,
            plane.dirty_local_indices.width);
        if (local >= plane.element_count) {
            return {atom_value_plane_validation_code_v1::
                        dirty_index_out_of_range,
                    index};
        }
        if (index != 0 && local <= previous) {
            return {atom_value_plane_validation_code_v1::
                        unordered_or_duplicate_dirty_index,
                    index};
        }
        previous = local;
    }
    return {atom_value_plane_validation_code_v1::valid,
            plane.element_count};
}

} // namespace cellshard::compiler::atom
