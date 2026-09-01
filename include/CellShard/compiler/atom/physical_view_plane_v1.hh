#pragma once

#include <CellShard/compiler/atom/identity_classes_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_physical_view_plane_schema_version_v1 = 1;

enum class atom_physical_target_kind_v1 : std::uint32_t {
    target_neutral = 1,
    target_specific = 2,
};

// Axis identity and order are semantic. Logical and physical extents may differ
// because padding, blocking, or compression belongs to this materialization.
struct atom_physical_extent_v1 {
    atom_persistent_identity_v1 axis_identity{};
    std::uint64_t logical_extent = 0;
    std::uint64_t physical_extent = 0;
    std::uint64_t byte_stride = 0;
};

// A view describes one physical materialization without redefining its semantic
// atom family. The payload may be target-neutral portable bytes or bytes for one
// explicitly named target ABI. Runtime residency and placement are not encoded.
struct atom_physical_view_plane_v1 {
    const void *payload = nullptr;
    std::uint64_t payload_bytes = 0;
    const atom_physical_extent_v1 *extents = nullptr;
    std::uint64_t extent_count = 0;
    atom_semantic_family_id_v1 semantic_family{};
    atom_materialization_id_v1 materialization{};
    atom_persistent_identity_v1 physical_view_identity{};
    atom_persistent_identity_v1 encoding_identity{};
    atom_persistent_identity_v1 persistent_order_identity{};
    atom_persistent_identity_v1 projection_abi_identity{};
    atom_persistent_identity_v1 target_identity{};
    std::uint64_t materialization_generation = 0;
    std::uint32_t payload_alignment = 0;
    atom_physical_target_kind_v1 target_kind =
        atom_physical_target_kind_v1::target_neutral;
};

enum class atom_physical_view_validation_code_v1 : std::uint32_t {
    valid = 0,
    missing_payload,
    empty_payload,
    invalid_payload_alignment,
    misaligned_payload,
    missing_extents,
    invalid_axis_identity,
    empty_logical_extent,
    physical_extent_smaller_than_logical,
    invalid_byte_stride,
    invalid_semantic_family,
    invalid_materialization,
    invalid_physical_view_identity,
    invalid_encoding_identity,
    invalid_persistent_order,
    invalid_projection_abi,
    invalid_target_kind,
    unexpected_target_identity,
    missing_target_identity,
    missing_materialization_generation,
};

struct atom_physical_view_validation_v1 {
    atom_physical_view_validation_code_v1 code =
        atom_physical_view_validation_code_v1::valid;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_physical_view_validation_code_v1::valid;
    }
};

static_assert(offsetof(atom_physical_view_plane_v1, payload) == 0,
              "physical views must remain pointer-first");
static_assert(std::is_standard_layout<atom_physical_extent_v1>::value);
static_assert(std::is_trivially_copyable<atom_physical_extent_v1>::value);
static_assert(std::is_standard_layout<atom_physical_view_plane_v1>::value);
static_assert(std::is_trivially_copyable<atom_physical_view_plane_v1>::value);

[[nodiscard]] constexpr bool valid_atom_physical_target_kind_v1(
    atom_physical_target_kind_v1 kind) noexcept {
    return kind == atom_physical_target_kind_v1::target_neutral
        || kind == atom_physical_target_kind_v1::target_specific;
}

// Validation is O(extent_count), O(1) storage, and allocation-free. It checks
// representation metadata only; it never parses or canonicalizes payload bytes.
[[nodiscard]] inline atom_physical_view_validation_v1
validate_atom_physical_view_plane_v1(
    const atom_physical_view_plane_v1 &plane) noexcept {
    if (plane.payload == nullptr) {
        return {atom_physical_view_validation_code_v1::missing_payload, 0};
    }
    if (plane.payload_bytes == 0) {
        return {atom_physical_view_validation_code_v1::empty_payload, 0};
    }
    if (plane.payload_alignment == 0
        || (plane.payload_alignment & (plane.payload_alignment - 1)) != 0) {
        return {atom_physical_view_validation_code_v1::
                    invalid_payload_alignment,
                0};
    }
    if (reinterpret_cast<std::uintptr_t>(plane.payload)
        % plane.payload_alignment != 0) {
        return {atom_physical_view_validation_code_v1::misaligned_payload, 0};
    }
    if (plane.extent_count == 0 || plane.extents == nullptr) {
        return {atom_physical_view_validation_code_v1::missing_extents, 0};
    }
    for (std::uint64_t index = 0; index < plane.extent_count; ++index) {
        const auto &extent = plane.extents[index];
        if (!validate_atom_persistent_identity_v1(extent.axis_identity).valid()) {
            return {atom_physical_view_validation_code_v1::
                        invalid_axis_identity,
                    index};
        }
        if (extent.logical_extent == 0) {
            return {atom_physical_view_validation_code_v1::empty_logical_extent,
                    index};
        }
        if (extent.physical_extent < extent.logical_extent) {
            return {atom_physical_view_validation_code_v1::
                        physical_extent_smaller_than_logical,
                    index};
        }
        if (extent.byte_stride == 0) {
            return {atom_physical_view_validation_code_v1::invalid_byte_stride,
                    index};
        }
    }
    if (!validate_atom_persistent_identity_v1(
             plane.semantic_family.persistent)
             .valid()) {
        return {atom_physical_view_validation_code_v1::
                    invalid_semantic_family,
                0};
    }
    if (!validate_atom_persistent_identity_v1(
             plane.materialization.persistent)
             .valid()) {
        return {atom_physical_view_validation_code_v1::
                    invalid_materialization,
                0};
    }
#define CELLSHARD_ATOM_PHYSICAL_CHECK_ID(field, code) \
    if (!validate_atom_persistent_identity_v1(plane.field).valid()) { \
        return {atom_physical_view_validation_code_v1::code, 0}; \
    }
    CELLSHARD_ATOM_PHYSICAL_CHECK_ID(physical_view_identity,
                                     invalid_physical_view_identity)
    CELLSHARD_ATOM_PHYSICAL_CHECK_ID(encoding_identity,
                                     invalid_encoding_identity)
    CELLSHARD_ATOM_PHYSICAL_CHECK_ID(persistent_order_identity,
                                     invalid_persistent_order)
    CELLSHARD_ATOM_PHYSICAL_CHECK_ID(projection_abi_identity,
                                     invalid_projection_abi)
#undef CELLSHARD_ATOM_PHYSICAL_CHECK_ID
    if (!valid_atom_physical_target_kind_v1(plane.target_kind)) {
        return {atom_physical_view_validation_code_v1::invalid_target_kind, 0};
    }
    const bool target_valid =
        validate_atom_persistent_identity_v1(plane.target_identity).valid();
    if (plane.target_kind == atom_physical_target_kind_v1::target_neutral
        && target_valid) {
        return {atom_physical_view_validation_code_v1::
                    unexpected_target_identity,
                0};
    }
    if (plane.target_kind == atom_physical_target_kind_v1::target_specific
        && !target_valid) {
        return {atom_physical_view_validation_code_v1::missing_target_identity,
                0};
    }
    if (plane.materialization_generation == 0) {
        return {atom_physical_view_validation_code_v1::
                    missing_materialization_generation,
                0};
    }
    return {atom_physical_view_validation_code_v1::valid,
            plane.extent_count};
}

} // namespace cellshard::compiler::atom
