#pragma once

#include <CellShard/compiler/atom/persistent_identity_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_plane_directory_contract_version_v1 = 1;

enum class atom_plane_representation_role_v1 : std::uint32_t {
    primary = 1,
    alternate_physical_mirror = 2,
};

// The descriptor payload is source-linked and schema-qualified. The directory
// never owns or interprets its bytes; A10+ define individual plane schemas.
struct atom_plane_descriptor_v1 {
    atom_persistent_identity_v1 plane_kind{};
    atom_persistent_identity_v1 plane_identity{};
    atom_persistent_identity_v1 descriptor_schema{};
    atom_persistent_identity_v1 primary_plane_identity{};
    const void *descriptor = nullptr;
    std::uint64_t descriptor_bytes = 0;
    std::uint32_t descriptor_alignment = 0;
    atom_plane_representation_role_v1 representation_role =
        atom_plane_representation_role_v1::primary;
};

struct atom_plane_directory_view_v1 {
    const atom_plane_descriptor_v1 *planes = nullptr;
    std::uint64_t plane_count = 0;
};

enum class atom_plane_directory_validation_code_v1 : std::uint32_t {
    valid = 0,
    empty_directory,
    missing_planes,
    invalid_plane_kind,
    invalid_plane_identity,
    invalid_descriptor_schema,
    missing_descriptor,
    empty_descriptor,
    invalid_descriptor_alignment,
    misaligned_descriptor,
    invalid_representation_role,
    unordered_or_duplicate_plane,
    duplicate_primary_plane,
    missing_primary_plane,
    unexpected_primary_reference,
    missing_primary_reference,
    mirror_primary_mismatch,
    mirror_self_reference,
};

struct atom_plane_directory_validation_v1 {
    atom_plane_directory_validation_code_v1 code =
        atom_plane_directory_validation_code_v1::valid;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_plane_directory_validation_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<atom_plane_descriptor_v1>::value);
static_assert(std::is_trivially_copyable<atom_plane_descriptor_v1>::value);
static_assert(offsetof(atom_plane_directory_view_v1, planes) == 0,
              "plane directories must remain pointer-first");
static_assert(std::is_standard_layout<atom_plane_directory_view_v1>::value);
static_assert(std::is_trivially_copyable<atom_plane_directory_view_v1>::value);

[[nodiscard]] constexpr bool atom_plane_key_less_v1(
    const atom_plane_descriptor_v1 &lhs,
    const atom_plane_descriptor_v1 &rhs) noexcept {
    return atom_persistent_identity_less_v1(lhs.plane_kind, rhs.plane_kind)
        || (lhs.plane_kind == rhs.plane_kind
            && atom_persistent_identity_less_v1(
                lhs.plane_identity, rhs.plane_identity));
}

[[nodiscard]] constexpr bool valid_atom_plane_representation_role_v1(
    atom_plane_representation_role_v1 role) noexcept {
    return role == atom_plane_representation_role_v1::primary
        || role
            == atom_plane_representation_role_v1::alternate_physical_mirror;
}

// Entries are sorted by (plane kind, stable plane identity). Each kind has
// exactly one primary; every alternate names that primary explicitly. This is
// O(plane_count) time and O(1) storage with no hidden allocation.
[[nodiscard]] inline atom_plane_directory_validation_v1
validate_atom_plane_directory_v1(
    atom_plane_directory_view_v1 directory) noexcept {
    if (directory.plane_count == 0) {
        return {atom_plane_directory_validation_code_v1::empty_directory, 0};
    }
    if (directory.planes == nullptr) {
        return {atom_plane_directory_validation_code_v1::missing_planes, 0};
    }
    for (std::uint64_t index = 0; index < directory.plane_count; ++index) {
        const auto &plane = directory.planes[index];
        if (!validate_atom_persistent_identity_v1(plane.plane_kind).valid()) {
            return {atom_plane_directory_validation_code_v1::invalid_plane_kind,
                    index};
        }
        if (!validate_atom_persistent_identity_v1(plane.plane_identity).valid()) {
            return {
                atom_plane_directory_validation_code_v1::invalid_plane_identity,
                index};
        }
        if (!validate_atom_persistent_identity_v1(plane.descriptor_schema)
                 .valid()) {
            return {atom_plane_directory_validation_code_v1::
                        invalid_descriptor_schema,
                    index};
        }
        if (plane.descriptor == nullptr) {
            return {atom_plane_directory_validation_code_v1::missing_descriptor,
                    index};
        }
        if (plane.descriptor_bytes == 0) {
            return {atom_plane_directory_validation_code_v1::empty_descriptor,
                    index};
        }
        if (plane.descriptor_alignment == 0
            || (plane.descriptor_alignment & (plane.descriptor_alignment - 1))
                   != 0) {
            return {atom_plane_directory_validation_code_v1::
                        invalid_descriptor_alignment,
                    index};
        }
        if (reinterpret_cast<std::uintptr_t>(plane.descriptor)
            % plane.descriptor_alignment != 0) {
            return {
                atom_plane_directory_validation_code_v1::misaligned_descriptor,
                index};
        }
        if (!valid_atom_plane_representation_role_v1(
                plane.representation_role)) {
            return {atom_plane_directory_validation_code_v1::
                        invalid_representation_role,
                    index};
        }
        if (index != 0
            && !atom_plane_key_less_v1(directory.planes[index - 1], plane)) {
            return {atom_plane_directory_validation_code_v1::
                        unordered_or_duplicate_plane,
                    index};
        }
    }

    std::uint64_t group_begin = 0;
    while (group_begin < directory.plane_count) {
        std::uint64_t group_end = group_begin + 1;
        while (group_end < directory.plane_count
               && directory.planes[group_end].plane_kind
                   == directory.planes[group_begin].plane_kind) {
            ++group_end;
        }
        std::uint64_t primary_index = directory.plane_count;
        for (std::uint64_t index = group_begin; index < group_end; ++index) {
            const auto &plane = directory.planes[index];
            if (plane.representation_role
                == atom_plane_representation_role_v1::primary) {
                if (primary_index != directory.plane_count) {
                    return {atom_plane_directory_validation_code_v1::
                                duplicate_primary_plane,
                            index};
                }
                if (validate_atom_persistent_identity_v1(
                        plane.primary_plane_identity)
                        .valid()) {
                    return {atom_plane_directory_validation_code_v1::
                                unexpected_primary_reference,
                            index};
                }
                primary_index = index;
            }
        }
        if (primary_index == directory.plane_count) {
            return {atom_plane_directory_validation_code_v1::
                        missing_primary_plane,
                    group_begin};
        }
        const auto primary_identity =
            directory.planes[primary_index].plane_identity;
        for (std::uint64_t index = group_begin; index < group_end; ++index) {
            const auto &plane = directory.planes[index];
            if (plane.representation_role
                != atom_plane_representation_role_v1::
                       alternate_physical_mirror) {
                continue;
            }
            if (!validate_atom_persistent_identity_v1(
                    plane.primary_plane_identity)
                    .valid()) {
                return {atom_plane_directory_validation_code_v1::
                            missing_primary_reference,
                        index};
            }
            if (plane.primary_plane_identity == plane.plane_identity) {
                return {atom_plane_directory_validation_code_v1::
                            mirror_self_reference,
                        index};
            }
            if (plane.primary_plane_identity != primary_identity) {
                return {atom_plane_directory_validation_code_v1::
                            mirror_primary_mismatch,
                        index};
            }
        }
        group_begin = group_end;
    }
    return {atom_plane_directory_validation_code_v1::valid,
            directory.plane_count};
}

} // namespace cellshard::compiler::atom
