#pragma once

#include <CellShard/compiler/atom/common_atom_v1.hh>
#include <CellShard/compiler/atom/physical_view_plane_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t physical_replica_contract_version_v1 = 1;

struct physical_replica_binding_v1 {
    const atom::atom_physical_view_plane_v1 *physical_view = nullptr;
    atom::atom_replica_id_v1 replica{};
    atom::atom_content_id_v1 content{};
};

enum class physical_replica_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_atom_identity,
    empty_replicas,
    missing_replicas,
    missing_physical_view,
    invalid_replica_identity,
    unordered_or_duplicate_replica,
    content_mismatch,
    invalid_physical_view,
    semantic_family_mismatch,
    materialization_mismatch,
};

struct physical_replica_validation_v1 {
    physical_replica_validation_code_v1 code =
        physical_replica_validation_code_v1::valid;
    std::uint64_t index = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == physical_replica_validation_code_v1::valid;
    }
};

static_assert(offsetof(physical_replica_binding_v1, physical_view) == 0);
static_assert(std::is_standard_layout<physical_replica_binding_v1>::value);
static_assert(std::is_trivially_copyable<physical_replica_binding_v1>::value);

// Replica identity stays independent of semantic family, content, and
// materialization. Validation is a linear metadata pass and never treats a
// second physical copy as a second exact contribution.
[[nodiscard]] inline physical_replica_validation_v1
validate_physical_replicas_v1(
    const atom::common_atom_view_v1 &common,
    const physical_replica_binding_v1 *replicas,
    std::uint64_t replica_count) noexcept {
    const auto identity_result =
        atom::validate_atom_identity_binding_v1(common.identities);
    if (!identity_result.valid()) {
        return {physical_replica_validation_code_v1::invalid_atom_identity,
                0,
                static_cast<std::uint32_t>(identity_result.code)};
    }
    if (replica_count == 0) {
        return {physical_replica_validation_code_v1::empty_replicas};
    }
    if (replicas == nullptr) {
        return {physical_replica_validation_code_v1::missing_replicas};
    }
    for (std::uint64_t index = 0; index < replica_count; ++index) {
        const auto &replica = replicas[index];
        if (replica.physical_view == nullptr) {
            return {physical_replica_validation_code_v1::
                        missing_physical_view,
                    index};
        }
        if (!atom::validate_atom_persistent_identity_v1(
                 replica.replica.persistent)
                 .valid()) {
            return {physical_replica_validation_code_v1::
                        invalid_replica_identity,
                    index};
        }
        if (index != 0
            && !atom::atom_persistent_identity_less_v1(
                replicas[index - 1].replica.persistent,
                replica.replica.persistent)) {
            return {physical_replica_validation_code_v1::
                        unordered_or_duplicate_replica,
                    index};
        }
        if (!(replica.content == common.identities.content)) {
            return {physical_replica_validation_code_v1::content_mismatch,
                    index};
        }
        const auto view_result =
            atom::validate_atom_physical_view_plane_v1(*replica.physical_view);
        if (!view_result.valid()) {
            return {physical_replica_validation_code_v1::invalid_physical_view,
                    index,
                    static_cast<std::uint32_t>(view_result.code)};
        }
        if (!(replica.physical_view->semantic_family
              == common.identities.semantic_family)) {
            return {physical_replica_validation_code_v1::
                        semantic_family_mismatch,
                    index};
        }
        if (!(replica.physical_view->materialization
              == common.identities.materialization)) {
            return {physical_replica_validation_code_v1::
                        materialization_mismatch,
                    index};
        }
    }
    return {physical_replica_validation_code_v1::valid, replica_count, 0};
}

} // namespace cellshard::compiler::certification
