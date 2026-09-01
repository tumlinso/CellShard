#pragma once

#include <CellShard/compiler/atom/partial_result_plane_v1.hh>

#include <cstdint>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t partial_result_compatibility_contract_version_v1 =
    1;

enum class partial_result_compatibility_code_v1 : std::uint32_t {
    compatible = 0,
    empty_set,
    missing_partials,
    invalid_partial,
    unordered_or_duplicate_owner,
    reconstruction_algebra_mismatch,
    numerical_policy_mismatch,
    subject_space_mismatch,
    persistent_order_mismatch,
    numeric_mismatch,
    subject_kind_mismatch,
    canonical_count_mismatch,
    dependency_count_mismatch,
    dependency_mismatch,
};

struct partial_result_compatibility_v1 {
    partial_result_compatibility_code_v1 code =
        partial_result_compatibility_code_v1::compatible;
    std::uint64_t partial_index = 0;
    std::uint64_t dependency_index = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool compatible() const noexcept {
        return code == partial_result_compatibility_code_v1::compatible;
    }
};

[[nodiscard]] constexpr bool same_partial_numeric_v1(
    atom::atom_port_numeric_v1 lhs,
    atom::atom_port_numeric_v1 rhs) noexcept {
    return lhs.storage_type == rhs.storage_type
        && lhs.logical_type == rhs.logical_type
        && lhs.accumulation_type == rhs.accumulation_type;
}

// Each partial is independently validated against exact coverage before its
// reconstruction contract is compared. Shared algebra, numerical policy,
// logical space/order/type and dependency generations are mandatory; physical
// layout identity and value pointers may differ.
[[nodiscard]] inline partial_result_compatibility_v1
validate_partial_result_algebra_compatibility_v1(
    const atom::atom_partial_result_plane_v1 *partials,
    std::uint64_t partial_count,
    std::uint32_t coverage_source_validation,
    std::uint8_t *canonical_marks,
    std::uint64_t mark_capacity) noexcept {
    if (partial_count == 0) {
        return {partial_result_compatibility_code_v1::empty_set};
    }
    if (partials == nullptr) {
        return {partial_result_compatibility_code_v1::missing_partials};
    }
    const auto &reference = partials[0];
    for (std::uint64_t index = 0; index < partial_count; ++index) {
        const auto &partial = partials[index];
        const auto validation = atom::validate_atom_partial_result_plane_v1(
            partial,
            coverage_source_validation,
            canonical_marks,
            mark_capacity);
        if (!validation.valid()) {
            return {partial_result_compatibility_code_v1::invalid_partial,
                    index,
                    validation.index,
                    static_cast<std::uint32_t>(validation.code)};
        }
        if (index != 0
            && !atom::atom_persistent_identity_less_v1(
                partials[index - 1].contribution_owner_identity,
                partial.contribution_owner_identity)) {
            return {partial_result_compatibility_code_v1::
                        unordered_or_duplicate_owner,
                    index};
        }
        if (partial.reconstruction_algebra_identity
            != reference.reconstruction_algebra_identity) {
            return {partial_result_compatibility_code_v1::
                        reconstruction_algebra_mismatch,
                    index};
        }
        if (partial.numerical_policy_identity
            != reference.numerical_policy_identity) {
            return {partial_result_compatibility_code_v1::
                        numerical_policy_mismatch,
                    index};
        }
        if (partial.partial_layout.subject_space_identity
            != reference.partial_layout.subject_space_identity) {
            return {partial_result_compatibility_code_v1::
                        subject_space_mismatch,
                    index};
        }
        if (partial.partial_layout.persistent_order_identity
            != reference.partial_layout.persistent_order_identity) {
            return {partial_result_compatibility_code_v1::
                        persistent_order_mismatch,
                    index};
        }
        if (!same_partial_numeric_v1(partial.partial_layout.numeric,
                                     reference.partial_layout.numeric)) {
            return {partial_result_compatibility_code_v1::numeric_mismatch,
                    index};
        }
        if (partial.partial_layout.subject_kind
            != reference.partial_layout.subject_kind) {
            return {partial_result_compatibility_code_v1::subject_kind_mismatch,
                    index};
        }
        if (partial.partial_layout.canonical_element_count
            != reference.partial_layout.canonical_element_count) {
            return {partial_result_compatibility_code_v1::
                        canonical_count_mismatch,
                    index};
        }
        if (partial.dependency_count != reference.dependency_count) {
            return {partial_result_compatibility_code_v1::
                        dependency_count_mismatch,
                    index};
        }
        for (std::uint64_t dependency_index = 0;
             dependency_index < partial.dependency_count;
             ++dependency_index) {
            const auto &actual = partial.dependencies[dependency_index];
            const auto &expected = reference.dependencies[dependency_index];
            if (actual.dependency_identity != expected.dependency_identity
                || actual.generation_kind != expected.generation_kind
                || actual.generation != expected.generation) {
                return {partial_result_compatibility_code_v1::
                            dependency_mismatch,
                        index,
                        dependency_index};
            }
        }
    }
    return {partial_result_compatibility_code_v1::compatible,
            partial_count};
}

} // namespace cellshard::compiler::certification
