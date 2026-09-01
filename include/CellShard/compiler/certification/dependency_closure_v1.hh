#pragma once

#include <CellShard/compiler/atom/common_atom_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t dependency_closure_contract_version_v1 = 1;

struct authoritative_generation_v1 {
    atom::atom_persistent_identity_v1 dependency_identity{};
    atom::atom_persistent_identity_v1 generation_namespace{};
    std::uint64_t generation = 0;
    atom::atom_dependency_generation_kind_v1 generation_kind =
        atom::atom_dependency_generation_kind_v1::structure;
    std::uint32_t reserved = 0;
};

enum class dependency_closure_validation_code_v1 : std::uint32_t {
    valid = 0,
    empty_authority,
    missing_authority,
    invalid_authority_identity,
    invalid_generation_namespace,
    missing_generation,
    invalid_generation_kind,
    nonzero_reserved,
    unordered_or_duplicate_authority,
    invalid_atom_dependencies,
    dependency_missing_from_authority,
    observed_generation_stale,
};

struct dependency_closure_validation_v1 {
    dependency_closure_validation_code_v1 code =
        dependency_closure_validation_code_v1::valid;
    std::uint64_t atom_index = 0;
    std::uint64_t dependency_index = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == dependency_closure_validation_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<authoritative_generation_v1>::value);
static_assert(std::is_trivially_copyable<authoritative_generation_v1>::value);

[[nodiscard]] constexpr bool authoritative_generation_less_v1(
    const authoritative_generation_v1 &lhs,
    const authoritative_generation_v1 &rhs) noexcept {
    return atom::atom_persistent_identity_less_v1(
               lhs.dependency_identity, rhs.dependency_identity)
        || (lhs.dependency_identity == rhs.dependency_identity
            && (static_cast<std::uint32_t>(lhs.generation_kind)
                    < static_cast<std::uint32_t>(rhs.generation_kind)
                || (lhs.generation_kind == rhs.generation_kind
                    && atom::atom_persistent_identity_less_v1(
                        lhs.generation_namespace,
                        rhs.generation_namespace))));
}

[[nodiscard]] constexpr bool generation_keys_equal_v1(
    const authoritative_generation_v1 &authority,
    const atom::atom_dependency_requirement_v1 &dependency) noexcept {
    return authority.dependency_identity == dependency.dependency_identity
        && authority.generation_namespace == dependency.generation_namespace
        && authority.generation_kind == dependency.generation_kind;
}

// The authority table is complete and sorted. Every declared dependency is
// resolved by bounded binary search and its observed generation is checked
// against live authority: O(R + D log R), O(1) storage.
[[nodiscard]] inline dependency_closure_validation_v1
validate_generation_dependency_closure_v1(
    const atom::common_atom_view_v1 *atoms,
    std::uint64_t atom_count,
    const authoritative_generation_v1 *authority,
    std::uint64_t authority_count) noexcept {
    if (authority_count == 0) {
        return {dependency_closure_validation_code_v1::empty_authority};
    }
    if (authority == nullptr) {
        return {dependency_closure_validation_code_v1::missing_authority};
    }
    for (std::uint64_t index = 0; index < authority_count; ++index) {
        const auto &record = authority[index];
        if (!atom::validate_atom_persistent_identity_v1(
                 record.dependency_identity)
                 .valid()) {
            return {dependency_closure_validation_code_v1::
                        invalid_authority_identity,
                    0,
                    index};
        }
        if (!atom::validate_atom_persistent_identity_v1(
                 record.generation_namespace)
                 .valid()) {
            return {dependency_closure_validation_code_v1::
                        invalid_generation_namespace,
                    0,
                    index};
        }
        if (record.generation == 0) {
            return {dependency_closure_validation_code_v1::missing_generation,
                    0,
                    index};
        }
        if (!atom::valid_atom_dependency_generation_kind_v1(
                record.generation_kind)) {
            return {dependency_closure_validation_code_v1::
                        invalid_generation_kind,
                    0,
                    index};
        }
        if (record.reserved != 0) {
            return {dependency_closure_validation_code_v1::nonzero_reserved,
                    0,
                    index};
        }
        if (index != 0
            && !authoritative_generation_less_v1(
                authority[index - 1], record)) {
            return {dependency_closure_validation_code_v1::
                        unordered_or_duplicate_authority,
                    0,
                    index};
        }
    }
    if (atom_count != 0 && atoms == nullptr) {
        return {dependency_closure_validation_code_v1::
                    invalid_atom_dependencies};
    }
    for (std::uint64_t atom_index = 0; atom_index < atom_count; ++atom_index) {
        const auto plane_result =
            atom::validate_atom_dependency_invalidation_plane_v1(
                atoms[atom_index].dependencies);
        if (!plane_result.valid()) {
            return {dependency_closure_validation_code_v1::
                        invalid_atom_dependencies,
                    atom_index,
                    plane_result.index,
                    static_cast<std::uint32_t>(plane_result.code)};
        }
        const auto &plane = atoms[atom_index].dependencies;
        for (std::uint64_t dependency_index = 0;
             dependency_index < plane.dependency_count;
             ++dependency_index) {
            const auto &dependency = plane.dependencies[dependency_index];
            std::uint64_t begin = 0;
            std::uint64_t end = authority_count;
            const authoritative_generation_v1 key{
                dependency.dependency_identity,
                dependency.generation_namespace,
                dependency.observed_generation,
                dependency.generation_kind,
                0};
            while (begin < end) {
                const auto middle = begin + (end - begin) / 2;
                if (authoritative_generation_less_v1(authority[middle], key)) {
                    begin = middle + 1;
                } else {
                    end = middle;
                }
            }
            if (begin == authority_count
                || !generation_keys_equal_v1(authority[begin], dependency)) {
                return {dependency_closure_validation_code_v1::
                            dependency_missing_from_authority,
                        atom_index,
                        dependency_index};
            }
            if (authority[begin].generation
                != dependency.observed_generation) {
                return {dependency_closure_validation_code_v1::
                            observed_generation_stale,
                        atom_index,
                        dependency_index};
            }
        }
    }
    return {dependency_closure_validation_code_v1::valid,
            atom_count,
            authority_count,
            0};
}

} // namespace cellshard::compiler::certification
