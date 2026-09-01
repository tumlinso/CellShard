#pragma once

#include <CellShard/compiler/certification/contribution_owner_v1.hh>

#include <cstdint>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t residual_coverage_contract_version_v1 = 1;

enum class residual_coverage_build_code_v1 : std::uint32_t {
    built = 0,
    missing_canonical,
    missing_owners,
    invalid_owner_identity,
    invalid_member_kind,
    unordered_or_duplicate_canonical,
    owner_key_mismatch,
    unordered_or_duplicate_owner,
    owner_not_canonical,
    missing_output,
    insufficient_output,
};

struct residual_coverage_build_result_v1 {
    residual_coverage_build_code_v1 code =
        residual_coverage_build_code_v1::built;
    std::uint64_t residual_count = 0;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool built() const noexcept {
        return code == residual_coverage_build_code_v1::built;
    }
};

// Owners contain exactly one semantic key (domain or relation). Two linear
// merge passes first determine capacity, then emit every unowned canonical u64
// identity. The explicit residual is a fallback path, not silent data loss.
[[nodiscard]] inline residual_coverage_build_result_v1
build_exact_residual_coverage_v1(
    const std::uint64_t *canonical_ids,
    std::uint64_t canonical_count,
    atom::atom_persistent_identity_v1 owner_identity,
    certification_member_kind_v1 kind,
    const exact_contribution_owner_v1 *owners,
    std::uint64_t owner_count,
    std::uint64_t *residual_ids,
    std::uint64_t residual_capacity) noexcept {
    if (canonical_count != 0 && canonical_ids == nullptr) {
        return {residual_coverage_build_code_v1::missing_canonical};
    }
    if (owner_count != 0 && owners == nullptr) {
        return {residual_coverage_build_code_v1::missing_owners};
    }
    if (!atom::validate_atom_persistent_identity_v1(owner_identity).valid()) {
        return {residual_coverage_build_code_v1::invalid_owner_identity};
    }
    if (kind != certification_member_kind_v1::entity
        && kind != certification_member_kind_v1::relation_edge) {
        return {residual_coverage_build_code_v1::invalid_member_kind};
    }
    for (std::uint64_t index = 0; index < canonical_count; ++index) {
        if (canonical_ids[index] == 0
            || (index != 0 && canonical_ids[index - 1] >= canonical_ids[index])) {
            return {residual_coverage_build_code_v1::
                        unordered_or_duplicate_canonical,
                    0,
                    index};
        }
    }
    for (std::uint64_t index = 0; index < owner_count; ++index) {
        const auto &owner = owners[index];
        if (owner.owner_identity != owner_identity || owner.kind != kind) {
            return {residual_coverage_build_code_v1::owner_key_mismatch,
                    0,
                    index};
        }
        if (index != 0
            && owners[index - 1].global_identity >= owner.global_identity) {
            return {residual_coverage_build_code_v1::
                        unordered_or_duplicate_owner,
                    0,
                    index};
        }
    }

    std::uint64_t residual_count = 0;
    std::uint64_t owner_index = 0;
    for (std::uint64_t canonical_index = 0;
         canonical_index < canonical_count;
         ++canonical_index) {
        const auto canonical = canonical_ids[canonical_index];
        if (owner_index < owner_count
            && owners[owner_index].global_identity < canonical) {
            return {residual_coverage_build_code_v1::owner_not_canonical,
                    residual_count,
                    owner_index};
        }
        if (owner_index < owner_count
            && owners[owner_index].global_identity == canonical) {
            ++owner_index;
        } else {
            ++residual_count;
        }
    }
    if (owner_index != owner_count) {
        return {residual_coverage_build_code_v1::owner_not_canonical,
                residual_count,
                owner_index};
    }
    if (residual_count != 0 && residual_ids == nullptr) {
        return {residual_coverage_build_code_v1::missing_output,
                residual_count};
    }
    if (residual_capacity < residual_count) {
        return {residual_coverage_build_code_v1::insufficient_output,
                residual_count};
    }

    std::uint64_t output = 0;
    owner_index = 0;
    for (std::uint64_t canonical_index = 0;
         canonical_index < canonical_count;
         ++canonical_index) {
        const auto canonical = canonical_ids[canonical_index];
        if (owner_index < owner_count
            && owners[owner_index].global_identity == canonical) {
            ++owner_index;
        } else {
            residual_ids[output++] = canonical;
        }
    }
    return {residual_coverage_build_code_v1::built, output, canonical_count};
}

} // namespace cellshard::compiler::certification
