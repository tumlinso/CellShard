#pragma once

#include <CellShard/compiler/certification/entity_coverage_v1.hh>
#include <CellShard/compiler/certification/relation_edge_coverage_v1.hh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t duplicate_detection_contract_version_v1 = 1;

struct atom_coverage_bundle_v1 {
    const atom_entity_coverage_claim_v1 *entity_claims = nullptr;
    const atom_relation_edge_coverage_claim_v1 *edge_claims = nullptr;
    std::uint64_t entity_claim_count = 0;
    std::uint64_t edge_claim_count = 0;
};

enum class certification_member_kind_v1 : std::uint8_t {
    entity = 1,
    relation_edge = 2,
};

struct certification_member_key_v1 {
    atom::atom_persistent_identity_v1 owner_identity{};
    std::uint64_t global_identity = 0;
    std::uint64_t atom_index = 0;
    certification_member_kind_v1 kind =
        certification_member_kind_v1::entity;
    std::uint8_t reserved[7]{};
};

enum class duplicate_detection_code_v1 : std::uint32_t {
    unique = 0,
    inconsistent_bundle_pointer,
    member_count_overflow,
    missing_workspace,
    insufficient_workspace,
    duplicate_entity,
    duplicate_relation_edge,
};

struct duplicate_detection_result_v1 {
    duplicate_detection_code_v1 code = duplicate_detection_code_v1::unique;
    std::uint64_t required_key_capacity = 0;
    std::uint64_t first_atom_index = no_failed_certification_index_v1;
    std::uint64_t duplicate_atom_index = no_failed_certification_index_v1;
    std::uint64_t global_identity = 0;

    [[nodiscard]] constexpr bool unique() const noexcept {
        return code == duplicate_detection_code_v1::unique;
    }
};

static_assert(std::is_standard_layout<atom_coverage_bundle_v1>::value);
static_assert(std::is_trivially_copyable<atom_coverage_bundle_v1>::value);
static_assert(std::is_standard_layout<certification_member_key_v1>::value);
static_assert(std::is_trivially_copyable<certification_member_key_v1>::value);

[[nodiscard]] constexpr bool certification_member_key_less_v1(
    const certification_member_key_v1 &lhs,
    const certification_member_key_v1 &rhs) noexcept {
    if (lhs.kind != rhs.kind) {
        return static_cast<std::uint8_t>(lhs.kind)
            < static_cast<std::uint8_t>(rhs.kind);
    }
    if (lhs.owner_identity != rhs.owner_identity) {
        return atom::atom_persistent_identity_less_v1(
            lhs.owner_identity, rhs.owner_identity);
    }
    if (lhs.global_identity != rhs.global_identity) {
        return lhs.global_identity < rhs.global_identity;
    }
    return lhs.atom_index < rhs.atom_index;
}

// Uses exactly one caller-owned key per claimed member and an in-place sort:
// O(N log N) time, O(N) declared workspace, and no atlas-by-atlas comparison.
[[nodiscard]] inline duplicate_detection_result_v1
detect_duplicate_coverage_v1(
    const atom_coverage_bundle_v1 *atoms,
    std::uint64_t atom_count,
    certification_member_key_v1 *workspace,
    std::uint64_t key_capacity) noexcept {
    if (atom_count != 0 && atoms == nullptr) {
        return {duplicate_detection_code_v1::inconsistent_bundle_pointer};
    }
    std::uint64_t required = 0;
    for (std::uint64_t atom_index = 0; atom_index < atom_count; ++atom_index) {
        const auto &bundle = atoms[atom_index];
        if ((bundle.entity_claim_count != 0 && bundle.entity_claims == nullptr)
            || (bundle.edge_claim_count != 0
                && bundle.edge_claims == nullptr)) {
            return {duplicate_detection_code_v1::
                        inconsistent_bundle_pointer};
        }
        for (std::uint64_t index = 0;
             index < bundle.entity_claim_count;
             ++index) {
            const auto count = bundle.entity_claims[index].entity_count;
            if (count > std::numeric_limits<std::uint64_t>::max() - required) {
                return {duplicate_detection_code_v1::member_count_overflow};
            }
            required += count;
        }
        for (std::uint64_t index = 0; index < bundle.edge_claim_count; ++index) {
            const auto count = bundle.edge_claims[index].edge_count;
            if (count > std::numeric_limits<std::uint64_t>::max() - required) {
                return {duplicate_detection_code_v1::member_count_overflow};
            }
            required += count;
        }
    }
    if (required != 0 && workspace == nullptr) {
        return {duplicate_detection_code_v1::missing_workspace, required};
    }
    if (key_capacity < required) {
        return {duplicate_detection_code_v1::insufficient_workspace,
                required};
    }
    if (required == 0) {
        return {duplicate_detection_code_v1::unique, 0};
    }

    std::uint64_t output = 0;
    for (std::uint64_t atom_index = 0; atom_index < atom_count; ++atom_index) {
        const auto &bundle = atoms[atom_index];
        for (std::uint64_t claim_index = 0;
             claim_index < bundle.entity_claim_count;
             ++claim_index) {
            const auto &claim = bundle.entity_claims[claim_index];
            for (std::uint64_t index = 0; index < claim.entity_count; ++index) {
                workspace[output++] = {claim.domain_identity,
                                       claim.global_entity_ids[index],
                                       atom_index,
                                       certification_member_kind_v1::entity,
                                       {}};
            }
        }
        for (std::uint64_t claim_index = 0;
             claim_index < bundle.edge_claim_count;
             ++claim_index) {
            const auto &claim = bundle.edge_claims[claim_index];
            for (std::uint64_t index = 0; index < claim.edge_count; ++index) {
                workspace[output++] = {
                    claim.relation_identity,
                    claim.global_edge_ids[index],
                    atom_index,
                    certification_member_kind_v1::relation_edge,
                    {}};
            }
        }
    }
    std::sort(workspace,
              workspace + required,
              certification_member_key_less_v1);
    for (std::uint64_t index = 1; index < required; ++index) {
        const auto &previous = workspace[index - 1];
        const auto &current = workspace[index];
        if (previous.kind == current.kind
            && previous.owner_identity == current.owner_identity
            && previous.global_identity == current.global_identity) {
            return {current.kind == certification_member_kind_v1::entity
                        ? duplicate_detection_code_v1::duplicate_entity
                        : duplicate_detection_code_v1::duplicate_relation_edge,
                    required,
                    previous.atom_index,
                    current.atom_index,
                    current.global_identity};
        }
    }
    return {duplicate_detection_code_v1::unique, required};
}

} // namespace cellshard::compiler::certification
