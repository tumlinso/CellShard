#pragma once

#include <CellShard/compiler/atom/relation_edge_spine_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t relation_edge_coverage_contract_version_v1 = 1;

struct atom_relation_edge_coverage_claim_v1 {
    const std::uint64_t *global_edge_ids = nullptr;
    std::uint64_t edge_count = 0;
    atom::atom_persistent_identity_v1 relation_identity{};
    std::uint64_t structure_epoch = 0;
};

enum class relation_edge_coverage_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_canonical_spine,
    empty_claim,
    missing_claim_edges,
    relation_mismatch,
    epoch_mismatch,
    zero_global_edge_identity,
    unordered_or_duplicate_claim_edge,
    edge_not_in_canonical_relation,
};

struct relation_edge_coverage_validation_v1 {
    relation_edge_coverage_validation_code_v1 code =
        relation_edge_coverage_validation_code_v1::valid;
    std::uint64_t index = 0;
    std::uint64_t canonical_index = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == relation_edge_coverage_validation_code_v1::valid;
    }
};

static_assert(offsetof(atom_relation_edge_coverage_claim_v1,
                       global_edge_ids)
              == 0);
static_assert(
    std::is_standard_layout<atom_relation_edge_coverage_claim_v1>::value);
static_assert(
    std::is_trivially_copyable<atom_relation_edge_coverage_claim_v1>::value);

// Exact ascending global edge identities permit a single allocation-free merge
// scan. Local edge ordinals and physical representation never establish edge
// membership.
[[nodiscard]] inline relation_edge_coverage_validation_v1
validate_exact_relation_edge_coverage_v1(
    atom::relation_edge_spine_view_v1 canonical,
    atom_relation_edge_coverage_claim_v1 claim) noexcept {
    const auto spine_result = atom::validate_relation_edge_spine_v1(canonical);
    if (!spine_result.valid()) {
        return {relation_edge_coverage_validation_code_v1::
                    invalid_canonical_spine,
                spine_result.index,
                0,
                static_cast<std::uint32_t>(spine_result.code)};
    }
    if (claim.edge_count == 0) {
        return {relation_edge_coverage_validation_code_v1::empty_claim};
    }
    if (claim.global_edge_ids == nullptr) {
        return {relation_edge_coverage_validation_code_v1::
                    missing_claim_edges};
    }
    if (claim.relation_identity != canonical.relation_identity) {
        return {relation_edge_coverage_validation_code_v1::relation_mismatch};
    }
    if (claim.structure_epoch != canonical.structure_epoch) {
        return {relation_edge_coverage_validation_code_v1::epoch_mismatch};
    }

    std::uint64_t canonical_index = 0;
    for (std::uint64_t index = 0; index < claim.edge_count; ++index) {
        const auto edge = claim.global_edge_ids[index];
        if (edge == 0) {
            return {relation_edge_coverage_validation_code_v1::
                        zero_global_edge_identity,
                    index,
                    canonical_index};
        }
        if (index != 0 && claim.global_edge_ids[index - 1] >= edge) {
            return {relation_edge_coverage_validation_code_v1::
                        unordered_or_duplicate_claim_edge,
                    index,
                    canonical_index};
        }
        while (canonical_index < canonical.edge_count
               && canonical.global_edge_ids[canonical_index] < edge) {
            ++canonical_index;
        }
        if (canonical_index == canonical.edge_count
            || canonical.global_edge_ids[canonical_index] != edge) {
            return {relation_edge_coverage_validation_code_v1::
                        edge_not_in_canonical_relation,
                    index,
                    canonical_index};
        }
        ++canonical_index;
    }
    return {relation_edge_coverage_validation_code_v1::valid,
            claim.edge_count,
            canonical_index,
            0};
}

} // namespace cellshard::compiler::certification
