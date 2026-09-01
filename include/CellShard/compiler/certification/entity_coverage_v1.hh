#pragma once

#include <CellShard/compiler/certification/canonical_domain_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t entity_coverage_contract_version_v1 = 1;

struct canonical_entity_spine_v1 {
    const std::uint64_t *global_entity_ids = nullptr;
    std::uint64_t entity_count = 0;
    atom::atom_persistent_identity_v1 domain_identity{};
    std::uint64_t source_generation = 0;
};

struct atom_entity_coverage_claim_v1 {
    const std::uint64_t *global_entity_ids = nullptr;
    std::uint64_t entity_count = 0;
    atom::atom_persistent_identity_v1 domain_identity{};
};

enum class entity_coverage_validation_code_v1 : std::uint32_t {
    valid = 0,
    empty_canonical_spine,
    missing_canonical_entities,
    invalid_domain_identity,
    missing_source_generation,
    zero_global_entity_identity,
    unordered_or_duplicate_canonical_entity,
    empty_claim,
    missing_claim_entities,
    domain_mismatch,
    unordered_or_duplicate_claim_entity,
    entity_not_in_canonical_domain,
};

struct entity_coverage_validation_v1 {
    entity_coverage_validation_code_v1 code =
        entity_coverage_validation_code_v1::valid;
    std::uint64_t index = 0;
    std::uint64_t canonical_index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == entity_coverage_validation_code_v1::valid;
    }
};

static_assert(offsetof(canonical_entity_spine_v1, global_entity_ids) == 0);
static_assert(offsetof(atom_entity_coverage_claim_v1, global_entity_ids) == 0);
static_assert(std::is_standard_layout<canonical_entity_spine_v1>::value);
static_assert(std::is_trivially_copyable<canonical_entity_spine_v1>::value);
static_assert(std::is_standard_layout<atom_entity_coverage_claim_v1>::value);
static_assert(
    std::is_trivially_copyable<atom_entity_coverage_claim_v1>::value);

// Both inputs use ascending global u64 identity. The merge scan is exact,
// O(canonical + claim), allocation-free, and never trusts proposal ordinals.
[[nodiscard]] inline entity_coverage_validation_v1
validate_exact_entity_coverage_v1(
    canonical_entity_spine_v1 canonical,
    atom_entity_coverage_claim_v1 claim) noexcept {
    if (canonical.entity_count == 0) {
        return {entity_coverage_validation_code_v1::empty_canonical_spine};
    }
    if (canonical.global_entity_ids == nullptr) {
        return {entity_coverage_validation_code_v1::
                    missing_canonical_entities};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             canonical.domain_identity)
             .valid()) {
        return {entity_coverage_validation_code_v1::invalid_domain_identity};
    }
    if (canonical.source_generation == 0) {
        return {entity_coverage_validation_code_v1::missing_source_generation};
    }
    for (std::uint64_t index = 0; index < canonical.entity_count; ++index) {
        if (canonical.global_entity_ids[index] == 0) {
            return {entity_coverage_validation_code_v1::
                        zero_global_entity_identity,
                    index};
        }
        if (index != 0
            && canonical.global_entity_ids[index - 1]
                >= canonical.global_entity_ids[index]) {
            return {entity_coverage_validation_code_v1::
                        unordered_or_duplicate_canonical_entity,
                    index};
        }
    }
    if (claim.entity_count == 0) {
        return {entity_coverage_validation_code_v1::empty_claim};
    }
    if (claim.global_entity_ids == nullptr) {
        return {entity_coverage_validation_code_v1::missing_claim_entities};
    }
    if (claim.domain_identity != canonical.domain_identity) {
        return {entity_coverage_validation_code_v1::domain_mismatch};
    }

    std::uint64_t canonical_index = 0;
    for (std::uint64_t index = 0; index < claim.entity_count; ++index) {
        const auto entity = claim.global_entity_ids[index];
        if (entity == 0
            || (index != 0 && claim.global_entity_ids[index - 1] >= entity)) {
            return {entity_coverage_validation_code_v1::
                        unordered_or_duplicate_claim_entity,
                    index,
                    canonical_index};
        }
        while (canonical_index < canonical.entity_count
               && canonical.global_entity_ids[canonical_index] < entity) {
            ++canonical_index;
        }
        if (canonical_index == canonical.entity_count
            || canonical.global_entity_ids[canonical_index] != entity) {
            return {entity_coverage_validation_code_v1::
                        entity_not_in_canonical_domain,
                    index,
                    canonical_index};
        }
        ++canonical_index;
    }
    return {entity_coverage_validation_code_v1::valid,
            claim.entity_count,
            canonical_index};
}

} // namespace cellshard::compiler::certification
