#pragma once

#include <CellShard/compiler/certification/entity_coverage_v1.hh>

#include <cstdint>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t read_only_halo_contract_version_v1 = 1;

enum class read_only_halo_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_owned_coverage,
    invalid_halo_coverage,
    owned_halo_overlap,
};

struct read_only_halo_validation_v1 {
    read_only_halo_validation_code_v1 code =
        read_only_halo_validation_code_v1::valid;
    std::uint64_t owned_index = 0;
    std::uint64_t halo_index = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == read_only_halo_validation_code_v1::valid;
    }
};

// Both coverage sets are independently rescanned against the canonical source.
// Their sorted identities then permit a linear disjointness check: a read-only
// halo may overlap another atom, but it cannot masquerade as this atom's owner.
[[nodiscard]] inline read_only_halo_validation_v1 validate_read_only_halo_v1(
    canonical_entity_spine_v1 canonical,
    atom_entity_coverage_claim_v1 owned,
    atom_entity_coverage_claim_v1 halo) noexcept {
    const auto owned_result =
        validate_exact_entity_coverage_v1(canonical, owned);
    if (!owned_result.valid()) {
        return {read_only_halo_validation_code_v1::invalid_owned_coverage,
                owned_result.index,
                0,
                static_cast<std::uint32_t>(owned_result.code)};
    }
    const auto halo_result = validate_exact_entity_coverage_v1(canonical, halo);
    if (!halo_result.valid()) {
        return {read_only_halo_validation_code_v1::invalid_halo_coverage,
                0,
                halo_result.index,
                static_cast<std::uint32_t>(halo_result.code)};
    }
    std::uint64_t owned_index = 0;
    std::uint64_t halo_index = 0;
    while (owned_index < owned.entity_count && halo_index < halo.entity_count) {
        const auto owned_id = owned.global_entity_ids[owned_index];
        const auto halo_id = halo.global_entity_ids[halo_index];
        if (owned_id < halo_id) {
            ++owned_index;
        } else if (halo_id < owned_id) {
            ++halo_index;
        } else {
            return {read_only_halo_validation_code_v1::owned_halo_overlap,
                    owned_index,
                    halo_index};
        }
    }
    return {read_only_halo_validation_code_v1::valid,
            owned.entity_count,
            halo.entity_count,
            0};
}

} // namespace cellshard::compiler::certification
