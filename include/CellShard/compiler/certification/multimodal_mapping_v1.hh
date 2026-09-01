#pragma once

#include <CellShard/compiler/certification/entity_coverage_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t multimodal_mapping_contract_version_v1 = 1;

struct multimodal_identity_edge_v1 {
    std::uint64_t source_global_entity_id = 0;
    std::uint64_t destination_global_entity_id = 0;
};

struct multimodal_identity_mapping_view_v1 {
    const multimodal_identity_edge_v1 *edges = nullptr;
    std::uint64_t edge_count = 0;
    atom::atom_persistent_identity_v1 mapping_identity{};
    atom::atom_persistent_identity_v1 source_domain_identity{};
    atom::atom_persistent_identity_v1 destination_domain_identity{};
    std::uint64_t mapping_generation = 0;
};

enum class multimodal_mapping_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_source_spine,
    invalid_destination_spine,
    empty_mapping,
    missing_edges,
    invalid_mapping_identity,
    source_domain_mismatch,
    destination_domain_mismatch,
    missing_mapping_generation,
    zero_entity_identity,
    unordered_or_duplicate_edge,
    source_not_canonical,
    destination_not_canonical,
};

struct multimodal_mapping_validation_v1 {
    multimodal_mapping_validation_code_v1 code =
        multimodal_mapping_validation_code_v1::valid;
    std::uint64_t edge_index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == multimodal_mapping_validation_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<multimodal_identity_edge_v1>::value);
static_assert(std::is_trivially_copyable<multimodal_identity_edge_v1>::value);
static_assert(offsetof(multimodal_identity_mapping_view_v1, edges) == 0);
static_assert(
    std::is_standard_layout<multimodal_identity_mapping_view_v1>::value);
static_assert(
    std::is_trivially_copyable<multimodal_identity_mapping_view_v1>::value);

[[nodiscard]] constexpr bool multimodal_identity_edge_less_v1(
    multimodal_identity_edge_v1 lhs,
    multimodal_identity_edge_v1 rhs) noexcept {
    return lhs.source_global_entity_id < rhs.source_global_entity_id
        || (lhs.source_global_entity_id == rhs.source_global_entity_id
            && lhs.destination_global_entity_id
                < rhs.destination_global_entity_id);
}

[[nodiscard]] inline bool canonical_spine_contains_v1(
    canonical_entity_spine_v1 spine,
    std::uint64_t identity) noexcept {
    std::uint64_t begin = 0;
    std::uint64_t end = spine.entity_count;
    while (begin < end) {
        const auto middle = begin + (end - begin) / 2;
        if (spine.global_entity_ids[middle] < identity) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    return begin < spine.entity_count
        && spine.global_entity_ids[begin] == identity;
}

// Exact pair identity is source/destination global u64 identity plus the two
// canonical domains. Validation is O(E(log S + log D)); equal ordinals or
// shapes across modalities have no meaning.
[[nodiscard]] inline multimodal_mapping_validation_v1
validate_multimodal_identity_mapping_v1(
    canonical_entity_spine_v1 source,
    canonical_entity_spine_v1 destination,
    multimodal_identity_mapping_view_v1 mapping) noexcept {
    const atom_entity_coverage_claim_v1 full_source{
        source.global_entity_ids, source.entity_count, source.domain_identity};
    if (!validate_exact_entity_coverage_v1(source, full_source).valid()) {
        return {multimodal_mapping_validation_code_v1::invalid_source_spine};
    }
    const atom_entity_coverage_claim_v1 full_destination{
        destination.global_entity_ids,
        destination.entity_count,
        destination.domain_identity};
    if (!validate_exact_entity_coverage_v1(destination, full_destination)
             .valid()) {
        return {multimodal_mapping_validation_code_v1::
                    invalid_destination_spine};
    }
    if (mapping.edge_count == 0) {
        return {multimodal_mapping_validation_code_v1::empty_mapping};
    }
    if (mapping.edges == nullptr) {
        return {multimodal_mapping_validation_code_v1::missing_edges};
    }
    if (!atom::validate_atom_persistent_identity_v1(mapping.mapping_identity)
             .valid()) {
        return {multimodal_mapping_validation_code_v1::
                    invalid_mapping_identity};
    }
    if (mapping.source_domain_identity != source.domain_identity) {
        return {multimodal_mapping_validation_code_v1::source_domain_mismatch};
    }
    if (mapping.destination_domain_identity != destination.domain_identity) {
        return {multimodal_mapping_validation_code_v1::
                    destination_domain_mismatch};
    }
    if (mapping.mapping_generation == 0) {
        return {multimodal_mapping_validation_code_v1::
                    missing_mapping_generation};
    }
    for (std::uint64_t index = 0; index < mapping.edge_count; ++index) {
        const auto edge = mapping.edges[index];
        if (edge.source_global_entity_id == 0
            || edge.destination_global_entity_id == 0) {
            return {multimodal_mapping_validation_code_v1::
                        zero_entity_identity,
                    index};
        }
        if (index != 0
            && !multimodal_identity_edge_less_v1(
                mapping.edges[index - 1], edge)) {
            return {multimodal_mapping_validation_code_v1::
                        unordered_or_duplicate_edge,
                    index};
        }
        if (!canonical_spine_contains_v1(
                source, edge.source_global_entity_id)) {
            return {multimodal_mapping_validation_code_v1::
                        source_not_canonical,
                    index};
        }
        if (!canonical_spine_contains_v1(
                destination, edge.destination_global_entity_id)) {
            return {multimodal_mapping_validation_code_v1::
                        destination_not_canonical,
                    index};
        }
    }
    return {multimodal_mapping_validation_code_v1::valid,
            mapping.edge_count};
}

} // namespace cellshard::compiler::certification
