#pragma once

#include <CellShard/compiler/discovery/sequence_compat/hierarchical_intervals_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::sequence_compat {

inline constexpr std::uint32_t long_range_relation_bridge_schema_version_v1 = 1;

enum class sequence_endpoint_kind_v1 : std::uint8_t {
    enhancer = 1,
    gene = 2,
    contact_locus = 3,
    provider_defined = 4,
};

enum class long_range_relation_kind_v1 : std::uint8_t {
    enhancer_to_gene = 1,
    chromatin_contact = 2,
    provider_defined = 3,
};

struct sequence_entity_identity_map_v1 {
    atom::atom_persistent_identity_v1 interval_identity{};
    atom::atom_persistent_identity_v1 biological_domain_identity{};
    atom::atom_persistent_identity_v1 biological_entity_identity{};
    sequence_endpoint_kind_v1 kind = sequence_endpoint_kind_v1::provider_defined;
    std::uint8_t reserved[7]{};
};

struct long_range_relation_production_v1 {
    atom::atom_persistent_identity_v1 logical_edge_identity{};
    std::uint64_t source_mapping_index = 0;
    std::uint64_t destination_mapping_index = 0;
    long_range_relation_kind_v1 kind =
        long_range_relation_kind_v1::provider_defined;
    std::uint8_t reserved[7]{};
};

// Explicit typed maps bridge provider-owned sequence intervals into a relation
// edge coverage. No sequence payload is re-encoded as a relation payload.
struct long_range_relation_bridge_v1 {
    const sequence_entity_identity_map_v1 *mappings = nullptr;
    std::uint64_t mapping_count = 0;
    const long_range_relation_production_v1 *productions = nullptr;
    std::uint64_t production_count = 0;
    const hierarchical_interval_dag_v1 *intervals = nullptr;
    const atom::atom_logical_coverage_ref_v1 *output_relation_coverage = nullptr;
    atom::atom_persistent_identity_v1 relation_species_identity{};
    std::uint32_t schema_version = long_range_relation_bridge_schema_version_v1;
    std::uint32_t record_bytes = sizeof(long_range_relation_bridge_v1);
};

enum class long_range_relation_bridge_validation_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_record_bytes,
    missing_mappings,
    missing_productions,
    invalid_intervals,
    invalid_output_coverage,
    invalid_relation_species,
    relation_coverage_count_mismatch,
    invalid_mapping_identity,
    duplicate_or_unordered_mapping,
    invalid_mapping_kind,
    nonzero_reserved,
    invalid_edge_identity,
    duplicate_or_unordered_edge,
    invalid_mapping_index,
    self_relation,
    invalid_relation_kind,
    incompatible_endpoint_kinds,
};

struct long_range_relation_bridge_validation_v1 {
    long_range_relation_bridge_validation_code_v1 code =
        long_range_relation_bridge_validation_code_v1::valid;
    std::uint64_t index = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == long_range_relation_bridge_validation_code_v1::valid;
    }
};

static_assert(offsetof(long_range_relation_bridge_v1, mappings) == 0,
              "long-range bridge views must remain pointer-first");
static_assert(std::is_standard_layout<sequence_entity_identity_map_v1>::value);
static_assert(
    std::is_trivially_copyable<sequence_entity_identity_map_v1>::value);
static_assert(std::is_standard_layout<long_range_relation_production_v1>::value);
static_assert(
    std::is_trivially_copyable<long_range_relation_production_v1>::value);
static_assert(std::is_standard_layout<long_range_relation_bridge_v1>::value);
static_assert(std::is_trivially_copyable<long_range_relation_bridge_v1>::value);

[[nodiscard]] inline long_range_relation_bridge_validation_v1
validate_long_range_relation_bridge_v1(
    const long_range_relation_bridge_v1 &bridge,
    std::uint32_t interval_coverage_source_validation,
    std::uint32_t relation_coverage_source_validation) noexcept {
    if (bridge.schema_version != long_range_relation_bridge_schema_version_v1) {
        return {long_range_relation_bridge_validation_code_v1::
                    unsupported_schema,
                0, 0};
    }
    if (bridge.record_bytes != sizeof(long_range_relation_bridge_v1)) {
        return {long_range_relation_bridge_validation_code_v1::
                    invalid_record_bytes,
                0, 0};
    }
    if (bridge.mapping_count == 0 || bridge.mappings == nullptr) {
        return {long_range_relation_bridge_validation_code_v1::missing_mappings,
                0, 0};
    }
    if (bridge.production_count == 0 || bridge.productions == nullptr) {
        return {long_range_relation_bridge_validation_code_v1::
                    missing_productions,
                0, 0};
    }
    if (bridge.intervals == nullptr) {
        return {long_range_relation_bridge_validation_code_v1::invalid_intervals,
                0, 0};
    }
    const auto interval_result = validate_hierarchical_interval_dag_v1(
        *bridge.intervals, interval_coverage_source_validation);
    if (!interval_result.valid()) {
        return {long_range_relation_bridge_validation_code_v1::invalid_intervals,
                interval_result.interval_index,
                static_cast<std::uint32_t>(interval_result.code)};
    }
    if (bridge.output_relation_coverage == nullptr) {
        return {long_range_relation_bridge_validation_code_v1::
                    invalid_output_coverage,
                0, 0};
    }
    const auto output_result = atom::validate_atom_logical_coverage_ref_v1(
        *bridge.output_relation_coverage, relation_coverage_source_validation);
    if (!output_result.valid()
        || bridge.output_relation_coverage->kind
            != atom::atom_logical_coverage_kind_v1::relation_edge_ids) {
        return {long_range_relation_bridge_validation_code_v1::
                    invalid_output_coverage,
                0, static_cast<std::uint32_t>(output_result.code)};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             bridge.relation_species_identity)
             .valid()) {
        return {long_range_relation_bridge_validation_code_v1::
                    invalid_relation_species,
                0, 0};
    }
    if (bridge.output_relation_coverage->logical_count
        != bridge.production_count) {
        return {long_range_relation_bridge_validation_code_v1::
                    relation_coverage_count_mismatch,
                0, 0};
    }

    for (std::uint64_t index = 0; index < bridge.mapping_count; ++index) {
        const auto &mapping = bridge.mappings[index];
        if (!atom::validate_atom_persistent_identity_v1(
                 mapping.interval_identity)
                 .valid()
            || !atom::validate_atom_persistent_identity_v1(
                    mapping.biological_domain_identity)
                    .valid()
            || !atom::validate_atom_persistent_identity_v1(
                    mapping.biological_entity_identity)
                    .valid()) {
            return {long_range_relation_bridge_validation_code_v1::
                        invalid_mapping_identity,
                    index, 0};
        }
        if (index != 0
            && !atom::atom_persistent_identity_less_v1(
                bridge.mappings[index - 1].interval_identity,
                mapping.interval_identity)) {
            return {long_range_relation_bridge_validation_code_v1::
                        duplicate_or_unordered_mapping,
                    index, 0};
        }
        const auto kind = static_cast<std::uint8_t>(mapping.kind);
        if (kind < 1u || kind > 4u) {
            return {long_range_relation_bridge_validation_code_v1::
                        invalid_mapping_kind,
                    index, 0};
        }
        for (const auto item : mapping.reserved) {
            if (item != 0) {
                return {long_range_relation_bridge_validation_code_v1::
                            nonzero_reserved,
                        index, 0};
            }
        }
    }

    for (std::uint64_t index = 0; index < bridge.production_count; ++index) {
        const auto &production = bridge.productions[index];
        if (!atom::validate_atom_persistent_identity_v1(
                 production.logical_edge_identity)
                 .valid()) {
            return {long_range_relation_bridge_validation_code_v1::
                        invalid_edge_identity,
                    index, 0};
        }
        if (index != 0
            && !atom::atom_persistent_identity_less_v1(
                bridge.productions[index - 1].logical_edge_identity,
                production.logical_edge_identity)) {
            return {long_range_relation_bridge_validation_code_v1::
                        duplicate_or_unordered_edge,
                    index, 0};
        }
        if (production.source_mapping_index >= bridge.mapping_count
            || production.destination_mapping_index >= bridge.mapping_count) {
            return {long_range_relation_bridge_validation_code_v1::
                        invalid_mapping_index,
                    index, 0};
        }
        if (production.source_mapping_index
            == production.destination_mapping_index) {
            return {long_range_relation_bridge_validation_code_v1::self_relation,
                    index, 0};
        }
        const auto kind = static_cast<std::uint8_t>(production.kind);
        if (kind < 1u || kind > 3u) {
            return {long_range_relation_bridge_validation_code_v1::
                        invalid_relation_kind,
                    index, 0};
        }
        const auto source_kind =
            bridge.mappings[production.source_mapping_index].kind;
        const auto destination_kind =
            bridge.mappings[production.destination_mapping_index].kind;
        if ((production.kind == long_range_relation_kind_v1::enhancer_to_gene
                && (source_kind != sequence_endpoint_kind_v1::enhancer
                    || destination_kind != sequence_endpoint_kind_v1::gene))
            || (production.kind == long_range_relation_kind_v1::chromatin_contact
                && (source_kind != sequence_endpoint_kind_v1::contact_locus
                    || destination_kind
                        != sequence_endpoint_kind_v1::contact_locus))) {
            return {long_range_relation_bridge_validation_code_v1::
                        incompatible_endpoint_kinds,
                    index, 0};
        }
        for (const auto item : production.reserved) {
            if (item != 0) {
                return {long_range_relation_bridge_validation_code_v1::
                            nonzero_reserved,
                        index, 0};
            }
        }
    }
    return {};
}

} // namespace cellshard::compiler::discovery::sequence_compat
