#include <CellShard/compiler/discovery/sequence_compat/long_range_relation_bridge_v1.hh>

#include <cassert>
#include <cstdint>

namespace atom = cellshard::compiler::atom;
namespace sequence = cellshard::compiler::discovery::sequence_compat;

int main() {
    alignas(std::uint64_t) const std::uint64_t payload[] = {1u};
    sequence::provider_coordinate_coverage_v1 coverage{};
    coverage.provider_payload = payload;
    coverage.exact_coverage.cellerator_coverage = payload;
    coverage.exact_coverage.coverage_identity = {1u, 1u};
    coverage.exact_coverage.logical_count = 2u;
    coverage.exact_coverage.source_schema_version = 1u;
    coverage.exact_coverage.source_record_bytes =
        atom::cellerator_logical_coverage_record_bytes_v1;
    coverage.exact_coverage.role_flags =
        atom::atom_certified_exact_coverage_role_v1;
    coverage.exact_coverage.kind =
        atom::atom_logical_coverage_kind_v1::provider_defined;
    coverage.reference_identity = {2u, 1u};
    coverage.payload_schema = {3u, 1u};
    coverage.coordinate_begin = 0u;
    coverage.coordinate_end = 100u;
    coverage.owned_count = 2u;

    sequence::reference_strand_identity_v1 reference{};
    reference.assembly_identity = {4u, 1u};
    reference.sequence_identity = {4u, 2u};
    sequence::hierarchical_interval_v1 intervals[] = {
        {{5u, 1u}, 10u, 20u, nullptr, 0u},
        {{5u, 2u}, 80u, 90u, nullptr, 0u},
    };
    sequence::hierarchical_interval_dag_v1 dag{};
    dag.intervals = intervals;
    dag.interval_count = 2u;
    dag.coordinate_coverage = &coverage;
    dag.reference = &reference;

    sequence::sequence_entity_identity_map_v1 mappings[] = {
        {{5u, 1u}, {6u, 1u}, {7u, 1u},
            sequence::sequence_endpoint_kind_v1::enhancer, {}},
        {{5u, 2u}, {6u, 2u}, {7u, 2u},
            sequence::sequence_endpoint_kind_v1::gene, {}},
    };
    sequence::long_range_relation_production_v1 production[] = {
        {{8u, 1u}, 0u, 1u,
            sequence::long_range_relation_kind_v1::enhancer_to_gene, {}}};
    atom::atom_logical_coverage_ref_v1 output{};
    output.cellerator_coverage = payload;
    output.coverage_identity = {9u, 1u};
    output.logical_count = 1u;
    output.source_schema_version = 1u;
    output.source_record_bytes =
        atom::cellerator_logical_coverage_record_bytes_v1;
    output.role_flags = atom::atom_certified_exact_coverage_role_v1;
    output.kind = atom::atom_logical_coverage_kind_v1::relation_edge_ids;

    sequence::long_range_relation_bridge_v1 bridge{};
    bridge.mappings = mappings;
    bridge.mapping_count = 2u;
    bridge.productions = production;
    bridge.production_count = 1u;
    bridge.intervals = &dag;
    bridge.output_relation_coverage = &output;
    bridge.relation_species_identity = {10u, 1u};
    assert(sequence::validate_long_range_relation_bridge_v1(bridge, 0, 0)
               .valid());

    production[0].source_mapping_index = 1u;
    assert(sequence::validate_long_range_relation_bridge_v1(bridge, 0, 0).code
        == sequence::long_range_relation_bridge_validation_code_v1::
            self_relation);
    production[0].source_mapping_index = 0u;

    mappings[0].kind = sequence::sequence_endpoint_kind_v1::gene;
    assert(sequence::validate_long_range_relation_bridge_v1(bridge, 0, 0).code
        == sequence::long_range_relation_bridge_validation_code_v1::
            incompatible_endpoint_kinds);
    mappings[0].kind = sequence::sequence_endpoint_kind_v1::enhancer;

    output.kind = atom::atom_logical_coverage_kind_v1::provider_defined;
    assert(sequence::validate_long_range_relation_bridge_v1(bridge, 0, 0).code
        == sequence::long_range_relation_bridge_validation_code_v1::
            invalid_output_coverage);
    output.kind = atom::atom_logical_coverage_kind_v1::relation_edge_ids;

    output.logical_count = 2u;
    assert(sequence::validate_long_range_relation_bridge_v1(bridge, 0, 0).code
        == sequence::long_range_relation_bridge_validation_code_v1::
            relation_coverage_count_mismatch);
    return 0;
}
