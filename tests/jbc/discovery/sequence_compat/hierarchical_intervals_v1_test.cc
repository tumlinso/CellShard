#include <CellShard/compiler/discovery/sequence_compat/hierarchical_intervals_v1.hh>

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
    coverage.exact_coverage.logical_count = 5u;
    coverage.exact_coverage.source_schema_version =
        atom::cellerator_logical_coverage_schema_version_v1;
    coverage.exact_coverage.source_record_bytes =
        atom::cellerator_logical_coverage_record_bytes_v1;
    coverage.exact_coverage.role_flags =
        atom::atom_certified_exact_coverage_role_v1;
    coverage.exact_coverage.kind =
        atom::atom_logical_coverage_kind_v1::provider_defined;
    coverage.reference_identity = {2u, 1u};
    coverage.payload_schema = {3u, 1u};
    coverage.coordinate_begin = 100u;
    coverage.coordinate_end = 120u;
    coverage.owned_count = 5u;

    sequence::reference_strand_identity_v1 reference{};
    reference.assembly_identity = {4u, 1u};
    reference.sequence_identity = {4u, 2u};
    reference.strand = sequence::strand_identity_v1::forward;

    const sequence::hierarchical_interval_parent_v1 child_parent[] = {
        {{5u, 1u}, 0u}};
    const sequence::hierarchical_interval_parent_v1 shared_parents[] = {
        {{5u, 1u}, 0u}, {{5u, 2u}, 1u}};
    sequence::hierarchical_interval_v1 intervals[] = {
        {{5u, 1u}, 100u, 120u, nullptr, 0u},
        {{5u, 2u}, 104u, 116u, child_parent, 1u},
        {{5u, 3u}, 108u, 112u, shared_parents, 2u},
    };
    sequence::hierarchical_interval_dag_v1 dag{};
    dag.intervals = intervals;
    dag.interval_count = 3u;
    dag.coordinate_coverage = &coverage;
    dag.reference = &reference;
    assert(sequence::validate_hierarchical_interval_dag_v1(dag, 0).valid());

    auto malformed_parent = shared_parents[1];
    malformed_parent.parent_index = 2u;
    intervals[2].parents = &malformed_parent;
    intervals[2].parent_count = 1u;
    assert(sequence::validate_hierarchical_interval_dag_v1(dag, 0).code
        == sequence::hierarchical_intervals_validation_code_v1::
            invalid_parent_index);
    intervals[2].parents = shared_parents;
    intervals[2].parent_count = 2u;

    intervals[1].begin = 99u;
    assert(sequence::validate_hierarchical_interval_dag_v1(dag, 0).code
        == sequence::hierarchical_intervals_validation_code_v1::
            interval_outside_coverage);
    intervals[1].begin = 104u;

    intervals[2].end = 118u;
    assert(sequence::validate_hierarchical_interval_dag_v1(dag, 0).code
        == sequence::hierarchical_intervals_validation_code_v1::
            parent_does_not_contain_child);
    intervals[2].end = 112u;

    intervals[2].interval_identity = intervals[1].interval_identity;
    assert(sequence::validate_hierarchical_interval_dag_v1(dag, 0).code
        == sequence::hierarchical_intervals_validation_code_v1::
            duplicate_or_unordered_interval_identity);
    return 0;
}
