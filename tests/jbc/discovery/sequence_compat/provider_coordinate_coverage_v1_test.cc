#include <CellShard/compiler/discovery/sequence_compat/provider_coordinate_coverage_v1.hh>

#include <cassert>
#include <cstdint>

namespace atom = cellshard::compiler::atom;
namespace sequence = cellshard::compiler::discovery::sequence_compat;

int main() {
    alignas(std::uint64_t) const std::uint64_t provider_payload[] = {3u, 5u};
    alignas(std::uint64_t) const std::uint64_t source_coverage[] = {7u, 11u};

    sequence::provider_coordinate_coverage_v1 coverage{};
    coverage.provider_payload = provider_payload;
    coverage.exact_coverage.cellerator_coverage = source_coverage;
    coverage.exact_coverage.coverage_identity = {1u, 1u};
    coverage.exact_coverage.logical_count = 4u;
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
    coverage.coordinate_end = 108u;
    coverage.owned_count = 4u;
    assert(sequence::validate_provider_coordinate_coverage_v1(coverage, 0)
               .valid());

    auto malformed = coverage;
    malformed.exact_coverage.kind =
        atom::atom_logical_coverage_kind_v1::relation_edge_ids;
    assert(sequence::validate_provider_coordinate_coverage_v1(malformed, 0)
               .code
        == sequence::provider_coordinate_coverage_validation_code_v1::
            non_provider_defined_coverage);

    malformed = coverage;
    malformed.coordinate_end = malformed.coordinate_begin;
    assert(sequence::validate_provider_coordinate_coverage_v1(malformed, 0)
               .code
        == sequence::provider_coordinate_coverage_validation_code_v1::
            invalid_coordinate_interval);

    malformed = coverage;
    malformed.owned_count = 9u;
    assert(sequence::validate_provider_coordinate_coverage_v1(malformed, 0)
               .code
        == sequence::provider_coordinate_coverage_validation_code_v1::
            invalid_owned_count);

    malformed = coverage;
    malformed.provider_validation_code = 17u;
    const auto provider_failure =
        sequence::validate_provider_coordinate_coverage_v1(malformed, 0);
    assert(provider_failure.code
        == sequence::provider_coordinate_coverage_validation_code_v1::
            provider_validation_failed);
    assert(provider_failure.nested_code == 17u);

    const auto source_failure =
        sequence::validate_provider_coordinate_coverage_v1(coverage, 23u);
    assert(source_failure.code
        == sequence::provider_coordinate_coverage_validation_code_v1::
            invalid_exact_coverage);
    assert(source_failure.nested_code
        == static_cast<std::uint32_t>(
            atom::atom_logical_coverage_validation_code_v1::
                source_validation_failed));
    return 0;
}
