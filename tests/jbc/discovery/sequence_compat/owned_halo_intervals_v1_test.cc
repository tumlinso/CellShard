#include <CellShard/compiler/discovery/sequence_compat/owned_halo_intervals_v1.hh>

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
    coverage.coordinate_begin = 90u;
    coverage.coordinate_end = 110u;
    coverage.owned_count = 5u;

    sequence::coordinate_interval_role_record_v1 intervals[] = {
        {92u, 95u, sequence::coordinate_interval_role_v1::read_only_halo,
            false, {}},
        {95u, 100u, sequence::coordinate_interval_role_v1::owned, true, {}},
        {100u, 104u, sequence::coordinate_interval_role_v1::read_only_halo,
            false, {}},
        {107u, 109u, sequence::coordinate_interval_role_v1::read_only_halo,
            false, {}},
    };
    sequence::owned_halo_intervals_v1 view{};
    view.intervals = intervals;
    view.interval_count = 4u;
    view.coordinate_coverage = &coverage;
    view.required_left_halo = 3u;
    view.required_right_halo = 4u;
    assert(sequence::validate_owned_halo_intervals_v1(view, 0).valid());

    intervals[0].contribution_allowed = true;
    assert(sequence::validate_owned_halo_intervals_v1(view, 0).code
        == sequence::owned_halo_intervals_validation_code_v1::
            invalid_contribution_permission);
    intervals[0].contribution_allowed = false;

    view.required_right_halo = 5u;
    assert(sequence::validate_owned_halo_intervals_v1(view, 0).code
        == sequence::owned_halo_intervals_validation_code_v1::
            insufficient_right_halo);
    view.required_right_halo = 4u;

    intervals[1].end = 101u;
    assert(sequence::validate_owned_halo_intervals_v1(view, 0).code
        == sequence::owned_halo_intervals_validation_code_v1::
            unordered_or_overlapping_interval);
    intervals[1].end = 100u;

    coverage.owned_count = 4u;
    coverage.exact_coverage.logical_count = 4u;
    assert(sequence::validate_owned_halo_intervals_v1(view, 0).code
        == sequence::owned_halo_intervals_validation_code_v1::
            owned_count_mismatch);
    return 0;
}
