#include <CellShard/compiler/atom/logical_coverage_v1.hh>

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

using namespace cellshard::compiler::atom;

alignas(std::uint64_t) std::byte source_record[248]{};

atom_logical_coverage_ref_v1 valid_reference() {
    return {
        source_record,
        {71, 901},
        4096,
        cellerator_logical_coverage_schema_version_v1,
        cellerator_logical_coverage_record_bytes_v1,
        atom_certified_exact_coverage_role_v1,
        atom_logical_coverage_kind_v1::relation_edge_ids,
        0,
    };
}

void test_exact_reference() {
    const auto coverage = valid_reference();
    assert(validate_atom_logical_coverage_ref_v1(coverage, 0).valid());
    assert(coverage.cellerator_coverage == source_record);
    assert((coverage.coverage_identity
            == atom_persistent_identity_v1{71, 901}));
    assert(coverage.logical_count == 4096);
}

void test_exact_read_and_physical_roles_remain_labels() {
    auto coverage = valid_reference();
    coverage.role_flags |= atom_exact_read_requirement_role_v1
        | atom_physical_replica_role_v1;
    assert(validate_atom_logical_coverage_ref_v1(coverage, 0).valid());
    assert((coverage.role_flags & atom_certified_exact_coverage_role_v1) != 0);
}

void test_deterministic_rejections() {
    auto coverage = valid_reference();
    coverage.cellerator_coverage = nullptr;
    auto result = validate_atom_logical_coverage_ref_v1(coverage, 0);
    assert(result.code
           == atom_logical_coverage_validation_code_v1::missing_source);

    coverage = valid_reference();
    coverage.cellerator_coverage = source_record + 1;
    result = validate_atom_logical_coverage_ref_v1(coverage, 0);
    assert(result.code
           == atom_logical_coverage_validation_code_v1::misaligned_source);

    coverage = valid_reference();
    coverage.source_schema_version = 2;
    result = validate_atom_logical_coverage_ref_v1(coverage, 0);
    assert(result.code
           == atom_logical_coverage_validation_code_v1::unsupported_schema);

    coverage = valid_reference();
    coverage.source_record_bytes -= 8;
    result = validate_atom_logical_coverage_ref_v1(coverage, 0);
    assert(result.code
           == atom_logical_coverage_validation_code_v1::invalid_record_bytes);

    coverage = valid_reference();
    coverage.coverage_identity = {};
    result = validate_atom_logical_coverage_ref_v1(coverage, 0);
    assert(result.code == atom_logical_coverage_validation_code_v1::
                              invalid_coverage_identity);

    coverage = valid_reference();
    coverage.kind = static_cast<atom_logical_coverage_kind_v1>(8);
    result = validate_atom_logical_coverage_ref_v1(coverage, 0);
    assert(result.code
           == atom_logical_coverage_validation_code_v1::invalid_kind);

    coverage = valid_reference();
    coverage.role_flags = atom_exact_read_requirement_role_v1;
    result = validate_atom_logical_coverage_ref_v1(coverage, 0);
    assert(result.code == atom_logical_coverage_validation_code_v1::
                              missing_exact_certification);

    coverage = valid_reference();
    coverage.role_flags |= atom_approximate_proposal_membership_role_v1;
    result = validate_atom_logical_coverage_ref_v1(coverage, 0);
    assert(result.code == atom_logical_coverage_validation_code_v1::
                              proposal_execution_mixture);

    coverage = valid_reference();
    coverage.role_flags |= 1u << 31u;
    result = validate_atom_logical_coverage_ref_v1(coverage, 0);
    assert(result.code
           == atom_logical_coverage_validation_code_v1::unknown_role);

    coverage = valid_reference();
    coverage.logical_count = 0;
    result = validate_atom_logical_coverage_ref_v1(coverage, 0);
    assert(result.code
           == atom_logical_coverage_validation_code_v1::empty_coverage);

    coverage = valid_reference();
    coverage.reserved = 1;
    result = validate_atom_logical_coverage_ref_v1(coverage, 0);
    assert(result.code
           == atom_logical_coverage_validation_code_v1::nonzero_reserved);

    coverage = valid_reference();
    result = validate_atom_logical_coverage_ref_v1(coverage, 17);
    assert(result.code == atom_logical_coverage_validation_code_v1::
                              source_validation_failed);
    assert(result.source_validation_code == 17);
}

std::uint64_t next_random(std::uint64_t *state) {
    *state = *state * UINT64_C(2862933555777941757) + UINT64_C(3037000493);
    return *state;
}

void test_randomized_exact_references() {
    std::uint64_t state = UINT64_C(0xa06c0de);
    for (std::size_t iteration = 0; iteration < 10000; ++iteration) {
        auto coverage = valid_reference();
        coverage.coverage_identity = {
            next_random(&state) | 1, next_random(&state) | 1};
        coverage.logical_count = next_random(&state) | 1;
        coverage.kind = static_cast<atom_logical_coverage_kind_v1>(
            1 + next_random(&state) % 7);
        coverage.role_flags |= static_cast<std::uint32_t>(
            next_random(&state))
            & (atom_exact_read_requirement_role_v1
               | atom_physical_replica_role_v1);
        assert(validate_atom_logical_coverage_ref_v1(coverage, 0).valid());
    }
}

} // namespace

int main() {
    test_exact_reference();
    test_exact_read_and_physical_roles_remain_labels();
    test_deterministic_rejections();
    test_randomized_exact_references();
    return 0;
}
