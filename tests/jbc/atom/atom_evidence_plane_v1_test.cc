#include <CellShard/compiler/atom/evidence_plane_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

using namespace cellshard;
using namespace cellshard::compiler::atom;

alignas(16) std::array<std::byte, 32> source_record{};

content_digest digest_with(std::uint64_t value) {
    content_digest digest{};
    digest.algorithm = digest_algorithm::legacy_fnv1a64;
    digest.used_bytes = sizeof(value);
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        digest.bytes[index] = static_cast<std::byte>(
            (value >> (index * 8)) & UINT64_C(0xff));
    }
    return digest;
}

atom_evidence_record_ref_v1 make_record(std::uint64_t local_identity) {
    atom_evidence_record_ref_v1 record{};
    record.record = source_record.data();
    record.record_bytes = source_record.size();
    record.record_identity = {1, local_identity};
    record.provenance_identity = {2, 1};
    record.provenance_schema = {2, 2};
    record.method_identity = {2, 3};
    record.subject_identity = {2, 4};
    record.record_digest = digest_with(local_identity);
    record.observation_generation = 9;
    record.confidence_numerator = 95;
    record.confidence_denominator = 100;
    record.record_alignment = 16;
    return record;
}

void test_multiple_source_linked_records() {
    std::array<atom_evidence_record_ref_v1, 3> records{
        make_record(1), make_record(2), make_record(3)};
    records[1].kind = atom_evidence_kind_v1::statistical_proposal;
    records[2].kind = atom_evidence_kind_v1::performance_measurement;
    const atom_evidence_plane_v1 plane{
        records.data(), records.size(), {3, 1}, 11};
    const auto result = validate_atom_evidence_plane_v1(plane);
    assert(result.valid());
    assert(result.index == records.size());
}

void test_confidence_does_not_grant_ownership() {
    auto record = make_record(1);
    record.confidence_numerator = 1;
    record.confidence_denominator = 1;
    const atom_evidence_plane_v1 plane{&record, 1, {3, 1}, 1};
    assert(validate_atom_evidence_plane_v1(plane).valid());
    // The evidence schema deliberately has no coverage or ownership field.
    static_assert(sizeof(record.confidence_numerator) == sizeof(std::uint64_t));
}

void test_deterministic_rejections() {
    std::array<atom_evidence_record_ref_v1, 2> records{
        make_record(1), make_record(2)};
    atom_evidence_plane_v1 plane{records.data(), records.size(), {3, 1}, 1};

    records[1].record_identity = records[0].record_identity;
    assert(validate_atom_evidence_plane_v1(plane).code
           == atom_evidence_validation_code_v1::
                  unordered_or_duplicate_record);

    records = {make_record(1), make_record(2)};
    records[0].provenance_identity = {};
    assert(validate_atom_evidence_plane_v1(plane).code
           == atom_evidence_validation_code_v1::invalid_provenance_identity);

    records = {make_record(1), make_record(2)};
    records[0].record_digest = {};
    assert(validate_atom_evidence_plane_v1(plane).code
           == atom_evidence_validation_code_v1::missing_record_digest);

    records = {make_record(1), make_record(2)};
    records[0].confidence_numerator = 101;
    assert(validate_atom_evidence_plane_v1(plane).code
           == atom_evidence_validation_code_v1::invalid_confidence);

    records = {make_record(1), make_record(2)};
    plane.evidence_generation = 0;
    assert(validate_atom_evidence_plane_v1(plane).code
           == atom_evidence_validation_code_v1::
                  missing_evidence_generation);
}

} // namespace

int main() {
    test_multiple_source_linked_records();
    test_confidence_does_not_grant_ownership();
    test_deterministic_rejections();
    return 0;
}
