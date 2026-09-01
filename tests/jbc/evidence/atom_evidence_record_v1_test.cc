#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <cassert>
#include <cstdint>

namespace evidence = cellshard::compiler::evidence;

namespace {

constexpr evidence::atom_evidence_record_v1 valid_record() {
    evidence::atom_evidence_record_v1 record{};
    record.evidence_identity = {1, 10};
    record.subject_atom_identity = {2, 20};
    record.source_identity = {3, 30};
    record.observation_generation = 4;
    record.observation_count = 5;
    record.kind = evidence::evidence_kind::support_signature;
    return record;
}

} // namespace

int main() {
    constexpr auto valid = valid_record();
    static_assert(evidence::validate_atom_evidence_record_v1(valid).valid());
    assert(evidence::validate_atom_evidence_record_v1(valid).valid());

    auto malformed = valid;
    malformed.subject_atom_identity = {};
    assert(evidence::validate_atom_evidence_record_v1(malformed).code
           == evidence::atom_evidence_record_validation_code_v1::
               invalid_subject_atom_identity);

    malformed = valid;
    malformed.observation_generation = 0;
    assert(evidence::validate_atom_evidence_record_v1(malformed).code
           == evidence::atom_evidence_record_validation_code_v1::
               missing_observation_generation);

    malformed = valid;
    malformed.kind = static_cast<evidence::evidence_kind>(UINT32_MAX);
    assert(evidence::validate_atom_evidence_record_v1(malformed).code
           == evidence::atom_evidence_record_validation_code_v1::invalid_kind);

    malformed = valid;
    malformed.disposition =
        static_cast<evidence::evidence_disposition_v1>(UINT32_MAX);
    assert(evidence::validate_atom_evidence_record_v1(malformed).code
           == evidence::atom_evidence_record_validation_code_v1::
               non_proposal_disposition);

    assert(evidence::evidence_identity_less_v1({1, 99}, {2, 1}));
    assert(evidence::evidence_identity_less_v1({2, 1}, {2, 2}));
    assert(!evidence::evidence_identity_less_v1({2, 2}, {2, 2}));
}
