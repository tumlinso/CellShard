#include <CellShard/compiler/evidence/negative_evidence_v1.hh>

#include <cassert>

namespace evidence = cellshard::compiler::evidence;

int main() {
    evidence::negative_evidence_v1 record{};
    record.evidence_identity = {1, 1};
    record.subject_identity = {1, 2};
    record.observation_scope_identity = {1, 3};
    record.observation_generation = 4;
    record.attempted_observations = 10;
    record.contradictory_observations = 2;
    record.reason = evidence::negative_evidence_reason_v1::contradicted;
    assert(evidence::validate_negative_evidence_v1(record).valid());
    assert(!evidence::certifies_absence(record));

    auto malformed = record;
    malformed.contradictory_observations = 11;
    assert(evidence::validate_negative_evidence_v1(malformed).code
           == evidence::negative_evidence_validation_code_v1::contradiction_overflow);
    malformed = record;
    malformed.contradictory_observations = 0;
    assert(evidence::validate_negative_evidence_v1(malformed).code
           == evidence::negative_evidence_validation_code_v1::inconsistent_contradiction);
    malformed = record;
    malformed.reason = evidence::negative_evidence_reason_v1::candidate_cap_reached;
    assert(evidence::validate_negative_evidence_v1(malformed).valid());
    assert(!evidence::certifies_absence(malformed));
}
