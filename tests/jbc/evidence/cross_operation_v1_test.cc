#include <CellShard/compiler/evidence/cross_operation_v1.hh>

#include <cassert>

namespace evidence = cellshard::compiler::evidence;

int main() {
    evidence::cross_operation_evidence_v1 record{};
    record.evidence_identity = {1, 1};
    record.subject_atom_identity = {1, 2};
    record.first_operation_identity = {2, 1};
    record.second_operation_identity = {2, 2};
    record.workload_scope_identity = {3, 1};
    record.observation_generation = 4;
    record.joint_observations = 5;
    record.total_observations = 8;
    assert(evidence::validate_cross_operation_evidence_v1(record).valid());
    assert(!evidence::implies_representation_overlap(record));
    assert(!evidence::implies_contribution_overlap(record));

    auto malformed = record;
    malformed.second_operation_identity = malformed.first_operation_identity;
    assert(evidence::validate_cross_operation_evidence_v1(malformed).code
           == evidence::cross_operation_validation_code_v1::
               identical_or_unordered_operations);
    malformed = record;
    malformed.joint_observations = 9;
    assert(evidence::validate_cross_operation_evidence_v1(malformed).code
           == evidence::cross_operation_validation_code_v1::invalid_observation_count);
}
