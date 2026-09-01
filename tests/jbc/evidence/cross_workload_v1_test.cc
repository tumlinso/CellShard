#include <CellShard/compiler/evidence/cross_workload_v1.hh>

#include <array>
#include <cassert>

namespace evidence = cellshard::compiler::evidence;

int main() {
    std::array<evidence::evidence_identity_v1, 3> workloads{{
        {1, 1}, {1, 2}, {2, 1}}};
    evidence::cross_workload_evidence_view_v1 view{};
    view.workload_identities = workloads.data();
    view.workload_count = workloads.size();
    view.workload_capacity = workloads.size();
    view.evidence_identity = {3, 1};
    view.subject_atom_identity = {3, 2};
    view.graph_family_identity = {3, 3};
    view.observation_generation = 4;
    assert(evidence::validate_cross_workload_evidence_v1(view).valid());
    assert(!evidence::certifies_graph_equivalence(view));

    auto malformed = view;
    malformed.workload_count = 1;
    assert(evidence::validate_cross_workload_evidence_v1(malformed).code
           == evidence::cross_workload_validation_code_v1::insufficient_workloads);
    malformed = view;
    workloads[2] = workloads[1];
    assert(evidence::validate_cross_workload_evidence_v1(malformed).code
           == evidence::cross_workload_validation_code_v1::unordered_or_duplicate_workload);
}
