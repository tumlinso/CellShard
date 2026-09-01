#include <CellShard/compiler/evidence/evidence_kind.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace evidence = cellshard::compiler::evidence;

int main() {
    constexpr std::array<evidence::evidence_kind, 17> kinds{{
        evidence::evidence_kind::support_signature,
        evidence::evidence_kind::co_support,
        evidence::evidence_kind::weighted_co_support,
        evidence::evidence_kind::normalized_affinity,
        evidence::evidence_kind::bicluster,
        evidence::evidence_kind::community_membership,
        evidence::evidence_kind::factor_membership,
        evidence::evidence_kind::motif_occurrence,
        evidence::evidence_kind::sequence_recurrence,
        evidence::evidence_kind::trajectory_recurrence,
        evidence::evidence_kind::operation_co_access,
        evidence::evidence_kind::persistent_order,
        evidence::evidence_kind::cross_operation,
        evidence::evidence_kind::cross_workload,
        evidence::evidence_kind::graph_family,
        evidence::evidence_kind::cross_dataset_template,
        evidence::evidence_kind::negative,
    }};

    for (const auto kind : kinds) {
        assert(evidence::valid_evidence_kind(kind));
        assert(evidence::is_proposal_evidence(kind));
        assert(evidence::family_of(kind) != evidence::evidence_family::invalid);
    }

    assert(!evidence::valid_evidence_kind(evidence::evidence_kind::invalid));
    assert(!evidence::is_proposal_evidence(evidence::evidence_kind::invalid));
    assert(evidence::family_of(evidence::evidence_kind::invalid)
           == evidence::evidence_family::invalid);

    const auto unknown = static_cast<evidence::evidence_kind>(UINT32_MAX);
    assert(!evidence::valid_evidence_kind(unknown));
    assert(evidence::family_of(unknown) == evidence::evidence_family::invalid);

    assert(evidence::family_of(evidence::evidence_kind::cross_operation)
           == evidence::evidence_family::operation);
    assert(evidence::family_of(evidence::evidence_kind::cross_workload)
           == evidence::evidence_family::reuse_template);
    assert(evidence::is_negative_evidence(evidence::evidence_kind::negative));
    assert(!evidence::is_negative_evidence(
        evidence::evidence_kind::support_signature));
}
