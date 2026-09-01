#pragma once

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::evidence {

// Evidence kinds identify why a candidate was proposed. They do not certify
// logical coverage and never authorize execution.
enum class evidence_kind : std::uint32_t {
    invalid = 0,
    support_signature = 1,
    co_support = 2,
    weighted_co_support = 3,
    normalized_affinity = 4,
    bicluster = 5,
    community_membership = 6,
    factor_membership = 7,
    motif_occurrence = 8,
    sequence_recurrence = 9,
    trajectory_recurrence = 10,
    operation_co_access = 11,
    persistent_order = 12,
    cross_operation = 13,
    cross_workload = 14,
    graph_family = 15,
    cross_dataset_template = 16,
    negative = 17,
};

enum class evidence_family : std::uint32_t {
    invalid = 0,
    support = 1,
    association = 2,
    grouping = 3,
    sequence = 4,
    trajectory = 5,
    operation = 6,
    reuse_template = 7,
    negative = 8,
};

[[nodiscard]] constexpr bool valid_evidence_kind(evidence_kind kind) noexcept {
    switch (kind) {
    case evidence_kind::support_signature:
    case evidence_kind::co_support:
    case evidence_kind::weighted_co_support:
    case evidence_kind::normalized_affinity:
    case evidence_kind::bicluster:
    case evidence_kind::community_membership:
    case evidence_kind::factor_membership:
    case evidence_kind::motif_occurrence:
    case evidence_kind::sequence_recurrence:
    case evidence_kind::trajectory_recurrence:
    case evidence_kind::operation_co_access:
    case evidence_kind::persistent_order:
    case evidence_kind::cross_operation:
    case evidence_kind::cross_workload:
    case evidence_kind::graph_family:
    case evidence_kind::cross_dataset_template:
    case evidence_kind::negative:
        return true;
    case evidence_kind::invalid:
        return false;
    }
    return false;
}

[[nodiscard]] constexpr evidence_family family_of(evidence_kind kind) noexcept {
    switch (kind) {
    case evidence_kind::support_signature:
        return evidence_family::support;
    case evidence_kind::co_support:
    case evidence_kind::weighted_co_support:
    case evidence_kind::normalized_affinity:
        return evidence_family::association;
    case evidence_kind::bicluster:
    case evidence_kind::community_membership:
    case evidence_kind::factor_membership:
        return evidence_family::grouping;
    case evidence_kind::motif_occurrence:
    case evidence_kind::sequence_recurrence:
        return evidence_family::sequence;
    case evidence_kind::trajectory_recurrence:
        return evidence_family::trajectory;
    case evidence_kind::operation_co_access:
    case evidence_kind::persistent_order:
    case evidence_kind::cross_operation:
        return evidence_family::operation;
    case evidence_kind::cross_workload:
    case evidence_kind::graph_family:
    case evidence_kind::cross_dataset_template:
        return evidence_family::reuse_template;
    case evidence_kind::negative:
        return evidence_family::negative;
    case evidence_kind::invalid:
        return evidence_family::invalid;
    }
    return evidence_family::invalid;
}

[[nodiscard]] constexpr bool is_negative_evidence(evidence_kind kind) noexcept {
    return kind == evidence_kind::negative;
}

// All values in this taxonomy remain proposal evidence, including evidence
// produced by an exact rescan. Exact coverage certification is a separate
// compiler contract and is intentionally absent from this enum.
[[nodiscard]] constexpr bool is_proposal_evidence(evidence_kind kind) noexcept {
    return valid_evidence_kind(kind);
}

static_assert(std::is_same<std::underlying_type<evidence_kind>::type,
                           std::uint32_t>::value,
              "evidence_kind must have a stable 32-bit representation");
static_assert(std::is_same<std::underlying_type<evidence_family>::type,
                           std::uint32_t>::value,
              "evidence_family must have a stable 32-bit representation");

} // namespace cellshard::compiler::evidence
