#pragma once

#include "CellShard/compiler/basis/input.hpp"

#include <array>

namespace cellshard::compiler::basis {

struct solver_report {
    global_id solver_id = 0;
    global_id basis_id = 0;
    std::uint64_t claimed_utility = 0;
    std::uint64_t elapsed_ns = 0;
    bool provider_feasible_claim = false;
};

struct independent_basis_evidence {
    global_id basis_id = 0;
    std::array<std::uint64_t, 4> input_digest{};
    std::uint64_t verified_utility = 0;
    std::uint64_t exact_reference_utility = 0;
    bool feasible = false;
    bool exact_reference_available = false;
};

struct promotion_policy {
    std::uint64_t maximum_utility_gap = 0;
    std::uint64_t maximum_elapsed_ns = UINT64_MAX;
    bool require_exact_reference = true;
};

enum class promotion_decision : std::uint8_t {
    promote,
    identity_mismatch,
    missing_reference,
    independently_infeasible,
    utility_mismatch,
    exceeds_reference,
    utility_gap,
    time_limit
};

inline promotion_decision evaluate_promotion(const solver_report& report,
                                             const independent_basis_evidence& evidence,
                                             const promotion_policy& policy) noexcept {
    if (report.basis_id == 0 || report.basis_id != evidence.basis_id) return promotion_decision::identity_mismatch;
    if (policy.require_exact_reference && !evidence.exact_reference_available) return promotion_decision::missing_reference;
    if (!evidence.feasible) return promotion_decision::independently_infeasible;
    if (report.claimed_utility != evidence.verified_utility) return promotion_decision::utility_mismatch;
    if (evidence.exact_reference_available && evidence.verified_utility > evidence.exact_reference_utility) return promotion_decision::exceeds_reference;
    if (evidence.exact_reference_available &&
        evidence.exact_reference_utility - evidence.verified_utility > policy.maximum_utility_gap) return promotion_decision::utility_gap;
    if (report.elapsed_ns > policy.maximum_elapsed_ns) return promotion_decision::time_limit;
    return promotion_decision::promote;
}

inline local_index select_promoted(const solver_report* reports,
                                   const independent_basis_evidence* evidence,
                                   local_index count,
                                   const promotion_policy& policy) noexcept {
    if (reports == nullptr || evidence == nullptr) return invalid_local_index;
    local_index best = invalid_local_index;
    for (local_index i = 0; i < count; ++i) {
        if (evaluate_promotion(reports[i], evidence[i], policy) != promotion_decision::promote) continue;
        if (best == invalid_local_index || evidence[i].verified_utility > evidence[best].verified_utility ||
            (evidence[i].verified_utility == evidence[best].verified_utility && reports[i].elapsed_ns < reports[best].elapsed_ns) ||
            (evidence[i].verified_utility == evidence[best].verified_utility && reports[i].elapsed_ns == reports[best].elapsed_ns &&
             reports[i].solver_id < reports[best].solver_id)) best = i;
    }
    return best;
}

}  // namespace cellshard::compiler::basis
