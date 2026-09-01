#pragma once
#include "metrics.hpp"
#include <array>
namespace cellshard::jbc::validation {
struct atom_ablation { bool exact_occurrence; bool residual_subtraction; bool context_constraints; bool budget_pruning; };
inline constexpr std::array<atom_ablation, 6> atom_ablation_matrix{{
    {false, false, false, false}, {true, true, true, true}, {false, true, true, true},
    {true, false, true, true}, {true, true, false, true}, {true, true, true, false}}};
struct atom_ablation_result { metric_record metric{}; std::uint64_t candidate_count = 0; std::uint64_t accepted_count = 0; };
inline bool comparable(const atom_ablation_result& baseline,
                       const atom_ablation_result& treatment) noexcept {
    return complete_metric(baseline.metric) && complete_metric(treatment.metric) &&
           baseline.metric.fixture_id == treatment.metric.fixture_id &&
           baseline.accepted_count <= baseline.candidate_count &&
           treatment.accepted_count <= treatment.candidate_count;
}
}  // namespace cellshard::jbc::validation
