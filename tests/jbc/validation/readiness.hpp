#pragma once
#include <cstdint>
namespace cellshard::jbc::validation {
struct novelty_evidence {
    bool producer_integration_complete = false;
    bool biological_exact = false;
    bool matched_null_exact = false;
    bool complete_cost = false;
    bool mechanism_ablation_complete = false;
    bool independent_validation = false;
    bool hardware_and_toolchain_recorded = false;
    bool benchmark_resource_reserved = false;
    std::uint64_t biological_saved_ns = 0;
    std::uint64_t matched_null_saved_ns = 0;
};
enum class readiness : std::uint8_t { ready, pending_integration, incomplete_correctness, incomplete_cost, incomplete_ablation, incomplete_provenance, no_biological_specificity };
inline readiness audit_novelty(const novelty_evidence& evidence) noexcept {
    if (!evidence.producer_integration_complete) return readiness::pending_integration;
    if (!evidence.biological_exact || !evidence.matched_null_exact) return readiness::incomplete_correctness;
    if (!evidence.complete_cost) return readiness::incomplete_cost;
    if (!evidence.mechanism_ablation_complete) return readiness::incomplete_ablation;
    if (!evidence.independent_validation || !evidence.hardware_and_toolchain_recorded ||
        !evidence.benchmark_resource_reserved) return readiness::incomplete_provenance;
    if (evidence.biological_saved_ns == 0 || evidence.biological_saved_ns <= evidence.matched_null_saved_ns)
        return readiness::no_biological_specificity;
    return readiness::ready;
}
}  // namespace cellshard::jbc::validation
