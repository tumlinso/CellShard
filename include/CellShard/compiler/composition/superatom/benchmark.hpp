#pragma once

#include "CellShard/compiler/composition/superatom/cost.hpp"

namespace cellshard::compiler::composition::superatom {

struct benchmark_report {
    global_id superatom_id = 0;
    std::uint64_t baseline_ns_per_use = 0;
    std::uint64_t promoted_ns_per_use = 0;
    std::uint64_t expected_uses = 0;
    std::uint64_t lifecycle_overhead_ns = 0;
    std::uint64_t storage_bytes = 0;
    bool provider_correctness_claim = false;
};
struct independent_benchmark_evidence {
    global_id superatom_id = 0;
    bool exact_output_match = false;
    bool representative_workload = false;
};
struct promotion_policy {
    std::uint64_t minimum_saved_ns = 0;
    std::uint64_t maximum_storage_bytes = UINT64_MAX;
};
enum class policy_outcome : std::uint8_t { promote, retain_atoms, invalid_evidence, arithmetic_overflow };
struct policy_result { policy_outcome outcome = policy_outcome::invalid_evidence; std::uint64_t saved_ns = 0; };

inline bool checked_product(std::uint64_t left, std::uint64_t right,
                            std::uint64_t& output) noexcept {
    if (left != 0 && right > UINT64_MAX / left) return false;
    output = left * right;
    return true;
}

inline policy_result evaluate_policy(const benchmark_report& report,
                                     const independent_benchmark_evidence& evidence,
                                     const promotion_policy& policy) noexcept {
    if (report.superatom_id == 0 || report.superatom_id != evidence.superatom_id ||
        !evidence.exact_output_match || !evidence.representative_workload) return {};
    if (report.storage_bytes > policy.maximum_storage_bytes) return {policy_outcome::retain_atoms, 0};
    std::uint64_t baseline_total = 0;
    std::uint64_t promoted_total = 0;
    if (!checked_product(report.baseline_ns_per_use, report.expected_uses, baseline_total) ||
        !checked_product(report.promoted_ns_per_use, report.expected_uses, promoted_total) ||
        report.lifecycle_overhead_ns > UINT64_MAX - promoted_total) return {policy_outcome::arithmetic_overflow, 0};
    promoted_total += report.lifecycle_overhead_ns;
    if (promoted_total >= baseline_total) return {policy_outcome::retain_atoms, 0};
    const std::uint64_t saved = baseline_total - promoted_total;
    return {saved >= policy.minimum_saved_ns ? policy_outcome::promote : policy_outcome::retain_atoms, saved};
}

}  // namespace cellshard::compiler::composition::superatom
