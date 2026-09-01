#pragma once

#include <CellShard/compiler/partial/dependency_freshness_v1.hh>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::partial {

inline constexpr std::uint32_t partial_cost_evidence_schema_version_v1 = 1;
inline constexpr std::uint64_t partial_complete_cost_mask_v1 = (1ULL << 13) - 1;

enum partial_cost_component_v1 : std::uint64_t {
    partial_cost_cold_build_v1 = 1ULL << 0,
    partial_cost_serialize_v1 = 1ULL << 1,
    partial_cost_publish_v1 = 1ULL << 2,
    partial_cost_acquire_v1 = 1ULL << 3,
    partial_cost_transfer_v1 = 1ULL << 4,
    partial_cost_validate_v1 = 1ULL << 5,
    partial_cost_freshness_v1 = 1ULL << 6,
    partial_cost_execute_v1 = 1ULL << 7,
    partial_cost_merge_v1 = 1ULL << 8,
    partial_cost_finalize_v1 = 1ULL << 9,
    partial_cost_output_transform_v1 = 1ULL << 10,
    partial_cost_synchronize_v1 = 1ULL << 11,
    partial_cost_fallback_v1 = 1ULL << 12,
};

struct partial_cost_evidence_v1 {
    atom_persistent_identity_v1 evidence_identity{};
    atom_persistent_identity_v1 hardware_topology_identity{};
    atom_persistent_identity_v1 toolchain_build_identity{};
    atom_persistent_identity_v1 workload_identity{};
    atom_persistent_identity_v1 fallback_identity{};
    std::uint64_t cost_model_generation = 0;
    std::uint64_t measured_component_mask = 0;
    std::uint64_t cold_build_ns = 0;
    std::uint64_t serialize_ns = 0;
    std::uint64_t publish_ns = 0;
    std::uint64_t acquire_ns = 0;
    std::uint64_t transfer_ns = 0;
    std::uint64_t validate_ns = 0;
    std::uint64_t freshness_ns = 0;
    std::uint64_t execute_ns = 0;
    std::uint64_t merge_ns = 0;
    std::uint64_t finalize_ns = 0;
    std::uint64_t output_transform_ns = 0;
    std::uint64_t synchronize_ns = 0;
    std::uint64_t fallback_ns = 0;
    std::uint64_t expected_reuse = 0;
    std::uint64_t artifact_bytes = 0;
    std::uint64_t fallback_artifact_bytes = 0;
    std::uint64_t warmup_count = 0;
    std::uint64_t repeat_count = 0;
    std::uint32_t correctness_passed = 0;
    std::uint32_t benchmark_mutex_used = 0;
    std::uint32_t schema_version = partial_cost_evidence_schema_version_v1;
    std::uint32_t reserved = 0;
};

enum class partial_promotion_decision_v1 : std::uint32_t {
    promote = 1,
    no_promotion = 2,
    invalid_evidence = 3,
    stale_partial = 4,
    unproven_freshness = 5,
};

enum class partial_promotion_reason_v1 : std::uint32_t {
    lower_complete_amortized_cost = 1,
    no_complete_cost_win,
    invalid_identity,
    evidence_binding_mismatch,
    incomplete_costs,
    missing_measurements,
    correctness_failed,
    unserialized_benchmark,
    invalid_freshness,
    stale_generation,
    missing_freshness_proof,
};

struct partial_promotion_result_v1 {
    partial_promotion_decision_v1 decision =
        partial_promotion_decision_v1::invalid_evidence;
    partial_promotion_reason_v1 reason =
        partial_promotion_reason_v1::invalid_identity;
    long double amortized_partial_ns = 0.0L;
    long double fallback_ns = 0.0L;
    std::uint64_t break_even_reuse = 0;
    [[nodiscard]] constexpr bool promoted() const noexcept {
        return decision == partial_promotion_decision_v1::promote;
    }
};

static_assert(std::is_standard_layout<partial_cost_evidence_v1>::value);
static_assert(std::is_trivially_copyable<partial_cost_evidence_v1>::value);

[[nodiscard]] inline partial_promotion_result_v1 evaluate_partial_promotion_v1(
    const partial_atom_header_v1 &partial,
    const partial_cost_evidence_v1 &evidence,
    const partial_freshness_result_v1 &freshness) noexcept {
    const atom_persistent_identity_v1 identities[]{
        evidence.evidence_identity, evidence.hardware_topology_identity,
        evidence.toolchain_build_identity, evidence.workload_identity,
        evidence.fallback_identity};
    for (const auto identity : identities) {
        if (!atom::validate_atom_persistent_identity_v1(identity).valid()) {
            return {partial_promotion_decision_v1::invalid_evidence,
                    partial_promotion_reason_v1::invalid_identity};
        }
    }
    if (evidence.schema_version != partial_cost_evidence_schema_version_v1
        || evidence.evidence_identity
            != partial.complete_cost_evidence_identity
        || evidence.cost_model_generation != partial.cost_model_generation) {
        return {partial_promotion_decision_v1::invalid_evidence,
                partial_promotion_reason_v1::evidence_binding_mismatch};
    }
    if ((evidence.measured_component_mask & partial_complete_cost_mask_v1)
        != partial_complete_cost_mask_v1) {
        return {partial_promotion_decision_v1::invalid_evidence,
                partial_promotion_reason_v1::incomplete_costs};
    }
    if (evidence.expected_reuse == 0 || evidence.artifact_bytes == 0
        || evidence.fallback_artifact_bytes == 0
        || evidence.warmup_count == 0 || evidence.repeat_count == 0
        || evidence.fallback_ns == 0) {
        return {partial_promotion_decision_v1::invalid_evidence,
                partial_promotion_reason_v1::missing_measurements};
    }
    if (evidence.correctness_passed != 1) {
        return {partial_promotion_decision_v1::invalid_evidence,
                partial_promotion_reason_v1::correctness_failed};
    }
    if (evidence.benchmark_mutex_used != 1) {
        return {partial_promotion_decision_v1::invalid_evidence,
                partial_promotion_reason_v1::unserialized_benchmark};
    }
    if (freshness.freshness == partial_freshness_v1::stale) {
        return {partial_promotion_decision_v1::stale_partial,
                partial_promotion_reason_v1::stale_generation};
    }
    if (freshness.freshness == partial_freshness_v1::unproven) {
        return {partial_promotion_decision_v1::unproven_freshness,
                partial_promotion_reason_v1::missing_freshness_proof};
    }
    if (!freshness.reusable()) {
        return {partial_promotion_decision_v1::invalid_evidence,
                partial_promotion_reason_v1::invalid_freshness};
    }
    const long double cold = static_cast<long double>(evidence.cold_build_ns)
        + evidence.serialize_ns + evidence.publish_ns;
    const long double recurring = static_cast<long double>(evidence.acquire_ns)
        + evidence.transfer_ns + evidence.validate_ns + evidence.freshness_ns
        + evidence.execute_ns + evidence.merge_ns + evidence.finalize_ns
        + evidence.output_transform_ns + evidence.synchronize_ns;
    const long double fallback = static_cast<long double>(evidence.fallback_ns);
    const long double amortized = recurring
        + cold / static_cast<long double>(evidence.expected_reuse);
    std::uint64_t break_even = std::numeric_limits<std::uint64_t>::max();
    if (recurring < fallback) {
        const long double needed = std::ceil(cold / (fallback - recurring));
        break_even = needed > static_cast<long double>(
                                  std::numeric_limits<std::uint64_t>::max())
            ? std::numeric_limits<std::uint64_t>::max()
            : static_cast<std::uint64_t>(needed);
        if (break_even == 0) break_even = 1;
    }
    if (amortized < fallback) {
        return {partial_promotion_decision_v1::promote,
                partial_promotion_reason_v1::lower_complete_amortized_cost,
                amortized, fallback, break_even};
    }
    return {partial_promotion_decision_v1::no_promotion,
            partial_promotion_reason_v1::no_complete_cost_win,
            amortized, fallback, break_even};
}

} // namespace cellshard::compiler::partial
