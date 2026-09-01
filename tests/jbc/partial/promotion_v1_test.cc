#include <CellShard/compiler/partial/promotion_v1.hh>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>

namespace {
using namespace cellshard::compiler::partial;

partial_atom_header_v1 header() {
    partial_atom_header_v1 value{};
    value.complete_cost_evidence_identity = {1, 1};
    value.cost_model_generation = 7;
    return value;
}

partial_cost_evidence_v1 evidence() {
    partial_cost_evidence_v1 value{};
    value.evidence_identity = {1, 1};
    value.hardware_topology_identity = {1, 2};
    value.toolchain_build_identity = {1, 3};
    value.workload_identity = {1, 4};
    value.fallback_identity = {1, 5};
    value.cost_model_generation = 7;
    value.measured_component_mask = partial_complete_cost_mask_v1;
    value.cold_build_ns = 1000; value.serialize_ns = 100; value.publish_ns = 100;
    value.acquire_ns = 10; value.transfer_ns = 10; value.validate_ns = 10;
    value.freshness_ns = 10; value.execute_ns = 20; value.merge_ns = 10;
    value.finalize_ns = 10; value.output_transform_ns = 10;
    value.synchronize_ns = 10; value.fallback_ns = 200;
    value.expected_reuse = 20; value.artifact_bytes = 1024;
    value.fallback_artifact_bytes = 2048; value.warmup_count = 10;
    value.repeat_count = 100; value.correctness_passed = 1;
    value.benchmark_mutex_used = 1;
    return value;
}

partial_freshness_result_v1 current() {
    return {partial_freshness_v1::current,
            partial_freshness_reason_v1::all_generations_match, 2};
}

void test_promotion_and_fail_closed() {
    auto result = evaluate_partial_promotion_v1(header(), evidence(), current());
    assert(result.promoted());
    assert(result.break_even_reuse == 12);
    auto cost = evidence();
    cost.expected_reuse = 2;
    result = evaluate_partial_promotion_v1(header(), cost, current());
    assert(result.decision == partial_promotion_decision_v1::no_promotion);
    cost = evidence();
    cost.measured_component_mask &= ~partial_cost_transfer_v1;
    assert(evaluate_partial_promotion_v1(header(), cost, current()).reason
           == partial_promotion_reason_v1::incomplete_costs);
    cost = evidence(); cost.benchmark_mutex_used = 0;
    assert(evaluate_partial_promotion_v1(header(), cost, current()).reason
           == partial_promotion_reason_v1::unserialized_benchmark);
    auto stale = current(); stale.freshness = partial_freshness_v1::stale;
    assert(evaluate_partial_promotion_v1(header(), evidence(), stale).decision
           == partial_promotion_decision_v1::stale_partial);
}

void test_randomized_complete_cost_differential() {
    std::mt19937_64 generator(0x5be0cd19137e2179ULL);
    for (std::uint32_t trial = 0; trial < 10000; ++trial) {
        auto cost = evidence();
        cost.cold_build_ns = generator() % 100000;
        cost.serialize_ns = generator() % 10000;
        cost.publish_ns = generator() % 10000;
        cost.acquire_ns = generator() % 1000;
        cost.transfer_ns = generator() % 1000;
        cost.validate_ns = generator() % 1000;
        cost.freshness_ns = generator() % 1000;
        cost.execute_ns = generator() % 1000;
        cost.merge_ns = generator() % 1000;
        cost.finalize_ns = generator() % 1000;
        cost.output_transform_ns = generator() % 1000;
        cost.synchronize_ns = generator() % 1000;
        cost.fallback_ns = 1 + generator() % 10000;
        cost.expected_reuse = 1 + generator() % 1000;
        const long double cold = static_cast<long double>(cost.cold_build_ns)
            + cost.serialize_ns + cost.publish_ns;
        const long double recurring = static_cast<long double>(cost.acquire_ns)
            + cost.transfer_ns + cost.validate_ns + cost.freshness_ns
            + cost.execute_ns + cost.merge_ns + cost.finalize_ns
            + cost.output_transform_ns + cost.synchronize_ns;
        const bool oracle = recurring
                + cold / static_cast<long double>(cost.expected_reuse)
            < static_cast<long double>(cost.fallback_ns);
        const auto result = evaluate_partial_promotion_v1(
            header(), cost, current());
        assert(result.promoted() == oracle);
    }
}
}

int main() {
    test_promotion_and_fail_closed();
    test_randomized_complete_cost_differential();
    return 0;
}
