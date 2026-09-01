#include <CellShard/compiler/discovery/factor_topic/execution_utility_v1.hh>

#include <cassert>
#include <cstdint>
#include <limits>

namespace factor_topic = cellshard::compiler::discovery::factor_topic;

int main() {
    const factor_topic::factor_exact_coverage_span_v1 coverage{{8, 1}, 0, 4};
    factor_topic::factor_execution_cost_v1 cost{};
    cost.candidate_identity = {8, 1};
    cost.baseline_cost_per_use = 100;
    cost.candidate_cost_per_use = 40;
    cost.residual_cost_per_member_per_use = 3;
    cost.preparation_cost = 20;
    cost.acquisition_transfer_cost = 10;
    cost.reconstruction_cost = 5;
    cost.expected_reuse = 4;

    factor_topic::factor_execution_utility_v1 utility{};
    auto result = factor_topic::score_factor_execution_utility_v1(
        coverage, 2, cost, &utility);
    assert(result.scored());
    assert(utility.gross_savings == 240);
    assert(utility.total_overhead == 59);
    assert(utility.net_utility == 181);
    assert(utility.disposition
           == factor_topic::factor_utility_disposition_v1::promote_proposal);
    assert(!factor_topic::authorizes_execution(utility));

    cost.preparation_cost = 300;
    result = factor_topic::score_factor_execution_utility_v1(
        coverage, 2, cost, &utility);
    assert(result.scored());
    assert(utility.net_utility == 0);
    assert(utility.disposition
           == factor_topic::factor_utility_disposition_v1::no_promotion);

    cost.preparation_cost = 0;
    cost.baseline_cost_per_use = 40;
    result = factor_topic::score_factor_execution_utility_v1(
        coverage, 0, cost, &utility);
    assert(result.code
           == factor_topic::factor_execution_utility_code_v1::
               candidate_not_faster);

    cost.baseline_cost_per_use = std::numeric_limits<std::uint64_t>::max();
    cost.candidate_cost_per_use = 0;
    cost.expected_reuse = 2;
    result = factor_topic::score_factor_execution_utility_v1(
        coverage, 0, cost, &utility);
    assert(result.code
           == factor_topic::factor_execution_utility_code_v1::
               arithmetic_overflow);
    return 0;
}
