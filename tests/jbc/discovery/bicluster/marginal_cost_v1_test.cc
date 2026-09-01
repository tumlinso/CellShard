#include <CellShard/compiler/discovery/bicluster/marginal_cost_v1.hh>

#include <cassert>
#include <cstdint>
#include <limits>

namespace bicluster = cellshard::compiler::discovery::bicluster;

int main() {
    bicluster::bicluster_complete_cost_v1 cost{};
    cost.baseline_cost_per_interaction = 20;
    cost.candidate_cost_per_interaction = 5;
    cost.residual_cost_per_interaction = 2;
    cost.discovery_cost = 10;
    cost.exact_census_cost = 5;
    cost.projection_cost = 10;
    cost.acquisition_transfer_cost = 5;
    cost.output_transform_cost = 5;
    cost.synchronization_cost = 5;
    cost.expected_reuse = 4;
    bicluster::bicluster_marginal_utility_v1 utility{};
    auto result = bicluster::score_bicluster_marginal_cost_v1(
        8, 2, cost, &utility);
    assert(result.scored());
    assert(utility.gross_savings == 480);
    assert(utility.complete_overhead == 56);
    assert(utility.net_utility == 424);
    assert(utility.disposition == bicluster::bicluster_promotion_v1::promote_proposal);
    assert(!bicluster::authorizes_execution(utility));

    cost.baseline_cost_per_interaction = std::numeric_limits<std::uint64_t>::max();
    cost.candidate_cost_per_interaction = 0;
    result = bicluster::score_bicluster_marginal_cost_v1(2, 0, cost, &utility);
    assert(result.code == bicluster::bicluster_marginal_cost_code_v1::arithmetic_overflow);
    return 0;
}
