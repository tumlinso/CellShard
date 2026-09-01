#include <CellShard/compiler/discovery/factor_topic/internal_experiment_v1.hh>

#include <cassert>

namespace factor_topic = cellshard::compiler::discovery::factor_topic;
namespace evidence = cellshard::compiler::evidence;

int main() {
    factor_topic::factor_execution_utility_v1 utility{};
    utility.candidate_identity = {8, 1};
    utility.exact_owner_count = 4;
    utility.gross_savings = 500;
    utility.total_overhead = 100;
    utility.net_utility = 400;
    utility.disposition = factor_topic::factor_utility_disposition_v1::promote_proposal;

    const factor_topic::factor_experiment_observation_v1 actual[] = {
        {{8, 1}, 100, 0, 1, 0},
        {{8, 1}, 120, 1, 1, 0},
        {{8, 1}, 110, 2, 1, 0},
    };
    const factor_topic::factor_experiment_observation_v1 nulls[] = {
        {{8, 1}, 20, 0, 1, 0},
        {{8, 1}, 40, 1, 1, 0},
        {{8, 1}, 60, 2, 1, 0},
    };
    const factor_topic::factor_null_gate_config_v1 config{8, 3, 80, 0};
    factor_topic::factor_null_gate_outcome_v1 outcome{};
    auto result = factor_topic::run_bounded_factor_null_gate_v1(
        utility, actual, 3, nulls, 3, config, &outcome);
    assert(result.evaluated());
    assert(outcome.disposition
           == factor_topic::factor_utility_disposition_v1::promote_proposal);
    assert(outcome.null_exceedance_count == 0);
    assert(outcome.minimum_actual_utility == 100);
    assert(outcome.maximum_null_utility == 60);
    assert(evidence::validate_confidence_stability_v1(outcome.assessment).valid());
    assert(outcome.assessment.assessment
           == evidence::evidence_assessment_v1::candidate_supported);
    assert(!factor_topic::authorizes_execution(outcome));

    const factor_topic::factor_experiment_observation_v1 strong_nulls[] = {
        {{8, 1}, 100, 0, 1, 0},
    };
    result = factor_topic::run_bounded_factor_null_gate_v1(
        utility, actual, 3, strong_nulls, 1, config, &outcome);
    assert(result.evaluated());
    assert(outcome.disposition
           == factor_topic::factor_utility_disposition_v1::no_promotion);
    assert(outcome.assessment.assessment
           == evidence::evidence_assessment_v1::null_result);

    auto inexact_actual = actual[1];
    inexact_actual.exact_reconstruction = 0;
    const factor_topic::factor_experiment_observation_v1 unstable[] = {
        actual[0], inexact_actual, actual[2]};
    result = factor_topic::run_bounded_factor_null_gate_v1(
        utility, unstable, 3, nulls, 3, config, &outcome);
    assert(result.evaluated());
    assert(outcome.assessment.assessment
           == evidence::evidence_assessment_v1::unstable_evidence);

    auto bounded = config;
    bounded.maximum_observations = 5;
    result = factor_topic::run_bounded_factor_null_gate_v1(
        utility, actual, 3, nulls, 3, bounded, &outcome);
    assert(result.code
           == factor_topic::factor_null_gate_code_v1::
               observation_limit_exceeded);
    return 0;
}
