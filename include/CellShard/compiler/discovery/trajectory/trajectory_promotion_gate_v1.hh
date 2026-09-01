#pragma once
#include <CellShard/compiler/discovery/trajectory/predictive_prefetch_v1.hh>
#include <cstdint>

namespace cellshard::compiler::discovery::trajectory {
struct trajectory_null_controls_v1{const std::uint64_t*complete_cost_ns=nullptr;std::uint64_t trial_count=0,maximum_trial_count=0;evidence::evidence_identity_v1 control_identity{},experiment_identity{};std::uint64_t observation_generation=0;};
struct trajectory_promotion_policy_v1{std::uint64_t minimum_null_trials=0,minimum_null_wins_numerator=0,minimum_null_wins_denominator=0;};
enum class trajectory_promotion_code_v1:std::uint32_t{evaluated,invalid_prefetch,invalid_identity,invalid_generation,invalid_controls,invalid_policy,count_overflow};
struct trajectory_promotion_result_v1{trajectory_promotion_code_v1 code=trajectory_promotion_code_v1::evaluated;bool promote=false;std::uint64_t null_trials_slower=0,null_trial_count=0,observed_complete_cost_ns=0;[[nodiscard]]constexpr bool evaluated()const noexcept{return code==trajectory_promotion_code_v1::evaluated;}};
[[nodiscard]]constexpr trajectory_promotion_result_v1 evaluate_trajectory_promotion_v1(predictive_prefetch_measurement_v1 observed,predictive_prefetch_policy_v1 prefetch_policy,trajectory_null_controls_v1 nulls,trajectory_promotion_policy_v1 policy)noexcept{
 const auto candidate=evaluate_predictive_prefetch_v1(observed,prefetch_policy);
 if(!candidate.evaluated())return{trajectory_promotion_code_v1::invalid_prefetch};
 if(!evidence::valid_evidence_identity_v1(nulls.control_identity)||!(nulls.experiment_identity==observed.experiment_identity))return{trajectory_promotion_code_v1::invalid_identity};
 if(nulls.observation_generation==0||nulls.observation_generation!=observed.observation_generation)return{trajectory_promotion_code_v1::invalid_generation};
 if(nulls.trial_count==0||nulls.complete_cost_ns==nullptr||nulls.trial_count>nulls.maximum_trial_count)return{trajectory_promotion_code_v1::invalid_controls};
 if(policy.minimum_null_trials==0||policy.minimum_null_wins_denominator==0||policy.minimum_null_wins_numerator>policy.minimum_null_wins_denominator)return{trajectory_promotion_code_v1::invalid_policy};
 if(nulls.trial_count<policy.minimum_null_trials)return{trajectory_promotion_code_v1::invalid_controls};
 std::uint64_t slower=0;for(std::uint64_t i=0;i<nulls.trial_count;++i){if(nulls.complete_cost_ns[i]==0)return{trajectory_promotion_code_v1::invalid_controls,false,0,0,i};if(candidate.complete_prefetch_ns<nulls.complete_cost_ns[i])++slower;}
 if(slower>UINT64_MAX/policy.minimum_null_wins_denominator||(policy.minimum_null_wins_numerator&&nulls.trial_count>UINT64_MAX/policy.minimum_null_wins_numerator))return{trajectory_promotion_code_v1::count_overflow};
 const bool passes_null=slower*policy.minimum_null_wins_denominator>=nulls.trial_count*policy.minimum_null_wins_numerator;
 return{trajectory_promotion_code_v1::evaluated,candidate.promote&&passes_null,slower,nulls.trial_count,candidate.complete_prefetch_ns};
}
[[nodiscard]]constexpr bool authorizes_execution(trajectory_promotion_result_v1)noexcept{return false;}
}
