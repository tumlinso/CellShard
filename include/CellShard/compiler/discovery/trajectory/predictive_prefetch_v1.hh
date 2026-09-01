#pragma once
#include <CellShard/compiler/discovery/trajectory/transition_operator_v1.hh>
#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>
#include <cstdint>

namespace cellshard::compiler::discovery::trajectory {
struct predictive_prefetch_measurement_v1{
 evidence::evidence_identity_v1 experiment_identity{};
 atom::atom_persistent_identity_v1 operator_atom_identity{},working_set_identity{};
 std::uint64_t observation_generation=0,trial_count=0,correct_trial_count=0;
 std::uint64_t requested_bytes=0,useful_bytes=0;
 std::uint64_t baseline_total_ns=0,prediction_ns=0,prefetch_transfer_ns=0,prefetch_execution_ns=0,prefetch_sync_ns=0;
};
struct predictive_prefetch_policy_v1{std::uint64_t minimum_trials=0,minimum_correct_numerator=0,minimum_correct_denominator=0,minimum_useful_numerator=0,minimum_useful_denominator=0;};
enum class predictive_prefetch_code_v1:std::uint32_t{evaluated,invalid_identity,invalid_generation,invalid_policy,invalid_measurement,cost_overflow};
struct predictive_prefetch_result_v1{predictive_prefetch_code_v1 code=predictive_prefetch_code_v1::evaluated;bool promote=false;std::uint64_t complete_prefetch_ns=0;[[nodiscard]]constexpr bool evaluated()const noexcept{return code==predictive_prefetch_code_v1::evaluated;}};
[[nodiscard]]constexpr predictive_prefetch_result_v1 evaluate_predictive_prefetch_v1(predictive_prefetch_measurement_v1 x,predictive_prefetch_policy_v1 p)noexcept{
 if(!evidence::valid_evidence_identity_v1(x.experiment_identity)||!atom::validate_atom_persistent_identity_v1(x.operator_atom_identity).valid()||!atom::validate_atom_persistent_identity_v1(x.working_set_identity).valid())return{predictive_prefetch_code_v1::invalid_identity};
 if(x.observation_generation==0)return{predictive_prefetch_code_v1::invalid_generation};
 if(p.minimum_trials==0||p.minimum_correct_denominator==0||p.minimum_correct_numerator>p.minimum_correct_denominator||p.minimum_useful_denominator==0||p.minimum_useful_numerator>p.minimum_useful_denominator)return{predictive_prefetch_code_v1::invalid_policy};
 if(x.trial_count<p.minimum_trials||x.correct_trial_count>x.trial_count||x.requested_bytes==0||x.useful_bytes>x.requested_bytes||x.baseline_total_ns==0)return{predictive_prefetch_code_v1::invalid_measurement};
 if(x.correct_trial_count>UINT64_MAX/p.minimum_correct_denominator||(p.minimum_correct_numerator&&x.trial_count>UINT64_MAX/p.minimum_correct_numerator)||x.useful_bytes>UINT64_MAX/p.minimum_useful_denominator||(p.minimum_useful_numerator&&x.requested_bytes>UINT64_MAX/p.minimum_useful_numerator))return{predictive_prefetch_code_v1::cost_overflow};
 if(x.prediction_ns>UINT64_MAX-x.prefetch_transfer_ns)return{predictive_prefetch_code_v1::cost_overflow};
 auto total=x.prediction_ns+x.prefetch_transfer_ns;
 if(total>UINT64_MAX-x.prefetch_execution_ns)return{predictive_prefetch_code_v1::cost_overflow};
 total+=x.prefetch_execution_ns;
 if(total>UINT64_MAX-x.prefetch_sync_ns)return{predictive_prefetch_code_v1::cost_overflow};
 total+=x.prefetch_sync_ns;
 const bool correct=x.correct_trial_count*p.minimum_correct_denominator>=x.trial_count*p.minimum_correct_numerator;
 const bool useful=x.useful_bytes*p.minimum_useful_denominator>=x.requested_bytes*p.minimum_useful_numerator;
 return{predictive_prefetch_code_v1::evaluated,correct&&useful&&total<x.baseline_total_ns,total};
}
[[nodiscard]]constexpr bool authorizes_execution(predictive_prefetch_result_v1)noexcept{return false;}
}
