#pragma once
#include <compiler/grammar/induced/frequency_stability_v1.hh>
#include <cstdint>

namespace cellshard::compiler::grammar::induced {
struct induced_cost_input_v1{induced_identity_v1 candidate_identity{},cost_model_identity{};std::uint64_t observation_generation=0,execution_ns=0,assembly_ns=0,storage_bytes=0,rebuild_ns=0,invalidation_numerator=0,invalidation_denominator=0,storage_ns_numerator=0,storage_ns_denominator=0;};
struct induced_complete_cost_v1{induced_identity_v1 candidate_identity{},cost_model_identity{};std::uint64_t observation_generation=0,execution_ns=0,assembly_ns=0,storage_ns=0,expected_invalidation_ns=0,total_ns=0;};
enum class complete_cost_code_v1:std::uint32_t{computed,invalid_identity,missing_generation,invalid_policy,overflow};
struct complete_cost_result_v1{complete_cost_code_v1 code=complete_cost_code_v1::computed;induced_complete_cost_v1 cost{};[[nodiscard]]constexpr bool computed()const noexcept{return code==complete_cost_code_v1::computed;}};
[[nodiscard]]constexpr bool multiply_ceil_v1(std::uint64_t a,std::uint64_t b,std::uint64_t d,std::uint64_t&out)noexcept{if(d==0||(a!=0&&b>UINT64_MAX/a))return false;const auto product=a*b;out=product/d+(product%d!=0);return true;}
[[nodiscard]]constexpr complete_cost_result_v1 compute_induced_complete_cost_v1(induced_cost_input_v1 x)noexcept{if(!valid(x.candidate_identity)||!valid(x.cost_model_identity))return{complete_cost_code_v1::invalid_identity};if(x.observation_generation==0)return{complete_cost_code_v1::missing_generation};if(x.execution_ns==0||x.invalidation_denominator==0||x.invalidation_numerator>x.invalidation_denominator||x.storage_ns_denominator==0)return{complete_cost_code_v1::invalid_policy};std::uint64_t storage=0,invalidation=0;if(!multiply_ceil_v1(x.storage_bytes,x.storage_ns_numerator,x.storage_ns_denominator,storage)||!multiply_ceil_v1(x.rebuild_ns,x.invalidation_numerator,x.invalidation_denominator,invalidation))return{complete_cost_code_v1::overflow};std::uint64_t total=x.execution_ns;if(total>UINT64_MAX-x.assembly_ns)return{complete_cost_code_v1::overflow};total+=x.assembly_ns;if(total>UINT64_MAX-storage)return{complete_cost_code_v1::overflow};total+=storage;if(total>UINT64_MAX-invalidation)return{complete_cost_code_v1::overflow};total+=invalidation;return{complete_cost_code_v1::computed,{x.candidate_identity,x.cost_model_identity,x.observation_generation,x.execution_ns,x.assembly_ns,storage,invalidation,total}};}
[[nodiscard]]constexpr bool authorizes_execution(induced_complete_cost_v1)noexcept{return false;}
}
