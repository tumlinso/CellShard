#pragma once
#include <cstdint>

namespace cellshard::compiler::grammar::induced {
struct induced_identity_v1{std::uint64_t producer_namespace=0,local_identity=0;};
[[nodiscard]]constexpr bool valid(induced_identity_v1 x)noexcept{return x.producer_namespace!=0&&x.local_identity!=0;}
[[nodiscard]]constexpr bool operator==(induced_identity_v1 a,induced_identity_v1 b)noexcept{return a.producer_namespace==b.producer_namespace&&a.local_identity==b.local_identity;}
struct execution_observation_v1{std::uint64_t sequence_ordinal=0;induced_identity_v1 operation_identity{},structure_identity{},domain_identity{},order_identity{};std::uint64_t structure_generation=0,value_generation=0,begin_tick=0,end_tick=0,bytes_moved=0,launch_count=0;};
struct induced_grammar_observation_view_v1{const execution_observation_v1*observations=nullptr;std::uint64_t observation_count=0,observation_capacity=0;induced_identity_v1 dataset_identity{},trace_identity{};std::uint64_t observation_generation=0;};
enum class observation_input_code_v1:std::uint32_t{valid,invalid_identity,missing_generation,empty,missing_observations,capacity_overflow,invalid_observation,unordered_or_duplicate_ordinal,overlapping_time};
struct observation_input_validation_v1{observation_input_code_v1 code=observation_input_code_v1::valid;std::uint64_t index=0;[[nodiscard]]constexpr bool valid()const noexcept{return code==observation_input_code_v1::valid;}};
[[nodiscard]]constexpr observation_input_validation_v1 validate_induced_grammar_observations_v1(induced_grammar_observation_view_v1 v)noexcept{if(!valid(v.dataset_identity)||!valid(v.trace_identity))return{observation_input_code_v1::invalid_identity};if(v.observation_generation==0)return{observation_input_code_v1::missing_generation};if(v.observation_count==0)return{observation_input_code_v1::empty};if(v.observations==nullptr)return{observation_input_code_v1::missing_observations};if(v.observation_count>v.observation_capacity)return{observation_input_code_v1::capacity_overflow};for(std::uint64_t i=0;i<v.observation_count;++i){const auto&x=v.observations[i];if(!valid(x.operation_identity)||!valid(x.structure_identity)||!valid(x.domain_identity)||!valid(x.order_identity)||x.structure_generation==0||x.value_generation==0||x.begin_tick>=x.end_tick||x.bytes_moved==0||x.launch_count==0)return{observation_input_code_v1::invalid_observation,i};if(i&&v.observations[i-1].sequence_ordinal>=x.sequence_ordinal)return{observation_input_code_v1::unordered_or_duplicate_ordinal,i};if(i&&v.observations[i-1].end_tick>x.begin_tick)return{observation_input_code_v1::overlapping_time,i};}return{observation_input_code_v1::valid,v.observation_count};}
[[nodiscard]]constexpr bool authorizes_execution(induced_grammar_observation_view_v1)noexcept{return false;}
}
