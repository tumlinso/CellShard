#pragma once
#include <compiler/grammar/induced/promotion_evidence_v1.hh>
#include <cstdint>

namespace cellshard::compiler::grammar::induced {
enum class fixture_kind_v1:std::uint32_t{planted_repetition=1,adversarial_unique=2};
struct flat_dag_comparison_v1{induced_identity_v1 fixture_identity{},flat_output_identity{},induced_output_identity{};std::uint64_t observation_generation=0,useful_interaction_count=0,flat_complete_ns=0,induced_complete_ns=0;fixture_kind_v1 fixture_kind=fixture_kind_v1::planted_repetition;};
enum class flat_dag_comparison_code_v1:std::uint32_t{compared,invalid_identity,missing_generation,empty_interactions,invalid_cost,output_mismatch,invalid_fixture};
struct flat_dag_comparison_result_v1{flat_dag_comparison_code_v1 code=flat_dag_comparison_code_v1::compared;bool induced_wins=false;std::uint64_t saved_ns=0;[[nodiscard]]constexpr bool compared()const noexcept{return code==flat_dag_comparison_code_v1::compared;}};
[[nodiscard]]constexpr flat_dag_comparison_result_v1 compare_with_flat_dag_v1(flat_dag_comparison_v1 x)noexcept{if(!valid(x.fixture_identity)||!valid(x.flat_output_identity)||!valid(x.induced_output_identity))return{flat_dag_comparison_code_v1::invalid_identity};if(x.observation_generation==0)return{flat_dag_comparison_code_v1::missing_generation};if(x.useful_interaction_count==0)return{flat_dag_comparison_code_v1::empty_interactions};if(x.flat_complete_ns==0||x.induced_complete_ns==0)return{flat_dag_comparison_code_v1::invalid_cost};if(!(x.flat_output_identity==x.induced_output_identity))return{flat_dag_comparison_code_v1::output_mismatch};if(x.fixture_kind!=fixture_kind_v1::planted_repetition&&x.fixture_kind!=fixture_kind_v1::adversarial_unique)return{flat_dag_comparison_code_v1::invalid_fixture};return{flat_dag_comparison_code_v1::compared,x.induced_complete_ns<x.flat_complete_ns,x.induced_complete_ns<x.flat_complete_ns?x.flat_complete_ns-x.induced_complete_ns:0};}
[[nodiscard]]constexpr bool authorizes_execution(flat_dag_comparison_result_v1)noexcept{return false;}
}
