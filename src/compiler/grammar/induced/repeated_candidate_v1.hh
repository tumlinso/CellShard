#pragma once
#include <compiler/grammar/induced/observation_input_v1.hh>
#include <cstdint>

namespace cellshard::compiler::grammar::induced {
struct repeated_production_candidate_v1{std::uint64_t first_begin=0,second_begin=0,symbol_count=0;induced_identity_v1 candidate_identity{},trace_identity{};std::uint64_t observation_generation=0;};
enum class repeated_candidate_code_v1:std::uint32_t{mined,invalid_input,invalid_policy,missing_ids,invalid_id,missing_output,insufficient_output,candidate_limit};
struct repeated_candidate_result_v1{repeated_candidate_code_v1 code=repeated_candidate_code_v1::mined;const repeated_production_candidate_v1*candidates=nullptr;std::uint64_t candidate_count=0,required=0;[[nodiscard]]constexpr bool mined()const noexcept{return code==repeated_candidate_code_v1::mined;}};
[[nodiscard]]constexpr bool same_structural_symbol_v1(const execution_observation_v1&a,const execution_observation_v1&b)noexcept{return a.operation_identity==b.operation_identity&&a.structure_identity==b.structure_identity&&a.domain_identity==b.domain_identity&&a.order_identity==b.order_identity&&a.structure_generation==b.structure_generation;}
[[nodiscard]]constexpr repeated_candidate_result_v1 mine_repeated_candidates_v1(induced_grammar_observation_view_v1 v,std::uint64_t minimum_length,std::uint64_t maximum_candidates,const induced_identity_v1*ids,std::uint64_t id_count,repeated_production_candidate_v1*out,std::uint64_t capacity)noexcept{
 if(!validate_induced_grammar_observations_v1(v).valid())return{repeated_candidate_code_v1::invalid_input};
 if(minimum_length==0||maximum_candidates==0)return{repeated_candidate_code_v1::invalid_policy};
 std::uint64_t required=0;
 for(std::uint64_t i=0;i<v.observation_count;++i)for(std::uint64_t j=i+minimum_length;j<v.observation_count;++j){std::uint64_t n=0;while(j+n<v.observation_count&&same_structural_symbol_v1(v.observations[i+n],v.observations[j+n]))++n;if(n>=minimum_length){if(required==maximum_candidates)return{repeated_candidate_code_v1::candidate_limit,nullptr,0,required};++required;}}
 if(ids==nullptr||id_count<required)return{repeated_candidate_code_v1::missing_ids,nullptr,0,required};
 if(out==nullptr)return{repeated_candidate_code_v1::missing_output,nullptr,0,required};
 if(capacity<required)return{repeated_candidate_code_v1::insufficient_output,nullptr,0,required};
 std::uint64_t cursor=0;
 for(std::uint64_t i=0;i<v.observation_count;++i)for(std::uint64_t j=i+minimum_length;j<v.observation_count;++j){std::uint64_t n=0;while(j+n<v.observation_count&&same_structural_symbol_v1(v.observations[i+n],v.observations[j+n]))++n;if(n>=minimum_length){if(!valid(ids[cursor]))return{repeated_candidate_code_v1::invalid_id,nullptr,0,cursor};out[cursor]={i,j,n,ids[cursor],v.trace_identity,v.observation_generation};++cursor;}}
 return{repeated_candidate_code_v1::mined,out,cursor,required};
}
[[nodiscard]]constexpr bool authorizes_execution(repeated_production_candidate_v1)noexcept{return false;}
}
