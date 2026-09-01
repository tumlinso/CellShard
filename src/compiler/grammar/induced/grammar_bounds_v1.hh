#pragma once
#include <compiler/grammar/induced/repeated_candidate_v1.hh>
#include <cstdint>

namespace cellshard::compiler::grammar::induced {
struct induced_production_shape_v1{induced_identity_v1 production_identity{};const std::uint64_t*child_depths=nullptr;std::uint64_t arity=0,child_capacity=0,declared_depth=0,observation_generation=0;};
struct induced_grammar_bounds_v1{std::uint64_t maximum_arity=0,maximum_depth=0;};
enum class grammar_bounds_code_v1:std::uint32_t{valid,invalid_identity,invalid_policy,empty_arity,missing_children,capacity_overflow,arity_exceeded,invalid_child_depth,depth_overflow,depth_mismatch,depth_exceeded,missing_generation};
struct grammar_bounds_validation_v1{grammar_bounds_code_v1 code=grammar_bounds_code_v1::valid;std::uint64_t index=0,computed_depth=0;[[nodiscard]]constexpr bool valid()const noexcept{return code==grammar_bounds_code_v1::valid;}};
[[nodiscard]]constexpr grammar_bounds_validation_v1 validate_induced_production_bounds_v1(induced_production_shape_v1 x,induced_grammar_bounds_v1 b)noexcept{if(!valid(x.production_identity))return{grammar_bounds_code_v1::invalid_identity};if(b.maximum_arity==0||b.maximum_depth==0)return{grammar_bounds_code_v1::invalid_policy};if(x.arity==0)return{grammar_bounds_code_v1::empty_arity};if(x.child_depths==nullptr)return{grammar_bounds_code_v1::missing_children};if(x.arity>x.child_capacity)return{grammar_bounds_code_v1::capacity_overflow};if(x.arity>b.maximum_arity)return{grammar_bounds_code_v1::arity_exceeded};if(x.observation_generation==0)return{grammar_bounds_code_v1::missing_generation};std::uint64_t maximum=0;for(std::uint64_t i=0;i<x.arity;++i){if(x.child_depths[i]==0)return{grammar_bounds_code_v1::invalid_child_depth,i};if(x.child_depths[i]>maximum)maximum=x.child_depths[i];}if(maximum==UINT64_MAX)return{grammar_bounds_code_v1::depth_overflow};const auto depth=maximum+1;if(x.declared_depth!=depth)return{grammar_bounds_code_v1::depth_mismatch,0,depth};if(depth>b.maximum_depth)return{grammar_bounds_code_v1::depth_exceeded,0,depth};return{grammar_bounds_code_v1::valid,x.arity,depth};}
[[nodiscard]]constexpr bool authorizes_execution(induced_production_shape_v1)noexcept{return false;}
}
