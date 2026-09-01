#pragma once
#include <CellShard/compiler/discovery/trajectory/temporal_window_v1.hh>
#include <CellShard/compiler/discovery/trajectory/branch_local_atom_v1.hh>
#include <cstdint>

namespace cellshard::compiler::discovery::trajectory {
enum class transition_algebra_v1:std::uint32_t{state_delta_affine=1};
enum class transition_accumulation_v1:std::uint32_t{fp32=1,fp64=2};
struct transition_operator_atom_v1{
 atom::atom_persistent_identity_v1 atom_identity{},branch_atom_identity{},window_atom_identity{},input_domain_identity{},input_order_identity{},output_domain_identity{},output_order_identity{};
 std::uint64_t structure_generation=0,value_generation=0,observation_generation=0,transition_count=0;
 transition_algebra_v1 algebra=transition_algebra_v1::state_delta_affine;
 transition_accumulation_v1 accumulation=transition_accumulation_v1::fp32;
};
enum class transition_operator_code_v1:std::uint32_t{built,invalid_identity,invalid_generation,empty_transitions,invalid_policy,dependency_mismatch};
struct transition_operator_result_v1{transition_operator_code_v1 code=transition_operator_code_v1::built;transition_operator_atom_v1 atom{};[[nodiscard]]constexpr bool built()const noexcept{return code==transition_operator_code_v1::built;}};
[[nodiscard]]constexpr transition_operator_result_v1 build_transition_operator_atom_v1(transition_operator_atom_v1 x,branch_local_atom_v1 branch,temporal_window_atom_v1 window)noexcept{
 const atom::atom_persistent_identity_v1 ids[]={x.atom_identity,x.branch_atom_identity,x.window_atom_identity,x.input_domain_identity,x.input_order_identity,x.output_domain_identity,x.output_order_identity};
 for(const auto&id:ids)if(!atom::validate_atom_persistent_identity_v1(id).valid())return{transition_operator_code_v1::invalid_identity};
 if(x.structure_generation==0||x.value_generation==0||x.observation_generation==0)return{transition_operator_code_v1::invalid_generation};
 if(x.transition_count==0)return{transition_operator_code_v1::empty_transitions};
 if(x.algebra!=transition_algebra_v1::state_delta_affine||(x.accumulation!=transition_accumulation_v1::fp32&&x.accumulation!=transition_accumulation_v1::fp64))return{transition_operator_code_v1::invalid_policy};
 if(x.branch_atom_identity!=branch.atom_identity||x.window_atom_identity!=window.atom_identity||x.observation_generation!=window.observation_generation||x.observation_generation==0)return{transition_operator_code_v1::dependency_mismatch};
 return{transition_operator_code_v1::built,x};
}
[[nodiscard]]constexpr bool authorizes_execution(transition_operator_atom_v1)noexcept{return false;}
}
