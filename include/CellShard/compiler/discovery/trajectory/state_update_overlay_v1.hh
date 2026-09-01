#pragma once
#include <CellShard/compiler/discovery/trajectory/transition_operator_v1.hh>
#include <cstdint>

namespace cellshard::compiler::discovery::trajectory {
struct state_update_entry_v1{std::uint64_t state_index=0;std::int64_t delta_numerator=0;std::uint64_t delta_denominator=0;};
struct state_update_overlay_v1{const state_update_entry_v1*entries=nullptr;std::uint64_t entry_count=0;atom::atom_persistent_identity_v1 overlay_identity{},operator_atom_identity{},state_domain_identity{},state_order_identity{};std::uint64_t structure_generation=0,base_value_generation=0,result_value_generation=0,observation_generation=0;transition_accumulation_v1 accumulation=transition_accumulation_v1::fp32;};
enum class state_update_overlay_code_v1:std::uint32_t{valid,empty,missing_entries,invalid_identity,invalid_generation,invalid_entry,unordered_or_duplicate,dependency_mismatch,invalid_policy};
struct state_update_overlay_validation_v1{state_update_overlay_code_v1 code=state_update_overlay_code_v1::valid;std::uint64_t index=0;[[nodiscard]]constexpr bool valid()const noexcept{return code==state_update_overlay_code_v1::valid;}};
[[nodiscard]]constexpr state_update_overlay_validation_v1 validate_state_update_overlay_v1(state_update_overlay_v1 x,transition_operator_atom_v1 op)noexcept{
 if(x.entry_count==0)return{state_update_overlay_code_v1::empty};
 if(x.entries==nullptr)return{state_update_overlay_code_v1::missing_entries};
 const atom::atom_persistent_identity_v1 ids[]={x.overlay_identity,x.operator_atom_identity,x.state_domain_identity,x.state_order_identity};
 for(const auto&id:ids)if(!atom::validate_atom_persistent_identity_v1(id).valid())return{state_update_overlay_code_v1::invalid_identity};
 if(x.structure_generation==0||x.base_value_generation==0||x.result_value_generation==0||x.observation_generation==0||x.result_value_generation<=x.base_value_generation)return{state_update_overlay_code_v1::invalid_generation};
 if(x.operator_atom_identity!=op.atom_identity||x.state_domain_identity!=op.output_domain_identity||x.state_order_identity!=op.output_order_identity||x.structure_generation!=op.structure_generation||x.base_value_generation!=op.value_generation||x.observation_generation!=op.observation_generation)return{state_update_overlay_code_v1::dependency_mismatch};
 if(x.accumulation!=op.accumulation)return{state_update_overlay_code_v1::invalid_policy};
 for(std::uint64_t i=0;i<x.entry_count;++i){if(x.entries[i].delta_denominator==0||x.entries[i].delta_numerator==0)return{state_update_overlay_code_v1::invalid_entry,i};if(i&&x.entries[i-1].state_index>=x.entries[i].state_index)return{state_update_overlay_code_v1::unordered_or_duplicate,i};}
 return{state_update_overlay_code_v1::valid,x.entry_count};
}
[[nodiscard]]constexpr bool authorizes_execution(state_update_overlay_v1)noexcept{return false;}
}
