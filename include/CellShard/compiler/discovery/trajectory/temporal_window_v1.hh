#pragma once
#include <CellShard/compiler/discovery/trajectory/input_contract_v1.hh>
#include <cstdint>

namespace cellshard::compiler::discovery::trajectory {
struct temporal_window_atom_v1 {
    std::uint64_t first_state_index=0,state_count=0,begin_tick=0,end_tick=0;
    atom::atom_persistent_identity_v1 atom_identity{},trajectory_identity{},state_order_identity{};
    std::uint64_t observation_generation=0;
};
struct temporal_window_view_v1 { const temporal_window_atom_v1*atoms=nullptr;std::uint64_t atom_count=0,window_width_ticks=0,stride_ticks=0; };
enum class temporal_window_code_v1:std::uint32_t{built,invalid_lineage,invalid_policy,missing_ids,invalid_id,missing_output,insufficient_output,tick_overflow};
struct temporal_window_result_v1{temporal_window_code_v1 code=temporal_window_code_v1::built;temporal_window_view_v1 view{};std::uint64_t required=0,index=0;[[nodiscard]]constexpr bool built()const noexcept{return code==temporal_window_code_v1::built;}};
[[nodiscard]]constexpr temporal_window_result_v1 build_temporal_windows_v1(trajectory_lineage_view_v1 lineage,std::uint64_t width,std::uint64_t stride,const atom::atom_persistent_identity_v1*ids,std::uint64_t id_count,temporal_window_atom_v1*out,std::uint64_t capacity)noexcept{
    if(!validate_trajectory_lineage_v1(lineage).valid())return{temporal_window_code_v1::invalid_lineage};
    if(width==0||stride==0)return{temporal_window_code_v1::invalid_policy};
    const auto first_tick=lineage.states[0].time_tick;
    const auto last_tick=lineage.states[lineage.state_count-1].time_tick;
    if(last_tick<first_tick)return{temporal_window_code_v1::invalid_lineage};
    const auto span=last_tick-first_tick;
    const auto required=span/stride+1;
    if(ids==nullptr||id_count<required)return{temporal_window_code_v1::missing_ids,{},required};
    if(out==nullptr)return{temporal_window_code_v1::missing_output,{},required};
    if(capacity<required)return{temporal_window_code_v1::insufficient_output,{},required};
    std::uint64_t first=0;
    for(std::uint64_t w=0;w<required;++w){
        if(w>(UINT64_MAX-first_tick)/stride)return{temporal_window_code_v1::tick_overflow,{},required,w};
        const auto begin=first_tick+w*stride;
        const auto end=begin>UINT64_MAX-width?UINT64_MAX:begin+width;
        while(first<lineage.state_count&&lineage.states[first].time_tick<begin)++first;
        auto past=first;
        while(past<lineage.state_count&&lineage.states[past].time_tick<end)++past;
        if(!atom::validate_atom_persistent_identity_v1(ids[w]).valid())return{temporal_window_code_v1::invalid_id,{},required,w};
        out[w]={first,past-first,begin,end,ids[w],lineage.trajectory_identity,lineage.state_order_identity,lineage.observation_generation};
    }
    return{temporal_window_code_v1::built,{out,required,width,stride},required,required};
}
[[nodiscard]]constexpr bool authorizes_execution(temporal_window_view_v1)noexcept{return false;}
}
