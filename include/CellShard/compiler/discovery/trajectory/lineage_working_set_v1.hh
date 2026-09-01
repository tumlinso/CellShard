#pragma once
#include <CellShard/compiler/discovery/trajectory/branch_local_atom_v1.hh>
#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>
#include <cstdint>

namespace cellshard::compiler::discovery::trajectory {
struct lineage_working_set_observation_v1{atom::atom_persistent_identity_v1 branch_atom_identity{};std::uint64_t resident_bytes=0,access_count=0,observation_generation=0;};
struct lineage_working_set_evidence_v1{atom::atom_persistent_identity_v1 branch_atom_identity{};evidence::evidence_identity_v1 evidence_identity{};std::uint64_t maximum_resident_bytes=0,total_access_count=0,observation_count=0,observation_generation=0;};
enum class lineage_working_set_code_v1:std::uint32_t{built,empty_observations,missing_observations,invalid_observation,unordered_observation,count_overflow,missing_ids,invalid_id,missing_output,insufficient_output};
struct lineage_working_set_result_v1{lineage_working_set_code_v1 code=lineage_working_set_code_v1::built;const lineage_working_set_evidence_v1*evidence=nullptr;std::uint64_t evidence_count=0,index=0;[[nodiscard]]constexpr bool built()const noexcept{return code==lineage_working_set_code_v1::built;}};
[[nodiscard]]constexpr lineage_working_set_result_v1 build_lineage_working_set_evidence_v1(const lineage_working_set_observation_v1*observations,std::uint64_t count,const evidence::evidence_identity_v1*ids,std::uint64_t id_count,lineage_working_set_evidence_v1*out,std::uint64_t capacity)noexcept{
    if(count==0)return{lineage_working_set_code_v1::empty_observations};
    if(observations==nullptr)return{lineage_working_set_code_v1::missing_observations};
    std::uint64_t required=1;
    for(std::uint64_t i=0;i<count;++i){
        const auto&x=observations[i];
        if(!atom::validate_atom_persistent_identity_v1(x.branch_atom_identity).valid()||x.resident_bytes==0||x.access_count==0||x.observation_generation==0)return{lineage_working_set_code_v1::invalid_observation,nullptr,0,i};
        if(i&&atom::atom_persistent_identity_less_v1(x.branch_atom_identity,observations[i-1].branch_atom_identity))return{lineage_working_set_code_v1::unordered_observation,nullptr,0,i};
        if(i&&x.branch_atom_identity!=observations[i-1].branch_atom_identity)++required;
    }
    if(ids==nullptr||id_count<required)return{lineage_working_set_code_v1::missing_ids,nullptr,required};
    if(out==nullptr)return{lineage_working_set_code_v1::missing_output,nullptr,required};
    if(capacity<required)return{lineage_working_set_code_v1::insufficient_output,nullptr,required};
    std::uint64_t begin=0,cursor=0;
    while(begin<count){
        auto end=begin+1;while(end<count&&observations[end].branch_atom_identity==observations[begin].branch_atom_identity)++end;
        if(!evidence::valid_evidence_identity_v1(ids[cursor]))return{lineage_working_set_code_v1::invalid_id,nullptr,required,cursor};
        std::uint64_t maximum=0,total=0;const auto generation=observations[begin].observation_generation;
        for(auto i=begin;i<end;++i){if(observations[i].observation_generation!=generation)return{lineage_working_set_code_v1::invalid_observation,nullptr,required,i};if(total>UINT64_MAX-observations[i].access_count)return{lineage_working_set_code_v1::count_overflow,nullptr,required,i};total+=observations[i].access_count;if(observations[i].resident_bytes>maximum)maximum=observations[i].resident_bytes;}
        out[cursor]={observations[begin].branch_atom_identity,ids[cursor],maximum,total,end-begin,generation};++cursor;begin=end;
    }
    return{lineage_working_set_code_v1::built,out,cursor,count};
}
[[nodiscard]]constexpr bool authorizes_execution(lineage_working_set_evidence_v1)noexcept{return false;}
}
