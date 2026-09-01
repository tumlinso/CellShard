#pragma once
#include <CellShard/compiler/discovery/trajectory/normalized_lineage_v1.hh>
#include <cstdint>
namespace cellshard::compiler::discovery::trajectory {
struct common_prefix_candidate_v1{std::uint32_t first_state_index=0,branch_state_index=0;std::uint64_t state_count=0;atom::atom_persistent_identity_v1 proposal_identity{};};
struct common_prefix_view_v1{const common_prefix_candidate_v1*candidates=nullptr;std::uint64_t candidate_count=0;atom::atom_persistent_identity_v1 lineage_identity{};std::uint64_t observation_generation=0;};
enum class common_prefix_code_v1:std::uint32_t{built,invalid_lineage,invalid_bound,missing_ids,invalid_id,missing_output,insufficient_output,walk_bound_exceeded};
struct common_prefix_result_v1{common_prefix_code_v1 code=common_prefix_code_v1::built;common_prefix_view_v1 view{};std::uint64_t index=0,required=0;[[nodiscard]]constexpr bool built()const noexcept{return code==common_prefix_code_v1::built;}};
[[nodiscard]] constexpr std::uint64_t child_count_v1(normalized_lineage_view_v1 v,std::uint32_t parent)noexcept{std::uint64_t n=0;for(std::uint64_t i=0;i<v.edge_count;++i)if(v.edges[i].parent_index==parent)++n;return n;}
[[nodiscard]] constexpr common_prefix_result_v1 detect_common_prefixes_v1(normalized_lineage_view_v1 v,std::uint64_t maximum_walk,const atom::atom_persistent_identity_v1*ids,std::uint64_t id_count,common_prefix_candidate_v1*out,std::uint64_t cap)noexcept{
 if(v.parent_offsets==nullptr||v.edges==nullptr||v.state_count==0||v.lineage_identity.producer_namespace==0||v.observation_generation==0)
     return{common_prefix_code_v1::invalid_lineage};
 if(maximum_walk==0)return{common_prefix_code_v1::invalid_bound};
 std::uint64_t required=0;
 for(std::uint32_t branch=0;branch<v.state_count;++branch)if(child_count_v1(v,branch)>1)++required;
 if(ids==nullptr)return{common_prefix_code_v1::missing_ids,{},0,required};
 if(id_count<required)return{common_prefix_code_v1::missing_ids,{},id_count,required};
 if(out==nullptr)return{common_prefix_code_v1::missing_output,{},0,required};
 if(cap<required)return{common_prefix_code_v1::insufficient_output,{},0,required};
 std::uint64_t cursor=0;for(std::uint32_t branch=0;branch<v.state_count;++branch){if(child_count_v1(v,branch)<=1)continue;std::uint32_t first=branch;std::uint64_t count=1;while(v.parent_offsets[first+1]-v.parent_offsets[first]==1){auto parent=v.edges[v.parent_offsets[first]].parent_index;if(child_count_v1(v,parent)!=1)break;if(count==maximum_walk)return{common_prefix_code_v1::walk_bound_exceeded,{},branch,required};first=parent;++count;}if(!atom::validate_atom_persistent_identity_v1(ids[cursor]).valid())return{common_prefix_code_v1::invalid_id,{},cursor,required};out[cursor]={first,branch,count,ids[cursor]};++cursor;}
 return{common_prefix_code_v1::built,{out,cursor,v.lineage_identity,v.observation_generation},v.state_count,required};}
[[nodiscard]]constexpr bool authorizes_execution(common_prefix_view_v1)noexcept{return false;}
}
