#pragma once
#include <CellShard/compiler/discovery/trajectory/common_prefix_v1.hh>
#include <algorithm>
#include <cstdint>
namespace cellshard::compiler::discovery::trajectory {
struct branch_local_atom_v1{const normalized_lineage_edge_v1*edges=nullptr;std::uint64_t edge_count=0;atom::atom_persistent_identity_v1 atom_identity{},branch_identity{};};
struct branch_local_atom_view_v1{const branch_local_atom_v1*atoms=nullptr;std::uint64_t atom_count=0;atom::atom_persistent_identity_v1 lineage_identity{};std::uint64_t observation_generation=0;};
struct branch_local_buffers_v1{normalized_lineage_edge_v1*edges=nullptr;std::uint64_t edge_capacity=0;branch_local_atom_v1*atoms=nullptr;std::uint64_t atom_capacity=0;};
enum class branch_local_code_v1:std::uint32_t{built,invalid_lineage,missing_ids,invalid_id,unordered_id,missing_output,insufficient_output};
struct branch_local_result_v1{branch_local_code_v1 code=branch_local_code_v1::built;branch_local_atom_view_v1 view{};std::uint64_t index=0,required_atoms=0;[[nodiscard]]constexpr bool built()const noexcept{return code==branch_local_code_v1::built;}};
[[nodiscard]]constexpr bool branch_edge_less_v1(normalized_lineage_edge_v1 a,normalized_lineage_edge_v1 b)noexcept{if(a.branch_identity!=b.branch_identity)return atom::atom_persistent_identity_less_v1(a.branch_identity,b.branch_identity);return a.parent_index<b.parent_index||(a.parent_index==b.parent_index&&a.child_index<b.child_index);}
[[nodiscard]]inline branch_local_result_v1 build_branch_local_atoms_v1(normalized_lineage_view_v1 v,const atom::atom_persistent_identity_v1*ids,std::uint64_t id_count,branch_local_buffers_v1 b)noexcept{
 if(v.edges==nullptr||v.edge_count==0||v.observation_generation==0)
     return{branch_local_code_v1::invalid_lineage};
 if(b.edges==nullptr||b.atoms==nullptr)return{branch_local_code_v1::missing_output};
 if(b.edge_capacity<v.edge_count)return{branch_local_code_v1::insufficient_output};
 for(std::uint64_t i=0;i<v.edge_count;++i)b.edges[i]=v.edges[i];
 std::sort(b.edges,b.edges+v.edge_count,branch_edge_less_v1);
 std::uint64_t required=1;
 for(std::uint64_t i=1;i<v.edge_count;++i)if(b.edges[i-1].branch_identity!=b.edges[i].branch_identity)++required;
 if(ids==nullptr||id_count<required)return{branch_local_code_v1::missing_ids,{},0,required};
 if(b.atom_capacity<required)return{branch_local_code_v1::insufficient_output,{},0,required};
 std::uint64_t cursor=0,begin=0;while(begin<v.edge_count){auto end=begin+1;while(end<v.edge_count&&b.edges[end].branch_identity==b.edges[begin].branch_identity)++end;if(!atom::validate_atom_persistent_identity_v1(ids[cursor]).valid())return{branch_local_code_v1::invalid_id,{},cursor,required};if(cursor&& !atom::atom_persistent_identity_less_v1(ids[cursor-1],ids[cursor]))return{branch_local_code_v1::unordered_id,{},cursor,required};b.atoms[cursor]={b.edges+begin,end-begin,ids[cursor],b.edges[begin].branch_identity};++cursor;begin=end;}
 return{branch_local_code_v1::built,{b.atoms,cursor,v.lineage_identity,v.observation_generation},v.edge_count,required};}
[[nodiscard]]constexpr bool authorizes_execution(branch_local_atom_view_v1)noexcept{return false;}
}
