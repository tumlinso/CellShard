#pragma once
#include <CellShard/compiler/discovery/trajectory/branch_local_atom_v1.hh>
#include <cstdint>
namespace cellshard::compiler::discovery::trajectory {
struct branch_delta_proposal_v1{const normalized_lineage_edge_v1*added_edges=nullptr,*removed_edges=nullptr;std::uint64_t added_count=0,removed_count=0;atom::atom_persistent_identity_v1 proposal_identity{},base_atom_identity{},target_atom_identity{};};
enum class branch_delta_code_v1:std::uint32_t{built,invalid_atom,invalid_identity,same_atom,size_overflow,missing_output,insufficient_output};
struct branch_delta_result_v1{branch_delta_code_v1 code=branch_delta_code_v1::built;branch_delta_proposal_v1 proposal{};std::uint64_t required=0;[[nodiscard]]constexpr bool built()const noexcept{return code==branch_delta_code_v1::built;}};
[[nodiscard]]constexpr bool same_edge_v1(normalized_lineage_edge_v1 a,normalized_lineage_edge_v1 b)noexcept{return a.parent_index==b.parent_index&&a.child_index==b.child_index;}
[[nodiscard]]constexpr bool edge_endpoint_less_v1(normalized_lineage_edge_v1 a,normalized_lineage_edge_v1 b)noexcept{return a.parent_index<b.parent_index||(a.parent_index==b.parent_index&&a.child_index<b.child_index);}
[[nodiscard]]constexpr branch_delta_result_v1 build_branch_delta_v1(branch_local_atom_v1 base,branch_local_atom_v1 target,atom::atom_persistent_identity_v1 proposal_identity,normalized_lineage_edge_v1*output,std::uint64_t capacity)noexcept{
 if(base.edges==nullptr||target.edges==nullptr||base.edge_count==0||target.edge_count==0)
  return{branch_delta_code_v1::invalid_atom};
 if(!atom::validate_atom_persistent_identity_v1(proposal_identity).valid())
  return{branch_delta_code_v1::invalid_identity};
 if(base.atom_identity==target.atom_identity)
  return{branch_delta_code_v1::same_atom};
 if(base.edge_count>UINT64_MAX-target.edge_count)
  return{branch_delta_code_v1::size_overflow};
 auto required=base.edge_count+target.edge_count;
 if(output==nullptr)
  return{branch_delta_code_v1::missing_output,{},required};
 if(capacity<required)
  return{branch_delta_code_v1::insufficient_output,{},required};
 std::uint64_t bi=0,ti=0,removed=0,added=0;while(bi<base.edge_count||ti<target.edge_count){if(ti==target.edge_count||(bi<base.edge_count&&edge_endpoint_less_v1(base.edges[bi],target.edges[ti]))){output[removed++]=base.edges[bi++];}else if(bi==base.edge_count||edge_endpoint_less_v1(target.edges[ti],base.edges[bi])){output[base.edge_count+added++]=target.edges[ti++];}else{++bi;++ti;}}
 return{branch_delta_code_v1::built,{output,output+base.edge_count,added,removed,proposal_identity,base.atom_identity,target.atom_identity},required};}
[[nodiscard]]constexpr bool authorizes_execution(branch_delta_proposal_v1)noexcept{return false;}
}
