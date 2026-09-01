#pragma once
#include <CellShard/compiler/discovery/trajectory/state_neighborhood_v1.hh>
#include <algorithm>
#include <cstdint>

namespace cellshard::compiler::discovery::trajectory {
struct normalized_lineage_edge_v1 { std::uint32_t parent_index=0, child_index=0; atom::atom_persistent_identity_v1 branch_identity{}; };
struct normalized_lineage_view_v1 {
    const std::uint64_t *parent_offsets=nullptr; const normalized_lineage_edge_v1 *edges=nullptr;
    std::uint64_t state_count=0, edge_count=0, root_count=0;
    atom::atom_persistent_identity_v1 lineage_identity{}, trajectory_identity{};
    std::uint64_t observation_generation=0;
};
enum class normalized_lineage_code_v1 : std::uint32_t { built, invalid_lineage, invalid_identity, too_many_states, missing_output, insufficient_output, no_root };
struct normalized_lineage_result_v1 { normalized_lineage_code_v1 code=normalized_lineage_code_v1::built; normalized_lineage_view_v1 view{}; std::uint64_t index=0; [[nodiscard]] constexpr bool built()const noexcept{return code==normalized_lineage_code_v1::built;} };
[[nodiscard]] constexpr bool normalized_edge_less_v1(normalized_lineage_edge_v1 a, normalized_lineage_edge_v1 b) noexcept { return a.child_index<b.child_index||(a.child_index==b.child_index&&a.parent_index<b.parent_index); }
[[nodiscard]] inline normalized_lineage_result_v1 normalize_lineage_v1(
 trajectory_lineage_view_v1 lineage, atom::atom_persistent_identity_v1 lineage_identity,
 std::uint64_t *parent_offsets,std::uint64_t offset_capacity,normalized_lineage_edge_v1 *edges,std::uint64_t edge_capacity) noexcept {
    if(!validate_trajectory_lineage_v1(lineage).valid()) return {normalized_lineage_code_v1::invalid_lineage};
    if(!atom::validate_atom_persistent_identity_v1(lineage_identity).valid()) return {normalized_lineage_code_v1::invalid_identity};
    if(lineage.state_count>UINT32_MAX) return {normalized_lineage_code_v1::too_many_states};
    if(parent_offsets==nullptr||(lineage.edge_count&&edges==nullptr)) return {normalized_lineage_code_v1::missing_output};
    if(offset_capacity<lineage.state_count+1||edge_capacity<lineage.edge_count) return {normalized_lineage_code_v1::insufficient_output};
    for(std::uint64_t i=0;i<lineage.edge_count;++i) edges[i]={static_cast<std::uint32_t>(find_trajectory_state_v1(lineage,lineage.edges[i].parent_state_id)),static_cast<std::uint32_t>(find_trajectory_state_v1(lineage,lineage.edges[i].child_state_id)),lineage.edges[i].branch_identity};
    std::sort(edges,edges+lineage.edge_count,normalized_edge_less_v1);
    std::uint64_t cursor=0,roots=0;
    for(std::uint64_t state=0;state<lineage.state_count;++state){parent_offsets[state]=cursor;while(cursor<lineage.edge_count&&edges[cursor].child_index==state)++cursor;if(parent_offsets[state]==cursor)++roots;}
    parent_offsets[lineage.state_count]=cursor;
    if(roots==0)return {normalized_lineage_code_v1::no_root};
    return {normalized_lineage_code_v1::built,{parent_offsets,edges,lineage.state_count,lineage.edge_count,roots,lineage_identity,lineage.trajectory_identity,lineage.observation_generation},lineage.edge_count};
}
[[nodiscard]] constexpr bool authorizes_execution(normalized_lineage_view_v1)noexcept{return false;}
} // namespace cellshard::compiler::discovery::trajectory
