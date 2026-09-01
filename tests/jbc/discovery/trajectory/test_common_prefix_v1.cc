#include <CellShard/compiler/discovery/trajectory/common_prefix_v1.hh>
#include <array>
#include <cassert>
namespace tr=cellshard::compiler::discovery::trajectory;
int main(){std::uint64_t o[]={0,0,1,2,3,4};tr::normalized_lineage_edge_v1 e[]={{0,1,{1,1}},{1,2,{1,1}},{2,3,{1,2}},{2,4,{1,3}}};tr::normalized_lineage_view_v1 v{o,e,5,4,1,{2,1},{3,1},7};cellshard::compiler::atom::atom_persistent_identity_v1 id[]={{4,1}};std::array<tr::common_prefix_candidate_v1,1> out{};auto r=tr::detect_common_prefixes_v1(v,8,id,1,out.data(),1);assert(r.built()&&out[0].first_state_index==0&&out[0].branch_state_index==2&&out[0].state_count==3&&!tr::authorizes_execution(r.view));}
