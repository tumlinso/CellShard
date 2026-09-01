#include <CellShard/compiler/discovery/trajectory/normalized_lineage_v1.hh>
#include <array>
#include <cassert>
namespace tr=cellshard::compiler::discovery::trajectory;
int main(){tr::trajectory_state_v1 s[]={{1,{1,1},1,0},{2,{1,2},1,1},{3,{1,3},1,2}};tr::lineage_edge_v1 e[]={{1,2,{2,1},7},{1,3,{2,2},7},{2,3,{2,3},7}};tr::trajectory_lineage_view_v1 l{s,e,3,3,3,3,{3,1},{4,1},{5,1},7};std::array<std::uint64_t,4> o{};std::array<tr::normalized_lineage_edge_v1,3> n{};auto r=tr::normalize_lineage_v1(l,{6,1},o.data(),o.size(),n.data(),n.size());assert(r.built()&&r.view.root_count==1&&o[0]==0&&o[1]==0&&o[2]==1&&o[3]==3&&!tr::authorizes_execution(r.view));}
