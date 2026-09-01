#include <CellShard/compiler/discovery/trajectory/branch_delta_v1.hh>
#include <array>
#include <cassert>
namespace tr=cellshard::compiler::discovery::trajectory;
int main(){tr::normalized_lineage_edge_v1 a[]={{0,1,{1,1}},{1,2,{1,1}}};tr::normalized_lineage_edge_v1 b[]={{0,1,{1,2}},{1,3,{1,2}}};tr::branch_local_atom_v1 x{a,2,{2,1},{1,1}},y{b,2,{2,2},{1,2}};std::array<tr::normalized_lineage_edge_v1,4>o{};auto r=tr::build_branch_delta_v1(x,y,{3,1},o.data(),o.size());assert(r.built()&&r.proposal.removed_count==1&&r.proposal.added_count==1&&!tr::authorizes_execution(r.proposal));}
