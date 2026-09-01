#include <CellShard/compiler/discovery/trajectory/branch_local_atom_v1.hh>
#include <array>
#include <cassert>
namespace tr=cellshard::compiler::discovery::trajectory;
int main(){std::uint64_t o[]={0,0,1,2};tr::normalized_lineage_edge_v1 e[]={{0,1,{1,2}},{1,2,{1,1}},{1,3,{1,2}}};tr::normalized_lineage_view_v1 v{o,e,3,3,1,{2,1},{3,1},7};cellshard::compiler::atom::atom_persistent_identity_v1 ids[]={{4,1},{4,2}};std::array<tr::normalized_lineage_edge_v1,3>x{};std::array<tr::branch_local_atom_v1,2>a{};auto r=tr::build_branch_local_atoms_v1(v,ids,2,{x.data(),3,a.data(),2});assert(r.built()&&r.view.atom_count==2&&a[0].edge_count==1&&a[1].edge_count==2&&!tr::authorizes_execution(r.view));}
