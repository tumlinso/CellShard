#include <CellShard/compiler/discovery/trajectory/transition_operator_v1.hh>
#include <cassert>
namespace tr=cellshard::compiler::discovery::trajectory;
int main(){
 tr::normalized_lineage_edge_v1 edges[]={{0,1,{1,9}}};
 tr::branch_local_atom_v1 branch{edges,1,{2,1},{1,9}};
 tr::temporal_window_atom_v1 window{0,2,0,4,{2,2},{3,1},{3,2},7};
 tr::transition_operator_atom_v1 x{{4,1},{2,1},{2,2},{5,1},{5,2},{5,1},{5,2},6,8,7,1,tr::transition_algebra_v1::state_delta_affine,tr::transition_accumulation_v1::fp64};
 auto r=tr::build_transition_operator_atom_v1(x,branch,window);
 assert(r.built()&&r.atom.value_generation==8&&!tr::authorizes_execution(r.atom));
 x.observation_generation=9;
 assert(tr::build_transition_operator_atom_v1(x,branch,window).code==tr::transition_operator_code_v1::dependency_mismatch);
}
