#include <CellShard/compiler/discovery/trajectory/state_update_overlay_v1.hh>
#include <cassert>
namespace tr=cellshard::compiler::discovery::trajectory;
int main(){
 tr::transition_operator_atom_v1 op{{1,1},{1,2},{1,3},{2,1},{2,2},{2,1},{2,2},4,5,6,2,tr::transition_algebra_v1::state_delta_affine,tr::transition_accumulation_v1::fp64};
 tr::state_update_entry_v1 e[]={{1,-1,2},{4,3,2}};
 tr::state_update_overlay_v1 x{e,2,{3,1},{1,1},{2,1},{2,2},4,5,7,6,tr::transition_accumulation_v1::fp64};
 assert(tr::validate_state_update_overlay_v1(x,op).valid()&&!tr::authorizes_execution(x));
 e[1].state_index=1;
 assert(tr::validate_state_update_overlay_v1(x,op).code==tr::state_update_overlay_code_v1::unordered_or_duplicate);
}
