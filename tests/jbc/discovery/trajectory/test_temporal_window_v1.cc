#include <CellShard/compiler/discovery/trajectory/temporal_window_v1.hh>
#include <array>
#include <cassert>
namespace tr=cellshard::compiler::discovery::trajectory;
int main(){
    tr::trajectory_state_v1 s[]={{1,{1,1},1,2},{2,{1,2},1,4},{3,{1,3},1,8}};
    tr::lineage_edge_v1 e[]={{1,2,{2,1},9},{2,3,{2,1},9}};
    tr::trajectory_lineage_view_v1 v{s,e,3,2,3,2,{3,1},{3,2},{3,3},9};
    cellshard::compiler::atom::atom_persistent_identity_v1 ids[]={{4,1},{4,2},{4,3},{4,4}};
    std::array<tr::temporal_window_atom_v1,4> out{};
    auto r=tr::build_temporal_windows_v1(v,4,2,ids,4,out.data(),out.size());
    assert(r.built()&&r.required==4&&out[0].state_count==2&&out[1].first_state_index==1&&out[3].state_count==1);
    assert(out[0].trajectory_identity==v.trajectory_identity&&out[0].observation_generation==9&&!tr::authorizes_execution(r.view));
}
