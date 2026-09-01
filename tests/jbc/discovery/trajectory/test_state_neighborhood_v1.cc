#include <CellShard/compiler/discovery/trajectory/state_neighborhood_v1.hh>
#include <array>
#include <cassert>
namespace tr = cellshard::compiler::discovery::trajectory;
int main() {
    tr::trajectory_state_v1 s[]={{1,{1,1},1,0},{2,{1,2},1,1},{3,{1,3},1,2}};
    tr::lineage_edge_v1 e[]={{1,2,{2,1},7},{2,3,{2,1},7}};
    tr::trajectory_lineage_view_v1 l{s,e,3,2,3,2,{3,1},{4,1},{5,1},7};
    tr::state_neighbor_observation_v1 o[]={{1,2,4,{6,1}},{1,3,25,{6,2}}};
    std::array<tr::state_neighbor_observation_v1,2> out{};
    auto r=tr::build_state_neighborhood_v1(l,o,2,10,9,out.data(),out.size());
    assert(r.built()&&r.view.neighbor_count==1&&!tr::authorizes_execution(r.view));
    o[1].second_state_id=2;
    assert(tr::build_state_neighborhood_v1(l,o,2,10,9,out.data(),out.size()).code==tr::state_neighborhood_code_v1::unordered_or_duplicate_pair);
}
