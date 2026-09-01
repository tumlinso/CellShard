#include <CellShard/compiler/discovery/trajectory/input_contract_v1.hh>

#include <cassert>

namespace trajectory = cellshard::compiler::discovery::trajectory;

int main() {
    trajectory::trajectory_state_v1 states[] = {
        {10, {1, 1}, 1, 0}, {20, {1, 2}, 1, 1}, {30, {1, 3}, 1, 2}};
    trajectory::lineage_edge_v1 edges[] = {
        {10, 20, {2, 1}, 7}, {10, 30, {2, 2}, 7}};
    trajectory::trajectory_lineage_view_v1 view{
        states, edges, 3, 2, 3, 2, {3, 1}, {4, 1}, {5, 1}, 7};
    assert(trajectory::validate_trajectory_lineage_v1(view).valid());
    assert(!trajectory::authorizes_execution(view));
    edges[1].child_state_id = 20;
    assert(trajectory::validate_trajectory_lineage_v1(view).code
           == trajectory::trajectory_input_code_v1::
                  unordered_or_duplicate_edge);
    edges[1] = {20, 30, {2, 2}, 6};
    assert(trajectory::validate_trajectory_lineage_v1(view).code
           == trajectory::trajectory_input_code_v1::
                  stale_transition_generation);
}
