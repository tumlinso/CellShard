#include "trajectory.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    trajectory_node nodes[] = {{10, no_parent, 0, 100}, {11, 0, 3, 100}, {12, 0, 2, 200}, {13, 1, 5, 100}};
    assert(valid_trajectory(nodes, 4)); nodes[3].parent = 3; assert(!valid_trajectory(nodes, 4)); return 0;
}
