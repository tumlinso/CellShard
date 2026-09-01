#include "CellShard/compiler/composition/superatom/evolution.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::composition::superatom;
    const superatom_record parent{{10, 20, 30, 40}, 10, 2, lifecycle_state::promoted};
    superatom_record left{}, right{};
    assert(split(parent, {11, 21, 30, 40}, {12, 22, 30, 40}, {4, 7, true}, left, right) == transition_result::applied);
    superatom_record merged{};
    assert(merge(left, right, {13, 23, 30, 40}, {7, 8, true}, merged) == transition_result::applied);
    assert(merged.lineage_id == 10 && merged.generation == 4);
    return 0;
}
