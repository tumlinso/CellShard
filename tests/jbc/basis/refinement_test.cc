#include "CellShard/compiler/basis/refinement.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::basis;
    const refinement_candidate candidates[] = {{30, 2, 0}, {10, 5, 1}, {20, 5, 1}};
    bool selected[] = {true, false, false};
    local_index count = 1;
    assert(refine_add_remove(candidates, 3, selected, 2, count));
    assert(count == 2 && !selected[0] && selected[1] && selected[2]);
    return 0;
}
