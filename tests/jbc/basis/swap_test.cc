#include "CellShard/compiler/basis/swap.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::basis;
    bool selected[] = {true, false, false};
    const local_index additions[] = {1, 2};
    assert(apply_swap(swap_proposal{0, additions, 2, 4, 7}, selected, 3));
    assert(!selected[0] && selected[1] && selected[2]);
    const local_index duplicate[] = {1, 1};
    assert(!apply_swap(swap_proposal{0, duplicate, 2, 1, 2}, selected, 3));
    return 0;
}
