#include "CellShard/compiler/basis/greedy.hpp"

#include <cassert>

int main() {
    using namespace cellshard::compiler::basis;
    const atom_utility utilities[] = {{30, {5, 0, 0, 0}},
                                      {10, {5, 0, 0, 0}},
                                      {20, {7, 0, 0, 0}}};
    local_index selected[2]{};
    const auto result = greedy_select(utilities, 3, utility_weights{{1, 0, 0, 0}},
                                      selected, 2);
    assert(result.count == 2);
    assert(selected[0] == 2);
    assert(selected[1] == 1);
    return 0;
}
