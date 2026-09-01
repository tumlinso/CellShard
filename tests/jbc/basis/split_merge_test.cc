#include "CellShard/compiler/basis/split_merge.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::basis;
    const split_merge_candidate candidates[] = {
        {refinement_kind::split, 20, 21, 100, 70},
        {refinement_kind::merge, 10, 11, 100, 60}};
    const auto* best = best_refinement(candidates, 2);
    assert(best != nullptr && best->kind == refinement_kind::merge);
    return 0;
}
