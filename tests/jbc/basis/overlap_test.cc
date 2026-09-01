#include "CellShard/compiler/basis/overlap.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::basis;
    const atom_coverage atoms[] = {{100, 0, 2}, {200, 2, 2}, {300, 4, 1}};
    const local_index refs[] = {0, 1, 1, 2, 2};
    const std::uint64_t frequencies[] = {5, 4, 3};
    coverage_view view{atoms, 3, refs, 5, frequencies, 3};
    bool covered[3]{}; local_index selected[2]{};
    const auto result = overlap_greedy_select(view, covered, selected, 2);
    assert(result.count == 2 && selected[0] == 0 && selected[1] == 1);
    return 0;
}
