#include "CellShard/compiler/basis/set_cover.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::basis;
    const atom_coverage atoms[] = {{10, 0, 2}, {20, 2, 1}};
    const local_index refs[] = {0, 1, 1};
    const std::uint64_t freq[] = {4, 4};
    const std::uint64_t costs[] = {8, 1};
    bool covered[2]{}; local_index selected[2]{};
    const auto result = weighted_set_cover({atoms, 2, refs, 3, freq, 2}, costs, covered, selected, 2);
    assert(result.count == 2 && selected[0] == 1 && selected[1] == 0);
    return 0;
}
