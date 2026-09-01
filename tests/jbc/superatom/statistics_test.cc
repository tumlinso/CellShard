#include "CellShard/compiler/composition/superatom/statistics.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::composition::superatom;
    const composition_observation observations[] = {{0, 4, 2}, {1, 3, 1}, {0, 5, 4}};
    composition_statistics output[2]{};
    assert(aggregate_statistics(observations, 3, output, 2));
    assert(output[0].frequency == 9 && output[0].reuse_count == 6);
    return 0;
}
