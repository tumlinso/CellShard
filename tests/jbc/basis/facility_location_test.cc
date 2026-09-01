#include "CellShard/compiler/basis/facility_location.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::basis;
    const facility_input facilities[] = {{20, 0, 2, 1}, {10, 2, 1, 0}};
    const facility_edge edges[] = {{0, 4}, {1, 4}, {0, 8}};
    std::uint64_t current[2]{}; local_index selected[2]{};
    const auto result = facility_location_select({facilities, 2, edges, 3, 2}, current, selected, 2);
    assert(result.count == 2 && selected[0] == 1 && selected[1] == 0);
    assert(current[0] == 8 && current[1] == 4);
    return 0;
}
