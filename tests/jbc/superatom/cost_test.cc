#include "CellShard/compiler/composition/superatom/cost.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::composition::superatom;
    const auto value = evaluate_value({10, 3, false}, 4, {5, 6, 7, 8});
    assert(value.benefit == 40 && value.cost == 26 && promotion_profitable(value));
    const auto overflow = evaluate_value({UINT64_MAX, 0, false}, 2, {});
    assert(overflow.saturated && !promotion_profitable(overflow));
    return 0;
}
