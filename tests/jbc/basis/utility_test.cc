#include "CellShard/compiler/basis/utility.hpp"

#include <cassert>
#include <cstdint>

int main() {
    using namespace cellshard::compiler::basis;
    atom_utility utility{UINT64_C(0x100000001), {4, 3, 2, 1}};
    utility_weights weights{{1, 2, 3, 4}};
    const auto result = score(utility, weights);
    assert(result.value == 20);
    assert(!result.saturated);
    utility.benefit[0] = UINT64_MAX;
    weights.value[0] = 2;
    assert(score(utility, weights).saturated);
    return 0;
}
