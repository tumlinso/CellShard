#include "CellShard/compiler/basis/baseline.hpp"

#include <cassert>

int main() {
    using namespace cellshard::compiler::basis;
    const workload_family_input families[] = {{9, 4, 0, 0}, {12, 7, 0, 0}};
    const basis_input_view input{families, 2, nullptr, 0, nullptr, 0};
    const auto result = canonical_no_basis(input);
    assert(result.count == 0);
    assert(result.uncovered_frequency == 11);
    assert(!result.saturated);
    return 0;
}
