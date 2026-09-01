#include "CellShard/compiler/basis/budget.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::basis;
    basis_usage usage{};
    const basis_budget budget{10, 20, 30, 40};
    assert(try_consume(atom_input{1, 4, 5, 6, 7}, budget, usage));
    assert(!try_consume(atom_input{2, 7, 1, 1, 1}, budget, usage));
    assert(usage.storage_bytes == 4 && usage.mutation_cost == 7);
    basis_usage overflow{UINT64_MAX, 0, 0, 0};
    assert(!try_consume(atom_input{3, 1, 0, 0, 0}, basis_budget{UINT64_MAX, 0, 0, 0}, overflow));
    return 0;
}
