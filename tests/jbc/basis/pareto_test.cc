#include "CellShard/compiler/basis/pareto.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::basis;
    basis_point portfolio[4]{}; local_index count = 0;
    assert(pareto_insert({10, 5, {5, 5, 5, 5}}, portfolio, 4, count));
    assert(!pareto_insert({20, 4, {6, 6, 6, 6}}, portfolio, 4, count));
    assert(pareto_insert({30, 8, {7, 4, 4, 4}}, portfolio, 4, count));
    assert(count == 2);
    assert(pareto_insert({5, 5, {5, 5, 5, 5}}, portfolio, 4, count));
    assert(count == 2 && portfolio[0].basis_id == 5);
    return 0;
}
