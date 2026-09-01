#include "CellShard/compiler/basis/membership.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::basis;
    const global_id bases[] = {100, 200, 200};
    const atom_membership atoms[] = {{20, 7, 0, 2}, {10, 7, 2, 1}};
    const membership_view view{atoms, 2, bases, 3};
    assert(valid_memberships(view));
    assert(canonical_redundant_atom(view, 7) == 10);
    return 0;
}
