#include "CellShard/compiler/composition/superatom/membership.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::composition::superatom;
    const global_id bases[] = {10, 20, 30};
    const lineage_membership records[] = {{100, 100, 0, 0, 0, 2}, {101, 100, 100, 1, 2, 1}};
    const lineage_view view{records, 2, bases, 3};
    assert(validate_lineage(view));
    assert(belongs_to_basis(view, 0, 20));
    assert(!belongs_to_basis(view, 1, 20));
    return 0;
}
