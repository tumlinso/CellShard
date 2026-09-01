#include "CellShard/compiler/composition/superatom/lifecycle.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::composition::superatom;
    superatom_record record{{10, 20, 30, 40}, 10, 3, lifecycle_state::promoted};
    assert(demote(record, {{1, 9, false}, false, false}) == transition_result::not_verified);
    assert(demote(record, {{1, 9, false}, false, true}) == transition_result::applied);
    assert(record.state == lifecycle_state::demoted && record.generation == 4);
    return 0;
}
