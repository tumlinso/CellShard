#include "CellShard/compiler/composition/superatom/lifecycle.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::composition::superatom;
    superatom_record record{{10, 20, 30, 40}, 0, 0, lifecycle_state::candidate};
    promotion_evidence provider_only{20, 30, {10, 1, false}, false};
    assert(promote(record, provider_only) == transition_result::not_verified);
    provider_only.composition_verified = true;
    assert(promote(record, provider_only) == transition_result::applied);
    assert(record.state == lifecycle_state::promoted && record.lineage_id == 10);
    return 0;
}
