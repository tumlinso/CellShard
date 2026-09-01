#include "vertical_slice.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    const metric_record metric{1, mechanism::modular_support, 10, 20, 8, 1, complete_phase_mask, true, false};
    slice_evidence evidence{slice_kind::support_signature_basis, 1, UINT64_C(0x100000001), 1, 8, 9, 1, {1,2,3,4}, {1,2,3,4}, metric, true};
    assert(valid_slice(evidence)); evidence.independent_reference = false; assert(!valid_slice(evidence));
    return 0;
}
