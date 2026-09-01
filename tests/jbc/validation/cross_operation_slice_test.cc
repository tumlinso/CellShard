#include "vertical_slice.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    const metric_record metric{2, mechanism::modular_support, 20, 30, 10, 2, complete_phase_mask, true, false};
    slice_evidence evidence{slice_kind::cross_operation_support_family, 2, UINT64_C(0x200000001), 4,
                            UINT64_C(0x300000001), UINT64_C(0x400000001), 3,
                            {5,6,7,8}, {5,6,7,8}, metric, true};
    assert(valid_slice(evidence)); evidence.operation_count = 0; assert(!valid_slice(evidence));
    return 0;
}
