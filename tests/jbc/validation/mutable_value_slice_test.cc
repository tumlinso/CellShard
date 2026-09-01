#include "vertical_slice.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    value_rebind_evidence evidence{UINT64_C(0x100000001), UINT64_C(0x100000001), 4, 5,
                                   {1,2,3,4}, {1,2,3,4}, false};
    assert(valid_value_rebind(evidence)); evidence.rebuilt_structure = true; assert(!valid_value_rebind(evidence));
    return 0;
}
