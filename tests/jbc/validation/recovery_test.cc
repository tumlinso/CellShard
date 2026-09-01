#include "recovery.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    recovery_evidence before{1, 4, 5, 4, {1,2,3,4}, {1,2,3,4}, false, true, false};
    assert(valid_recovery(before));
    recovery_evidence after{1, 4, 5, 5, {1,2,3,4}, {5,6,7,8}, true, true, false};
    assert(valid_recovery(after)); after.partial_generation_visible = true; assert(!valid_recovery(after));
    return 0;
}
