#include "baselines.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    const std::uint64_t frequency[] = {2, 5, 5, 1}; std::uint32_t order[4]{};
    baseline_order(frequency, 4, baseline_kind::frequency_only, order);
    assert(order[0] == 1 && order[1] == 2 && order[3] == 3);
    baseline_order(frequency, 4, baseline_kind::identity_csr, order);
    for (std::uint32_t i = 0; i < 4; ++i) assert(order[i] == i);
    return 0;
}
