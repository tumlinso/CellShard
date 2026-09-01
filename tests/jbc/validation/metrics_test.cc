#include "metrics.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    metric_record record{corpus[0].fixture_id, mechanism::modular_support, 10, 20, 8, 1, complete_phase_mask, true, false};
    assert(complete_metric(record)); record.included_phases &= static_cast<std::uint16_t>(~UINT16_C(1));
    assert(!complete_metric(record)); return 0;
}
