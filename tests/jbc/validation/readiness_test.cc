#include "readiness.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    novelty_evidence evidence{};
    assert(audit_novelty(evidence) == readiness::pending_integration);
    evidence = {true, true, true, true, true, true, true, true, 100, 100};
    assert(audit_novelty(evidence) == readiness::no_biological_specificity);
    evidence.biological_saved_ns = 101;
    assert(audit_novelty(evidence) == readiness::ready);
    return 0;
}
