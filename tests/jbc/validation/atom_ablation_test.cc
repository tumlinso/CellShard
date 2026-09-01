#include "atom_ablation.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    metric_record metric{corpus[0].fixture_id, mechanism::modular_support, 10, 20, 8, 1, complete_phase_mask, true, false};
    atom_ablation_result baseline{metric, 10, 4}, treatment{metric, 8, 5};
    assert(atom_ablation_matrix.size() == 6 && comparable(baseline, treatment));
    treatment.accepted_count = 9; assert(!comparable(baseline, treatment)); return 0;
}
