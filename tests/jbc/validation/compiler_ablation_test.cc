#include "compiler_ablation.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    assert(compiler_ablation_matrix.size() == 6);
    assert(layer_enabled(compiler_ablation_matrix[1], compiler_layer::superatom));
    assert(!layer_enabled(compiler_ablation_matrix[5], compiler_layer::superatom));
    metric_record metric{1, mechanism::repeated_composition, 1, 1, 1, 1, complete_phase_mask, true, false};
    assert(comparable(compiler_ablation_result{metric, 1, 1}, compiler_ablation_result{metric, 2, 2}));
    return 0;
}
