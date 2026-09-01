#include "CellShard/compiler/basis/promotion.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::basis;
    const promotion_policy policy{1, 100, true};
    solver_report reports[] = {{20, 200, 9, 50, true}, {10, 100, 10, 70, true}};
    independent_basis_evidence evidence[] = {{200, {}, 9, 10, false, true}, {100, {}, 10, 10, true, true}};
    assert(evaluate_promotion(reports[0], evidence[0], policy) == promotion_decision::independently_infeasible);
    assert(select_promoted(reports, evidence, 2, policy) == 1);
    // Deterministic adversarial sweep: a provider claim can never replace independent utility.
    std::uint64_t state = 1;
    for (unsigned i = 0; i < 1000; ++i) {
        state = state * UINT64_C(6364136223846793005) + 1;
        solver_report report{1, 9, state, 1, true};
        independent_basis_evidence proof{9, {}, state ^ 1U, UINT64_MAX, true, true};
        assert(evaluate_promotion(report, proof, {UINT64_MAX, UINT64_MAX, true}) == promotion_decision::utility_mismatch);
    }
    return 0;
}
