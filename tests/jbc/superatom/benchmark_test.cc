#include "CellShard/compiler/composition/superatom/benchmark.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::composition::superatom;
    const promotion_policy policy{10, 100};
    const benchmark_report losing{9, 10, 12, 100, 0, 20, true};
    assert(evaluate_policy(losing, {9, true, true}, policy).outcome == policy_outcome::retain_atoms);
    const benchmark_report winning{9, 10, 5, 100, 20, 20, true};
    const auto result = evaluate_policy(winning, {9, true, true}, policy);
    assert(result.outcome == policy_outcome::promote && result.saved_ns == 480);
    assert(evaluate_policy(winning, {9, false, true}, policy).outcome == policy_outcome::invalid_evidence);
    std::uint64_t state = 3;
    for (unsigned i = 0; i < 1000; ++i) {
        state = state * UINT64_C(2862933555777941757) + UINT64_C(3037000493);
        const std::uint64_t baseline = state & 1023U;
        const std::uint64_t promoted = (state >> 10U) & 1023U;
        const auto observed = evaluate_policy({7, baseline, promoted, 2, 0, 0, true}, {7, true, true}, {0, UINT64_MAX});
        assert((observed.outcome == policy_outcome::promote) == (promoted < baseline));
    }
    return 0;
}
