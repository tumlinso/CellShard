#include <CellShard/compiler/partial/extrema_state_v1.hh>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

using namespace cellshard::compiler::partial;

extrema_state_v1 state(double minimum, std::uint64_t min_id,
                       double maximum, std::uint64_t max_id,
                       std::uint64_t count) {
    return {{minimum, {2, min_id}, min_id},
            {maximum, {2, max_id}, max_id}, count,
            {1, 1}, {1, 2}, {1, 3}, 7, 1, 0};
}

void test_adversarial_contracts_and_ties() {
    auto left = state(-2.0, 3, 5.0, 8, 2);
    auto right = state(-2.0, 1, 5.0, 4, 2);
    const auto merged = merge_extrema_states_v1(left, right);
    assert(merged.valid());
    assert(merged.state.minimum.biological_identity.local_identity == 1);
    assert(merged.state.maximum.biological_identity.local_identity == 4);
    right.value_generation = 8;
    assert(merge_extrema_states_v1(left, right).code
           == extrema_state_code_v1::incompatible_contract);
    left.minimum.value = std::numeric_limits<double>::quiet_NaN();
    assert(validate_extrema_state_v1(left)
           == extrema_state_code_v1::nonfinite_value);
}

void test_randomized_differential_merge_tree() {
    std::mt19937_64 generator(0xa4093822299f31d0ULL);
    std::uniform_real_distribution<double> distribution(-1000.0, 1000.0);
    for (std::uint32_t trial = 0; trial < 512; ++trial) {
        std::vector<extrema_state_v1> states;
        double expected_min = std::numeric_limits<double>::infinity();
        double expected_max = -std::numeric_limits<double>::infinity();
        for (std::uint64_t index = 1; index <= 64; ++index) {
            const double value = distribution(generator);
            expected_min = std::min(expected_min, value);
            expected_max = std::max(expected_max, value);
            states.push_back(state(value, index, value, index, 1));
        }
        while (states.size() > 1) {
            std::shuffle(states.begin(), states.end(), generator);
            const auto merged = merge_extrema_states_v1(states.back(),
                                                        states[states.size() - 2]);
            assert(merged.valid());
            states.pop_back();
            states.back() = merged.state;
        }
        assert(states.front().minimum.value == expected_min);
        assert(states.front().maximum.value == expected_max);
        assert(states.front().contribution_count == 64);
    }
}

} // namespace

int main() {
    test_adversarial_contracts_and_ties();
    test_randomized_differential_merge_tree();
    return 0;
}
