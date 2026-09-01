#include <CellShard/compiler/partial/additive_state_v1.hh>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

using namespace cellshard::compiler::partial;

additive_state_result_v1 make(const double *values, std::uint64_t count) {
    return make_additive_state_v1(
        values, count, {1, 1}, {1, 2}, {1, 3}, 7);
}

void test_adversarial_cancellation() {
    const std::array<double, 3> values{1.0e16, 1.0, -1.0e16};
    const auto state = make(values.data(), values.size());
    assert(state.valid());
    assert(finalize_additive_state_v1(state.state) == 1.0);
    auto invalid = state.state;
    invalid.value_generation = 0;
    assert(validate_additive_state_v1(invalid)
           == additive_state_code_v1::missing_generation);
    invalid = state.state;
    invalid.sum = std::numeric_limits<double>::quiet_NaN();
    assert(validate_additive_state_v1(invalid)
           == additive_state_code_v1::nonfinite_state);
}

void test_merge_contracts() {
    const std::array<double, 2> left_values{1.0, 2.0};
    const std::array<double, 2> right_values{3.0, 4.0};
    const auto left = make(left_values.data(), left_values.size());
    auto right = make(right_values.data(), right_values.size());
    auto merged = merge_additive_states_v1(left.state, right.state);
    assert(merged.valid());
    assert(finalize_additive_state_v1(merged.state) == 10.0);
    right.state.numerical_policy_identity = {9, 9};
    assert(merge_additive_states_v1(left.state, right.state).code
           == additive_state_code_v1::incompatible_contract);
}

void test_randomized_long_double_differential() {
    std::mt19937_64 generator(0x13198a2e03707344ULL);
    std::uniform_real_distribution<double> distribution(-1.0e8, 1.0e8);
    for (std::uint32_t trial = 0; trial < 512; ++trial) {
        std::vector<double> values(257);
        long double reference = 0.0L;
        for (auto &value : values) {
            value = distribution(generator);
            reference += static_cast<long double>(value);
        }
        const auto left = make(values.data(), 113);
        const auto right = make(values.data() + 113, values.size() - 113);
        const auto merged = merge_additive_states_v1(left.state, right.state);
        assert(merged.valid());
        const long double error = std::fabs(
            static_cast<long double>(finalize_additive_state_v1(merged.state))
            - reference);
        assert(error <= 1.0e-6L);
    }
}

} // namespace

int main() {
    test_adversarial_cancellation();
    test_merge_contracts();
    test_randomized_long_double_differential();
    return 0;
}
