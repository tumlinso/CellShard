#include <CellShard/compiler/partial/moments_state_v1.hh>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

using namespace cellshard::compiler::partial;

moments_state_result_v1 make(const double *values, std::uint64_t count) {
    return make_moments_state_v1(values, count, {1, 1}, {1, 2}, {1, 3}, 7);
}

void test_adversarial_contracts() {
    const double values[]{1.0e12 + 1.0, 1.0e12 + 2.0, 1.0e12 + 3.0};
    const auto result = make(values, 3);
    assert(result.valid());
    assert(std::fabs(result.state.mean - (1.0e12 + 2.0)) < 1.0e-6);
    assert(std::fabs(population_variance_v1(result.state) - (2.0 / 3.0))
           < 1.0e-12);
    auto invalid = result.state;
    invalid.centered_sum_squares = -1.0;
    assert(validate_moments_state_v1(invalid)
           == moments_state_code_v1::invalid_numeric_state);
}

void test_randomized_long_double_differential() {
    std::mt19937_64 generator(0x452821e638d01377ULL);
    std::uniform_real_distribution<double> distribution(-1.0e5, 1.0e5);
    for (std::uint32_t trial = 0; trial < 512; ++trial) {
        std::vector<double> values(257);
        long double sum = 0.0L;
        for (auto &value : values) {
            value = distribution(generator);
            sum += value;
        }
        const long double mean = sum / values.size();
        long double m2 = 0.0L;
        for (const auto value : values) {
            const long double delta = value - mean;
            m2 += delta * delta;
        }
        const auto left = make(values.data(), 113);
        const auto right = make(values.data() + 113, values.size() - 113);
        const auto merged = merge_moments_states_v1(left.state, right.state);
        assert(merged.valid());
        assert(std::fabs(static_cast<long double>(merged.state.mean) - mean)
               < 1.0e-9L);
        const long double variance_error = std::fabs(
            static_cast<long double>(population_variance_v1(merged.state))
            - m2 / values.size());
        assert(variance_error < 1.0e-5L);
    }
}

} // namespace

int main() {
    test_adversarial_contracts();
    test_randomized_long_double_differential();
    return 0;
}
