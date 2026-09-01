#include <CellShard/compiler/partial/log_sum_exp_state_v1.hh>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

using namespace cellshard::compiler::partial;

log_sum_exp_result_v1 make(const double *values, std::uint64_t count) {
    return make_log_sum_exp_state_v1(values, count, {1, 1}, {1, 2}, {1, 3}, 7);
}

void test_extremes_and_invalid_inputs() {
    const double values[]{1000.0, 999.0, -1000.0};
    const auto result = make(values, 3);
    assert(result.valid());
    const double expected = 1000.0 + std::log1p(std::exp(-1.0));
    assert(std::fabs(finalize_log_sum_exp_state_v1(result.state) - expected)
           < 1.0e-12);
    const double negatives[]{-std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()};
    const auto all_negative = make(negatives, 2);
    assert(all_negative.valid());
    assert(finalize_log_sum_exp_state_v1(all_negative.state)
           == -std::numeric_limits<double>::infinity());
    const double invalid[]{std::numeric_limits<double>::quiet_NaN()};
    assert(make(invalid, 1).code == log_sum_exp_code_v1::invalid_numeric_state);
}

void test_randomized_differential_merge() {
    std::mt19937_64 generator(0x082efa98ec4e6c89ULL);
    std::uniform_real_distribution<double> distribution(-700.0, 700.0);
    for (std::uint32_t trial = 0; trial < 512; ++trial) {
        std::vector<double> values(129);
        for (auto &value : values) value = distribution(generator);
        const auto left = make(values.data(), 47);
        const auto right = make(values.data() + 47, values.size() - 47);
        const auto merged = merge_log_sum_exp_states_v1(left.state, right.state);
        const auto direct = make(values.data(), values.size());
        assert(merged.valid() && direct.valid());
        assert(std::fabs(finalize_log_sum_exp_state_v1(merged.state)
                         - finalize_log_sum_exp_state_v1(direct.state))
               < 1.0e-12);
    }
}

} // namespace

int main() {
    test_extremes_and_invalid_inputs();
    test_randomized_differential_merge();
    return 0;
}
