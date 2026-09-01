#include <CellShard/compiler/partial/additive_state_v1.hh>
#include <CellShard/compiler/partial/extrema_state_v1.hh>
#include <CellShard/compiler/partial/gathered_panel_v1.hh>
#include <CellShard/compiler/partial/gradient_partial_v1.hh>
#include <CellShard/compiler/partial/log_sum_exp_state_v1.hh>
#include <CellShard/compiler/partial/moments_state_v1.hh>
#include <CellShard/compiler/partial/parameterized_function_v1.hh>
#include <CellShard/compiler/partial/partial_image_v1.hh>
#include <CellShard/compiler/partial/promotion_v1.hh>
#include <CellShard/compiler/partial/relation_contribution_v1.hh>
#include <CellShard/compiler/partial/segment_summary_v1.hh>
#include <CellShard/compiler/partial/static_transform_output_v1.hh>
#include <CellShard/compiler/partial/structural_partial_v1.hh>
#include <CellShard/compiler/partial/trajectory_prefix_v1.hh>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

using namespace cellshard::compiler::partial;

constexpr cellshard::compiler::atom::atom_persistent_identity_v1 algebra{1, 1};
constexpr cellshard::compiler::atom::atom_persistent_identity_v1 policy{1, 2};
constexpr cellshard::compiler::atom::atom_persistent_identity_v1 order{1, 3};

template <class State, class Merge>
State random_adjacent_tree(std::vector<State> states,
                           std::mt19937_64 &generator, Merge merge) {
    while (states.size() > 1) {
        const std::size_t index = generator() % (states.size() - 1);
        states[index] = merge(states[index], states[index + 1]);
        states.erase(states.begin() + static_cast<std::ptrdiff_t>(index + 1));
    }
    return states.front();
}

std::vector<std::pair<std::size_t, std::size_t>> random_chunks(
    std::size_t count, std::mt19937_64 &generator) {
    std::vector<std::pair<std::size_t, std::size_t>> chunks;
    std::size_t begin = 0;
    while (begin < count) {
        const std::size_t size = std::min<std::size_t>(
            1 + generator() % 31, count - begin);
        chunks.push_back({begin, size});
        begin += size;
    }
    return chunks;
}

void validate_numeric_merge_trees() {
    std::mt19937_64 generator(0x6a09e667f3bcc909ULL);
    std::uniform_real_distribution<double> distribution(-1.0e6, 1.0e6);
    for (std::uint32_t trial = 0; trial < 256; ++trial) {
        std::vector<double> values(513);
        long double sum_reference = 0.0L;
        long double mean_reference = 0.0L;
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();
        for (auto &value : values) {
            value = distribution(generator);
            sum_reference += value;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        mean_reference = sum_reference / values.size();
        long double m2_reference = 0.0L;
        long double exp_reference = 0.0L;
        for (const auto value : values) {
            const long double centered = value - mean_reference;
            m2_reference += centered * centered;
            exp_reference += std::exp(static_cast<long double>(value - maximum));
        }
        const long double lse_reference = maximum + std::log(exp_reference);

        const auto chunks = random_chunks(values.size(), generator);
        std::vector<additive_state_v1> additive;
        std::vector<moments_state_v1> moments;
        std::vector<log_sum_exp_state_v1> lse;
        std::vector<extrema_state_v1> extrema;
        for (const auto &chunk : chunks) {
            const double *begin = values.data() + chunk.first;
            const auto a = make_additive_state_v1(
                begin, chunk.second, algebra, policy, order, 7);
            const auto m = make_moments_state_v1(
                begin, chunk.second, algebra, policy, order, 7);
            const auto l = make_log_sum_exp_state_v1(
                begin, chunk.second, algebra, policy, order, 7);
            assert(a.valid() && m.valid() && l.valid());
            additive.push_back(a.state);
            moments.push_back(m.state);
            lse.push_back(l.state);
            const auto local_min = std::min_element(begin, begin + chunk.second);
            const auto local_max = std::max_element(begin, begin + chunk.second);
            const auto local_min_offset =
                static_cast<std::size_t>(local_min - begin);
            const auto local_max_offset =
                static_cast<std::size_t>(local_max - begin);
            extrema.push_back({
                {*local_min, {2, chunk.first + local_min_offset + 1},
                 chunk.first + local_min_offset},
                {*local_max, {2, chunk.first + local_max_offset + 1},
                 chunk.first + local_max_offset},
                chunk.second, algebra, policy, order, 7, 1, 0});
        }

        const auto additive_result = random_adjacent_tree(
            additive, generator, [](const auto &left, const auto &right) {
                const auto result = merge_additive_states_v1(left, right);
                assert(result.valid());
                return result.state;
            });
        const auto moments_result = random_adjacent_tree(
            moments, generator, [](const auto &left, const auto &right) {
                const auto result = merge_moments_states_v1(left, right);
                assert(result.valid());
                return result.state;
            });
        const auto lse_result = random_adjacent_tree(
            lse, generator, [](const auto &left, const auto &right) {
                const auto result = merge_log_sum_exp_states_v1(left, right);
                assert(result.valid());
                return result.state;
            });
        const auto extrema_result = random_adjacent_tree(
            extrema, generator, [](const auto &left, const auto &right) {
                const auto result = merge_extrema_states_v1(left, right);
                assert(result.valid());
                return result.state;
            });

        assert(std::fabs(static_cast<long double>(
                             finalize_additive_state_v1(additive_result))
                         - sum_reference)
               < 1.0e-6L);
        assert(std::fabs(static_cast<long double>(moments_result.mean)
                         - mean_reference)
               < 1.0e-9L);
        assert(std::fabs(
                   static_cast<long double>(moments_result.centered_sum_squares)
                   - m2_reference)
               < 0.25L);
        assert(std::fabs(static_cast<long double>(
                             finalize_log_sum_exp_state_v1(lse_result))
                         - lse_reference)
               < 1.0e-9L);
        assert(extrema_result.minimum.value == minimum);
        assert(extrema_result.maximum.value == maximum);
        assert(additive_result.contribution_count == values.size());
        assert(moments_result.contribution_count == values.size());
        assert(lse_result.contribution_count == values.size());
        assert(extrema_result.contribution_count == values.size());
    }
}

void validate_generation_interlocks() {
    const double values[]{1.0, 2.0};
    const auto first = make_additive_state_v1(
        values, 2, algebra, policy, order, 7).state;
    auto stale = first;
    stale.value_generation = 8;
    assert(merge_additive_states_v1(first, stale).code
           == additive_state_code_v1::incompatible_contract);
    const auto moments = make_moments_state_v1(
        values, 2, algebra, policy, order, 7).state;
    auto stale_moments = moments;
    stale_moments.value_generation = 8;
    assert(merge_moments_states_v1(moments, stale_moments).code
           == moments_state_code_v1::incompatible_contract);
}

} // namespace

int main() {
    validate_numeric_merge_trees();
    validate_generation_interlocks();
    return 0;
}
