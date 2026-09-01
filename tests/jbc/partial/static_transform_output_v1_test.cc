#include <CellShard/compiler/partial/static_transform_output_v1.hh>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>

namespace {
using namespace cellshard::compiler::partial;

void test_randomized_differential_output() {
    std::mt19937_64 generator(0x9216d5d98979fb1bULL);
    for (std::uint32_t trial = 0; trial < 1024; ++trial) {
        std::array<double, 32> source{};
        std::array<std::uint64_t, 32> permutation{};
        std::iota(permutation.begin(), permutation.end(), 0);
        std::shuffle(permutation.begin(), permutation.end(), generator);
        std::array<transform_index_v1, 32> map{};
        for (std::size_t i = 0; i < source.size(); ++i) {
            source[i] = double(i) + 0.25;
            map[i] = {i, permutation[i]};
        }
        transform_partial_view_v1 transform{map.data(), map.size(), {1, 1},
            {1, 2}, {1, 3}, 32, 32, 7, transform_partial_kind_v1::permutation, 1};
        std::array<double, 32> output{};
        assert(materialize_static_transform_output_v1(source.data(), source.size(),
                   transform, output.data(), output.size()).valid());
        for (std::size_t i = 0; i < source.size(); ++i)
            assert(output[permutation[i]] == source[i]);
        static_transform_output_view_v1 view{output.data(), output.size(), {1, 4},
            {1, 1}, {1, 2}, {1, 3}, {1, 5}, 7, 8, 9, 1, 0};
        std::array<std::uint8_t, 32> marks{};
        assert(validate_static_transform_output_v1(
                   view, transform, marks.data(), marks.size()).valid());
        view.source_value_generation = 0;
        assert(validate_static_transform_output_v1(
                   view, transform, marks.data(), marks.size()).code
               == static_transform_output_code_v1::missing_generation);
    }
}
}

int main() {
    test_randomized_differential_output();
    return 0;
}
