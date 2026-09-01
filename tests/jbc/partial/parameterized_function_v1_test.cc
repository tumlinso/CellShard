#include <CellShard/compiler/partial/parameterized_function_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <random>

namespace {
using namespace cellshard::compiler::partial;
alignas(16) std::array<std::byte, 32> state{};

parameterized_partial_function_view_v1 function() {
    return {state.data(), state.size(), 16, 0, {1, 1}, {1, 2}, {1, 3},
            {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8}, 9, 10, 11, 12, 1, 0};
}

void test_exact_parameter_binding() {
    const auto f = function();
    assert(validate_parameterized_partial_function_v1(f).valid());
    parameter_binding_v1 binding{{1, 3}, {1, 4}, 11};
    assert(validate_parameter_binding_v1(f, binding).valid());
    binding.parameter_generation = 12;
    assert(validate_parameter_binding_v1(f, binding).code
           == parameterized_function_code_v1::parameter_generation_mismatch);
    binding = {{1, 3}, {9, 9}, 11};
    assert(validate_parameter_binding_v1(f, binding).code
           == parameterized_function_code_v1::parameter_content_mismatch);
}

void test_randomized_generation_fail_closed() {
    std::mt19937_64 generator(0x2ffd72dbd01adfb7ULL);
    const auto f = function();
    for (std::uint32_t trial = 0; trial < 4096; ++trial) {
        const std::uint64_t generation = 1 + generator() % 32;
        const parameter_binding_v1 binding{{1, 3}, {1, 4}, generation};
        assert(validate_parameter_binding_v1(f, binding).valid()
               == (generation == f.parameter_generation));
    }
}
}

int main() {
    test_exact_parameter_binding();
    test_randomized_generation_fail_closed();
    return 0;
}
