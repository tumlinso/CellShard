#include <CellShard/compiler/partial/trajectory_prefix_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <random>

namespace {
using namespace cellshard::compiler::partial;

alignas(16) std::array<std::byte, 16> state{};

trajectory_prefix_state_view_v1 prefix(std::uint64_t begin,
                                       std::uint64_t end) {
    return {state.data(), state.size(), 16, 0, {1, 1}, {2, begin + 1},
            {2, end + 1}, {1, 2}, {1, 3}, {1, 4}, begin, end,
            7, 8, 9, 1, 0};
}

void test_adversarial_composition() {
    auto left = prefix(0, 5);
    auto right = prefix(5, 9);
    right.begin_node_identity = left.end_node_identity;
    assert(validate_trajectory_prefix_composition_v1(left, right).valid());
    right.begin_step = 6;
    assert(validate_trajectory_prefix_composition_v1(left, right).code
           == trajectory_prefix_code_v1::noncontiguous_prefix);
    right = prefix(5, 9);
    right.begin_node_identity = {9, 9};
    assert(validate_trajectory_prefix_composition_v1(left, right).code
           == trajectory_prefix_code_v1::node_boundary_mismatch);
    left.state_generation = 0;
    assert(validate_trajectory_prefix_v1(left).code
           == trajectory_prefix_code_v1::missing_generation);
}

void test_randomized_exact_boundaries() {
    std::mt19937_64 generator(0xd1310ba698dfb5acULL);
    for (std::uint32_t trial = 0; trial < 4096; ++trial) {
        const std::uint64_t split = 1 + generator() % 1000;
        auto left = prefix(0, split);
        auto right = prefix(split, split + 1 + generator() % 1000);
        right.begin_node_identity = left.end_node_identity;
        assert(validate_trajectory_prefix_composition_v1(left, right).valid());
    }
}
}

int main() {
    test_adversarial_composition();
    test_randomized_exact_boundaries();
    return 0;
}
