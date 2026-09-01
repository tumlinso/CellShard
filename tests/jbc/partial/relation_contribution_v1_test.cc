#include <CellShard/compiler/partial/relation_contribution_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using namespace cellshard::compiler::partial;

relation_contribution_v1 item(std::uint64_t edge) {
    return {{2, edge}, {3, edge % 7 + 1}, edge % 7, double(edge)};
}

relation_contribution_view_v1 view(const relation_contribution_v1 *items,
                                   std::uint64_t count) {
    return {items, count, {1, 1}, {1, 2}, {1, 3}, {1, 4}, 7, 8, 1, 0};
}

void test_adversarial_failures() {
    std::array<relation_contribution_v1, 2> items{item(1), item(2)};
    assert(validate_relation_contributions_v1(view(items.data(), items.size()))
               .valid());
    auto invalid = items;
    invalid[1].logical_edge_identity = invalid[0].logical_edge_identity;
    assert(validate_relation_contributions_v1(view(invalid.data(), invalid.size()))
               .code == relation_contribution_code_v1::
                            unordered_or_duplicate_edge);
    std::array<relation_contribution_v1, 4> output{};
    assert(merge_relation_contributions_v1(
               view(items.data(), items.size()), view(items.data(), items.size()),
               output.data(), output.size())
               .code == relation_contribution_code_v1::
                            duplicate_edge_contribution);
}

void test_randomized_differential_union() {
    std::mt19937_64 generator(0xc0ac29b7c97c50ddULL);
    for (std::uint32_t trial = 0; trial < 512; ++trial) {
        std::vector<relation_contribution_v1> left, right, expected;
        for (std::uint64_t edge = 1; edge <= 100; ++edge) {
            if ((generator() & 1U) == 0) left.push_back(item(edge));
            else right.push_back(item(edge));
            expected.push_back(item(edge));
        }
        std::vector<relation_contribution_v1> output(expected.size());
        const auto result = merge_relation_contributions_v1(
            view(left.data(), left.size()), view(right.data(), right.size()),
            output.data(), output.size());
        assert(result.valid() && result.output_count == expected.size());
        for (std::size_t index = 0; index < expected.size(); ++index) {
            assert(output[index].logical_edge_identity
                   == expected[index].logical_edge_identity);
        }
    }
}

} // namespace

int main() {
    test_adversarial_failures();
    test_randomized_differential_union();
    return 0;
}
