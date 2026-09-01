#include <CellShard/compiler/partial/transform_partial_v1.hh>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>

namespace {

using namespace cellshard::compiler::partial;

transform_partial_view_v1 view(const transform_index_v1 *indices,
                               std::uint64_t count) {
    return {indices, count, {1, 1}, {1, 2}, {1, 3}, count, count, 7,
            transform_partial_kind_v1::permutation, 1};
}

void test_adversarial_contracts() {
    std::array<transform_index_v1, 3> indices{{{0, 2}, {1, 0}, {2, 1}}};
    std::array<std::uint8_t, 3> marks{};
    assert(validate_transform_partial_v1(
               view(indices.data(), indices.size()), marks.data(), marks.size())
               .valid());
    auto invalid = indices;
    invalid[2].destination_ordinal = invalid[1].destination_ordinal;
    assert(validate_transform_partial_v1(
               view(invalid.data(), invalid.size()), marks.data(), marks.size())
               .code == transform_partial_validation_code_v1::
                            duplicate_destination);
    invalid = indices;
    invalid[2].source_ordinal = 1;
    assert(validate_transform_partial_v1(
               view(invalid.data(), invalid.size()), marks.data(), marks.size())
               .code == transform_partial_validation_code_v1::
                            unordered_or_duplicate_source);
}

void test_randomized_differential_gather() {
    std::mt19937_64 generator(0x243f6a8885a308d3ULL);
    for (std::uint32_t trial = 0; trial < 1024; ++trial) {
        std::array<std::uint64_t, 32> permutation{};
        std::iota(permutation.begin(), permutation.end(), 0);
        std::shuffle(permutation.begin(), permutation.end(), generator);
        std::array<transform_index_v1, 32> indices{};
        for (std::size_t index = 0; index < indices.size(); ++index) {
            indices[index] = {index, permutation[index]};
        }
        std::array<std::uint8_t, 32> marks{};
        const auto transform = view(indices.data(), indices.size());
        assert(validate_transform_partial_v1(
                   transform, marks.data(), marks.size())
                   .valid());
        std::array<std::uint64_t, 32> gather{};
        assert(materialize_transform_gather_v1(
            transform, gather.data(), gather.size()));
        for (std::size_t source = 0; source < indices.size(); ++source) {
            assert(gather[permutation[source]] == source);
        }
    }
}

} // namespace

int main() {
    test_adversarial_contracts();
    test_randomized_differential_gather();
    return 0;
}
