#include <CellShard/compiler/discovery/support_signature/minhash_v1.hh>

#include <array>
#include <cassert>

namespace signature =
    cellshard::compiler::discovery::support_signature;

int main() {
    const std::uint64_t destinations[] = {10, 20};
    const std::uint64_t offsets[] = {0, 3, 6};
    const std::uint64_t sources[] = {1, 2, 3, 1, 2, 4};
    const signature::exact_destination_support_view_v1 support{
        destinations, offsets, sources, 2, 6, {1, 1}, {2, 1}, {3, 1}, 4};
    std::array<std::uint64_t, 64> first{};
    std::array<std::uint64_t, 64> second{};
    auto result = signature::build_minhash_v1(
        support, 32, 64, 99, first.data(), first.size());
    assert(result.built() && result.required_minima == 64);
    result = signature::build_minhash_v1(
        support, 32, 64, 99, second.data(), second.size());
    assert(result.built() && first == second);
    std::uint32_t equal = 0;
    for (std::uint32_t index = 0; index < 32; ++index) {
        if (first[index] == first[32 + index]) ++equal;
    }
    assert(equal > 0 && equal < 32);
    result = signature::build_minhash_v1(
        support, 32, 64, 100, second.data(), second.size());
    assert(result.built() && first != second);
    assert(!signature::authorizes_execution(result.view));
    assert(signature::build_minhash_v1(
               support, 65, 64, 99, first.data(), first.size()).code
           == signature::deterministic_minhash_code_v1::invalid_sketch_size);
}
