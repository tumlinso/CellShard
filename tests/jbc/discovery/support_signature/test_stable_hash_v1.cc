#include <CellShard/compiler/discovery/support_signature/stable_hash_v1.hh>

#include <array>
#include <cassert>

namespace signature =
    cellshard::compiler::discovery::support_signature;

int main() {
    const std::uint64_t destinations[] = {10, 20};
    const std::uint64_t offsets[] = {0, 2, 5};
    const std::uint64_t sources[] = {1, 3, 2, 4, 8};
    signature::exact_destination_support_view_v1 view{
        destinations, offsets, sources, 2, 5, {1, 1}, {2, 1}, {3, 1}, 4};
    std::array<signature::destination_support_hash_v1, 2> first{};
    std::array<signature::destination_support_hash_v1, 2> second{};
    auto result = signature::hash_exact_destination_support_v1(
        view, first.data(), first.size());
    assert(result.hashed() && result.hash_count == 2);
    result = signature::hash_exact_destination_support_v1(
        view, second.data(), second.size());
    assert(result.hashed());
    assert(first[0].hash.low == second[0].hash.low);
    assert(first[0].hash.high == second[0].hash.high);
    view.relation_generation = 5;
    result = signature::hash_exact_destination_support_v1(
        view, second.data(), second.size());
    assert(result.hashed());
    assert(first[0].hash.low != second[0].hash.low
           || first[0].hash.high != second[0].hash.high);
    view.destination_offsets = nullptr;
    assert(signature::hash_exact_destination_support_v1(
               view, second.data(), second.size()).code
           == signature::stable_support_hash_code_v1::invalid_view);
}
