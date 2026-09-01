#include <CellShard/compiler/discovery/support_signature/evidence_image_v1.hh>

#include <array>
#include <cassert>

namespace signature =
    cellshard::compiler::discovery::support_signature;
namespace evidence = cellshard::compiler::evidence;

int main() {
    const std::uint64_t destinations[] = {10, 20};
    const std::uint64_t offsets[] = {0, 2, 4};
    const std::uint64_t sources[] = {1, 2, 2, 3};
    const signature::exact_destination_support_view_v1 support{
        destinations, offsets, sources, 2, 4, {1, 1}, {2, 1}, {3, 1}, 4};
    const std::uint64_t minima[] = {11, 12, 13, 14, 21, 22, 23, 24};
    const signature::deterministic_minhash_view_v1 sketch{
        minima, 2, 4, 0, 99, {1, 1}, 4};
    const evidence::negative_evidence_v1 negatives[] = {
        {{5, 1}, {6, 1}, {7, 1}, 4, 10, 0,
         evidence::negative_evidence_reason_v1::not_observed, 0},
        {{5, 2}, {6, 2}, {7, 1}, 4, 10, 1,
         evidence::negative_evidence_reason_v1::contradicted, 0}};
    std::array<std::byte, 400> first{};
    std::array<std::byte, 400> second{};
    auto result = signature::build_support_evidence_image_v1(
        support, sketch, negatives, 2, first.data(), first.size());
    assert(result.built());
    const auto image_bytes = result.image_bytes;
    result = signature::build_support_evidence_image_v1(
        support, sketch, negatives, 2, second.data(), second.size());
    assert(result.built());
    for (std::uint64_t index = 0; index < image_bytes; ++index)
        assert(first[index] == second[index]);
    result = signature::validate_support_evidence_image_v1(
        first.data(), image_bytes);
    assert(result.valid() && result.index == 2);
    assert(!signature::authorizes_execution(result));
    first[0] = std::byte{0};
    assert(signature::validate_support_evidence_image_v1(
               first.data(), image_bytes).code
           == signature::support_evidence_image_code_v1::invalid_magic);
}
