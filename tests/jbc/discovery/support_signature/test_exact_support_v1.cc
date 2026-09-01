#include <CellShard/compiler/discovery/support_signature/exact_support_v1.hh>

#include <algorithm>
#include <array>
#include <cassert>
#include <random>

namespace signature =
    cellshard::compiler::discovery::support_signature;

int main() {
    std::array<signature::destination_source_edge_v1, 6> edges{{
        {20, 5}, {10, 3}, {20, 1}, {10, 2}, {30, 9}, {20, 4}}};
    std::array<signature::destination_source_edge_v1, 6> scratch{};
    std::array<std::uint64_t, 6> destinations{};
    std::array<std::uint64_t, 7> offsets{};
    std::array<std::uint64_t, 6> sources{};
    const signature::exact_destination_support_buffers_v1 buffers{
        scratch.data(), scratch.size(), destinations.data(),
        destinations.size(), offsets.data(), offsets.size(), sources.data(),
        sources.size()};
    auto build = [&] {
        return signature::build_exact_destination_support_v1(
            edges.data(), edges.size(), {1, 1}, {2, 1}, {3, 1}, 4,
            buffers);
    };
    auto result = build();
    assert(result.built() && result.view.destination_count == 3);
    assert((destinations[0] == 10 && destinations[1] == 20
            && destinations[2] == 30));
    assert((offsets[0] == 0 && offsets[1] == 2 && offsets[2] == 5
            && offsets[3] == 6));
    assert((sources[0] == 2 && sources[1] == 3 && sources[2] == 1
            && sources[3] == 4 && sources[4] == 5 && sources[5] == 9));
    assert(!signature::authorizes_execution(result.view));
    std::mt19937_64 random(0x535550504f5254ULL);
    for (std::uint32_t trial = 0; trial < 200; ++trial) {
        std::shuffle(edges.begin(), edges.end(), random);
        result = build();
        assert(result.built());
        assert((sources[0] == 2 && sources[1] == 3 && sources[2] == 1
                && sources[3] == 4 && sources[4] == 5 && sources[5] == 9));
    }
    edges[0] = edges[1];
    assert(build().code == signature::exact_destination_support_code_v1::
                               duplicate_edge);
}
