#include <CellShard/compiler/discovery/bicluster/seed_rectangles_v1.hh>

#include <cassert>
#include <cstdint>
#include <limits>

namespace bicluster = cellshard::compiler::discovery::bicluster;

int main() {
    const bicluster::bicluster_edge_v1 edges[] = {
        {{1, 1}, {2, 1}, {3, 1}, 1, 4},
        {{1, 1}, {2, 2}, {3, 1}, 3, 4},
        {{1, 2}, {2, 1}, {3, 1}, 1, 1},
    };
    const bicluster::bicluster_provider_request_v1 request{
        edges, 3, 3, {9, 1}, {1, 99}, {2, 99}, {3, 99}, 1, 3, 16};
    bicluster::bicluster_seed_rectangle_v1 seeds[3]{};
    auto result = bicluster::generate_bicluster_seeds_v1(
        request, {1, 2, 3}, seeds, 3);
    assert(result.generated());
    assert(result.seed_count == 2);
    assert(seeds[0].edge_index == 1);
    assert(seeds[1].edge_index == 2);
    assert((seeds[0].destination_anchor
            == cellshard::compiler::evidence::evidence_identity_v1{2, 2}));

    result = bicluster::generate_bicluster_seeds_v1(
        request, {1, 2, 3}, seeds, 1);
    assert(result.code == bicluster::bicluster_seed_code_v1::insufficient_capacity);
    assert(bicluster::rational_at_least_v1(
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max(), 1, 1));
    return 0;
}
