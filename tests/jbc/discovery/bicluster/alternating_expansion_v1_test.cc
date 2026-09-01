#include <CellShard/compiler/discovery/bicluster/alternating_expansion_v1.hh>

#include <cassert>

namespace bicluster = cellshard::compiler::discovery::bicluster;

int main() {
    const bicluster::bicluster_edge_v1 edges[] = {
        {{1, 1}, {2, 1}, {3, 1}, 1, 1},
        {{1, 1}, {2, 2}, {3, 1}, 1, 1},
        {{1, 2}, {2, 1}, {3, 1}, 1, 1},
        {{1, 2}, {2, 2}, {3, 1}, 3, 4},
        {{1, 3}, {2, 1}, {3, 1}, 1, 4},
    };
    const bicluster::bicluster_provider_request_v1 request{
        edges, 5, 5, {9, 1}, {1, 9}, {2, 9}, {3, 9}, 1, 4, 512};
    const bicluster::bicluster_seed_rectangle_v1 seed{
        {9, 1}, {1, 1}, {2, 1}, {3, 1}, 0, 0};
    cellshard::compiler::evidence::evidence_identity_v1 sources[4]{};
    cellshard::compiler::evidence::evidence_identity_v1 destinations[4]{};
    bicluster::expanded_bicluster_v1 expanded{};
    auto result = bicluster::expand_bicluster_alternating_v1(
        request, seed, {1, 2, 4, 512}, sources, 4, destinations, 4, &expanded);
    assert(result.expanded());
    assert(result.source_count == 2);
    assert(result.destination_count == 2);
    assert((sources[1]
            == cellshard::compiler::evidence::evidence_identity_v1{1, 2}));
    assert((destinations[1]
            == cellshard::compiler::evidence::evidence_identity_v1{2, 2}));
    assert(expanded.completed_rounds <= 4);

    result = bicluster::expand_bicluster_alternating_v1(
        request, seed, {1, 2, 4, 1}, sources, 4, destinations, 4, &expanded);
    assert(result.code == bicluster::bicluster_expansion_code_v1::work_limit_exceeded);
    return 0;
}
