#include <CellShard/compiler/discovery/bicluster/spectral_coclustering_v1.hh>

#include <cassert>

namespace bicluster = cellshard::compiler::discovery::bicluster;
namespace evidence = cellshard::compiler::evidence;

int main() {
    const bicluster::bicluster_edge_v1 edges[] = {
        {{1, 1}, {2, 1}, {3, 1}, 10, 10},
        {{1, 1}, {2, 2}, {3, 1}, 1, 10},
        {{1, 2}, {2, 1}, {3, 1}, 1, 10},
        {{1, 2}, {2, 2}, {3, 1}, 10, 10},
    };
    const bicluster::bicluster_provider_request_v1 request{
        edges, 4, 4, {9, 1}, {1, 9}, {2, 9}, {3, 9}, 1, 4, 128};
    const evidence::evidence_identity_v1 sources[] = {{1, 1}, {1, 2}};
    const evidence::evidence_identity_v1 destinations[] = {{2, 1}, {2, 2}};
    std::int64_t source_scores[2]{};
    std::int64_t destination_scores[2]{};
    evidence::evidence_identity_v1 output_sources[2]{};
    evidence::evidence_identity_v1 output_destinations[2]{};
    auto result = bicluster::generate_spectral_cocluster_v1(
        request, sources, 2, destinations, 2, {3, 1}, {100, 2, 64},
        source_scores, destination_scores, output_sources, 2,
        output_destinations, 2);
    assert(result.generated());
    assert(result.source_count == 1);
    assert(result.destination_count == 1);
    assert((output_sources[0] == evidence::evidence_identity_v1{1, 1}));
    assert((output_destinations[0] == evidence::evidence_identity_v1{2, 1}));

    result = bicluster::generate_spectral_cocluster_v1(
        request, sources, 2, destinations, 2, {3, 1}, {100, 2, 1},
        source_scores, destination_scores, output_sources, 2,
        output_destinations, 2);
    assert(result.code == bicluster::bicluster_spectral_code_v1::work_limit_exceeded);
    return 0;
}
