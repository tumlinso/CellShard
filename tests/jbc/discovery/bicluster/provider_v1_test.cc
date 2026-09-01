#include <CellShard/compiler/discovery/bicluster/provider_v1.hh>

#include <cassert>

namespace bicluster = cellshard::compiler::discovery::bicluster;

int main() {
    const bicluster::bicluster_edge_v1 edges[] = {
        {{1, 1}, {2, 1}, {3, 1}, 3, 4},
        {{1, 1}, {2, 2}, {3, 1}, 1, 2},
        {{1, 2}, {2, 1}, {3, 1}, 1, 1},
    };
    bicluster::bicluster_provider_request_v1 request{
        edges, 3, 3, {9, 1}, {1, 99}, {2, 99}, {3, 99}, 7, 4, 64};
    assert(bicluster::validate_bicluster_provider_request_v1(request).valid());
    assert(!bicluster::authorizes_execution(request));

    request.edge_capacity = 2;
    assert(bicluster::validate_bicluster_provider_request_v1(request).code
           == bicluster::bicluster_provider_validation_code_v1::
               edge_capacity_overflow);
    request.edge_capacity = 3;

    const bicluster::bicluster_edge_v1 duplicate[] = {edges[0], edges[0]};
    request.edges = duplicate;
    request.edge_count = 2;
    request.edge_capacity = 2;
    assert(bicluster::validate_bicluster_provider_request_v1(request).code
           == bicluster::bicluster_provider_validation_code_v1::
               unordered_or_duplicate_edge);
    return 0;
}
