#include <CellShard/compiler/discovery/bicluster/exact_census_v1.hh>

#include <cassert>

namespace bicluster = cellshard::compiler::discovery::bicluster;
namespace evidence = cellshard::compiler::evidence;

int main() {
    const bicluster::bicluster_edge_v1 edges[] = {
        {{1, 1}, {2, 1}, {3, 1}, 1, 1},
        {{1, 1}, {2, 2}, {3, 1}, 1, 1},
        {{1, 2}, {2, 1}, {3, 1}, 1, 1},
        {{1, 3}, {2, 3}, {3, 1}, 1, 1},
    };
    const bicluster::bicluster_provider_request_v1 request{
        edges, 4, 4, {9, 1}, {1, 9}, {2, 9}, {3, 9}, 5, 4, 64};
    const evidence::evidence_identity_v1 sources[] = {{1, 1}, {1, 2}};
    const evidence::evidence_identity_v1 destinations[] = {{2, 1}, {2, 2}};
    const bicluster::expanded_bicluster_v1 rectangle{
        {9, 1}, {3, 1}, sources, 2, 2, destinations, 2, 2, 0, 1};
    bicluster::bicluster_edge_v1 residual[4]{};
    bicluster::bicluster_exact_census_v1 census{};
    auto result = bicluster::construct_bicluster_exact_census_v1(
        request, rectangle, residual, 4, &census);
    assert(result.constructed());
    assert(census.rectangle_interaction_count == 4);
    assert(census.covered_edge_count == 3);
    assert(census.missing_rectangle_count == 1);
    assert(census.residual_edge_count == 1);
    assert((residual[0].source_identity == evidence::evidence_identity_v1{1, 3}));
    assert(!bicluster::authorizes_execution(census));

    result = bicluster::construct_bicluster_exact_census_v1(
        request, rectangle, residual, 0, &census);
    assert(result.code
           == bicluster::bicluster_exact_census_code_v1::
               insufficient_residual_capacity);
    return 0;
}
