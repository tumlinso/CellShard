#include <CellShard/compiler/certification/trajectory_lineage_v1.hh>

#include <cassert>
#include <cstdint>
#include <vector>

using namespace cellshard::compiler;

int main() {
    const std::uint64_t nodes_data[]{10, 20, 30, 40, 50};
    const certification::canonical_entity_spine_v1 nodes{
        nodes_data, 5, {1, 1}, 1};
    certification::trajectory_lineage_edge_v1 edges[]{
        {10, 20}, {10, 30}, {20, 40}, {30, 40}, {40, 50}};
    certification::trajectory_lineage_mapping_view_v1 mapping{
        edges, 5, {2, 1}, {1, 1}, 1};
    std::vector<std::uint64_t> offsets(6);
    std::vector<std::uint64_t> indegrees(5);
    std::vector<std::uint64_t> queue(5);
    certification::trajectory_lineage_workspace_v1 workspace{
        offsets.data(),
        offsets.size(),
        indegrees.data(),
        indegrees.size(),
        queue.data(),
        queue.size()};
    assert(certification::validate_trajectory_lineage_mapping_v1(
               nodes, mapping, workspace)
               .valid());

    certification::trajectory_lineage_edge_v1 cyclic_edges[]{
        {10, 20}, {20, 30}, {30, 10}};
    mapping.edges = cyclic_edges;
    mapping.edge_count = 3;
    assert(certification::validate_trajectory_lineage_mapping_v1(
               nodes, mapping, workspace)
               .code
           == certification::trajectory_lineage_validation_code_v1::cycle);

    cyclic_edges[2] = {30, 60};
    assert(certification::validate_trajectory_lineage_mapping_v1(
               nodes, mapping, workspace)
               .code
           == certification::trajectory_lineage_validation_code_v1::
               child_not_canonical);
}
