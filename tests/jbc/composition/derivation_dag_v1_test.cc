#include <CellShard/compiler/composition/derivation_dag_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace composition = cellshard::compiler::composition;

namespace {

composition::derivation_node_v1 node(std::uint64_t identity) {
    return {composition::composition_production_id{identity},
            composition::composition_transform_kind_v1::disjoint_union, {},
            composition::atom_interface_id{100},
            composition::atom_interface_id{100}};
}

} // namespace

int main() {
    const std::array<composition::derivation_node_v1, 4> nodes{{
        node(1), node(2), node(3), node(4)}};
    const std::array<composition::derivation_edge_v1, 4> edges{{
        {0, 1}, {0, 2}, {1, 3}, {2, 3}}};
    const composition::derivation_dag_view_v1 dag{
        composition::composition_lineage_id{5}, nodes.data(), edges.data(),
        nodes.size(), edges.size()};
    std::array<std::uint32_t, 4> indegrees{};
    std::array<std::uint32_t, 5> offsets{};
    std::array<std::uint32_t, 4> queue{};
    std::array<std::uint32_t, 4> topological{};
    composition::compiled_derivation_dag_v1 output{};
    const auto workspace = composition::derivation_dag_workspace_v1{
        indegrees.data(), offsets.data(), queue.data(), topological.data(),
        indegrees.size(), offsets.size()};
    assert(composition::compile_derivation_dag_v1(dag, workspace, &output)
               .compiled());
    assert(topological[0] == 0 && topological[3] == 3);

    const std::array<composition::derivation_edge_v1, 2> cycle_edges{{
        {0, 1}, {1, 0}}};
    auto cycle = dag;
    cycle.edges = cycle_edges.data();
    cycle.edge_count = cycle_edges.size();
    assert(composition::compile_derivation_dag_v1(cycle, workspace, &output).code
           == composition::derivation_dag_code_v1::cycle_detected);

    std::uint64_t random_state = 0x6a09e667f3bcc909ULL;
    std::array<composition::derivation_node_v1, 16> random_nodes{};
    std::array<composition::derivation_edge_v1, 120> random_edges{};
    std::array<std::uint32_t, 16> random_indegrees{};
    std::array<std::uint32_t, 17> random_offsets{};
    std::array<std::uint32_t, 16> random_queue{};
    std::array<std::uint32_t, 16> random_topological{};
    for (std::uint32_t trial = 0; trial < 200; ++trial) {
        for (std::uint32_t index = 0; index < random_nodes.size(); ++index) {
            random_nodes[index] = node(1000 + trial * 20u + index);
        }
        std::uint32_t edge_count = 0;
        for (std::uint32_t producer = 0;
             producer < random_nodes.size();
             ++producer) {
            for (std::uint32_t consumer = producer + 1u;
                 consumer < random_nodes.size();
                 ++consumer) {
                random_state = random_state * 6364136223846793005ULL + 1ULL;
                if ((random_state >> 61u) == 0) {
                    random_edges[edge_count++] = {producer, consumer};
                }
            }
        }
        const composition::derivation_dag_view_v1 random_dag{
            composition::composition_lineage_id{2000 + trial},
            random_nodes.data(), random_edges.data(), random_nodes.size(),
            edge_count};
        const composition::derivation_dag_workspace_v1 random_workspace{
            random_indegrees.data(), random_offsets.data(), random_queue.data(),
            random_topological.data(), random_indegrees.size(),
            random_offsets.size()};
        assert(composition::compile_derivation_dag_v1(
                   random_dag, random_workspace, &output).compiled());
        std::array<std::uint32_t, 16> positions{};
        for (std::uint32_t position = 0;
             position < random_topological.size();
             ++position) {
            positions[random_topological[position]] = position;
        }
        for (std::uint32_t edge = 0; edge < edge_count; ++edge) {
            assert(positions[random_edges[edge].producer_node]
                   < positions[random_edges[edge].consumer_node]);
        }
    }
}
