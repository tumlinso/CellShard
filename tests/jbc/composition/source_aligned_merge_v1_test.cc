#include <CellShard/compiler/composition/relation_merge_v1.hh>

#include <array>
#include <cassert>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<composition::typed_relation_edge_v1, 3> left_edges{{
        {1, 2, 1}, {1, 5, 3}, {4, 7, 5}}};
    const std::array<composition::typed_relation_edge_v1, 3> right_edges{{
        {1, 3, 2}, {2, 6, 4}, {8, 9, (std::uint64_t{1} << 40u)}}};
    const composition::typed_relation_view_v1 left{
        cellshard::structure_id{1}, cellshard::domain_id{2},
        cellshard::order_id{3}, cellshard::domain_id{4},
        cellshard::order_id{5}, left_edges.data(), left_edges.size()};
    const composition::typed_relation_view_v1 right{
        cellshard::structure_id{6}, cellshard::domain_id{2},
        cellshard::order_id{3}, cellshard::domain_id{4},
        cellshard::order_id{5}, right_edges.data(), right_edges.size()};
    std::array<composition::typed_relation_edge_v1, 6> edges{};
    composition::typed_relation_view_v1 output{};
    assert(composition::compose_source_aligned_merge_v1(
               cellshard::structure_id{7}, left, right,
               {edges.data(), edges.size()}, &output).composed());
    assert(output.edge_count == 6);
    assert(edges[0].logical_edge_identity == 1);
    assert(edges[1].logical_edge_identity == 2);
    assert(edges[5].logical_edge_identity == (std::uint64_t{1} << 40u));

    auto duplicate_edges = right_edges;
    duplicate_edges[0].logical_edge_identity = 1;
    auto duplicate = right;
    duplicate.edges = duplicate_edges.data();
    assert(composition::compose_source_aligned_merge_v1(
               cellshard::structure_id{7}, left, duplicate,
               {edges.data(), edges.size()}, &output).code
           == composition::relation_merge_code_v1::duplicate_logical_edge);
}
