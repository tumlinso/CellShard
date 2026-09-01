#include <CellShard/compiler/composition/destination_merge_v1.hh>

#include <array>
#include <cassert>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<composition::typed_relation_edge_v1, 3> left_edges{{
        {2, 1, 1}, {5, 1, 3}, {7, 4, 5}}};
    const std::array<composition::typed_relation_edge_v1, 3> right_edges{{
        {3, 1, 2}, {6, 2, 4}, {9, 8, (std::uint64_t{1} << 41u)}}};
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
    assert(composition::compose_destination_aligned_merge_v1(
               cellshard::structure_id{7}, left, right,
               {edges.data(), edges.size()}, &output).composed());
    assert(output.edge_count == 6);
    assert(edges[0].source_identity == 2 && edges[0].destination_identity == 1);
    assert(edges[1].source_identity == 3 && edges[1].destination_identity == 1);

    auto wrong_destination = right;
    wrong_destination.destination_order = cellshard::order_id{9};
    assert(composition::compose_destination_aligned_merge_v1(
               cellshard::structure_id{7}, left, wrong_destination,
               {edges.data(), edges.size()}, &output).code
           == composition::relation_merge_code_v1::destination_axis_mismatch);
}
