#include <CellShard/compiler/composition/transpose_overlay_v1.hh>

#include <array>
#include <cassert>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<composition::typed_relation_edge_v1, 3> forward_edges{{
        {1, 4, 1}, {1, 7, 2}, {3, 9, (std::uint64_t{1} << 42u)}}};
    const std::array<composition::typed_relation_edge_v1, 3> transpose_edges{{
        {4, 1, 1}, {7, 1, 2}, {9, 3, (std::uint64_t{1} << 42u)}}};
    const composition::typed_relation_view_v1 forward{
        cellshard::structure_id{1}, cellshard::domain_id{2},
        cellshard::order_id{3}, cellshard::domain_id{4},
        cellshard::order_id{5}, forward_edges.data(), forward_edges.size()};
    const composition::typed_relation_view_v1 transpose{
        cellshard::structure_id{6}, cellshard::domain_id{4},
        cellshard::order_id{5}, cellshard::domain_id{2},
        cellshard::order_id{3}, transpose_edges.data(), transpose_edges.size()};
    composition::transpose_overlay_composition_v1 output{};
    assert(composition::compose_transpose_overlay_v1(
               composition::composition_production_id{7}, forward, transpose,
               &output).composed());
    assert(output.logical_edge_count == 3);

    auto wrong_edges = transpose_edges;
    wrong_edges[1].source_identity = 8;
    auto wrong = transpose;
    wrong.edges = wrong_edges.data();
    assert(composition::compose_transpose_overlay_v1(
               composition::composition_production_id{7}, forward, wrong,
               &output).code
           == composition::transpose_overlay_code_v1::
                  endpoint_not_transposed);
}
