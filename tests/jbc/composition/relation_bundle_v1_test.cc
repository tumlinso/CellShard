#include <CellShard/compiler/composition/relation_bundle_v1.hh>

#include <array>
#include <cassert>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<composition::typed_relation_edge_v1, 1> first_edges{{
        {1, 2, 1}}};
    const std::array<composition::typed_relation_edge_v1, 1> second_edges{{
        {3, 4, (std::uint64_t{1} << 45u)}}};
    const std::array<composition::typed_relation_view_v1, 2> relations{{
        {cellshard::structure_id{1}, cellshard::domain_id{2},
         cellshard::order_id{3}, cellshard::domain_id{4},
         cellshard::order_id{5}, first_edges.data(), first_edges.size()},
        {cellshard::structure_id{6}, cellshard::domain_id{7},
         cellshard::order_id{8}, cellshard::domain_id{9},
         cellshard::order_id{10}, second_edges.data(), second_edges.size()}}};
    composition::relation_bundle_composition_v1 output{};
    assert(composition::compose_relation_bundle_v1(
               composition::relation_bundle_id{11},
               composition::composition_production_id{12}, relations.data(),
               relations.size(), &output).composed());
    assert(output.relation_count == 2 && output.total_logical_edges == 2);

    auto unordered = relations;
    unordered[1].identity = cellshard::structure_id{1};
    assert(composition::compose_relation_bundle_v1(
               composition::relation_bundle_id{11},
               composition::composition_production_id{12}, unordered.data(),
               unordered.size(), &output).code
           == composition::relation_bundle_code_v1::
                  unordered_relation_identity);
}
