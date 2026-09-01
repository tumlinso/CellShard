#include <CellShard/compiler/discovery/motif/regulatory_baseline_v1.hh>

#include <array>
#include <cassert>

namespace motif = cellshard::compiler::discovery::motif;
namespace atom = cellshard::compiler::atom;

int main() {
    atom::atom_persistent_identity_v1 node_types[] = {
        {1, 1}, {1, 2}, {1, 3}, {1, 4}};
    atom::atom_persistent_identity_v1 relation_types[] = {
        {2, 1}, {2, 2}, {2, 3}, {2, 4}};
    std::array<motif::typed_motif_node_v1, 4> nodes{};
    std::array<motif::typed_motif_edge_v1, 4> edges{};
    for (std::uint32_t value = 1; value <= 5; ++value) {
        const auto kind = static_cast<motif::regulatory_motif_kind_v1>(value);
        const auto shape = motif::regulatory_motif_shape_v1_for(kind);
        auto result = motif::build_regulatory_motif_baseline_v1(
            {kind, node_types, relation_types, shape.node_count,
             shape.edge_count, {3, value}, {4, 1}, 8},
            {nodes.data(), nodes.size(), edges.data(), edges.size()});
        assert(result.built());
        assert(result.motif.node_count == shape.node_count);
        assert(result.motif.edge_count == shape.edge_count);
        assert(motif::validate_typed_motif_vocabulary_v1(result.motif).valid());
    }
    auto result = motif::build_regulatory_motif_baseline_v1(
        {motif::regulatory_motif_kind_v1::bi_fan, node_types, relation_types,
         3, 4, {3, 1}, {4, 1}, 8},
        {nodes.data(), nodes.size(), edges.data(), edges.size()});
    assert(result.code
           == motif::regulatory_motif_baseline_code_v1::incorrect_type_count);
}
