#include <CellShard/compiler/discovery/motif/typed_vocabulary_v1.hh>

#include <cassert>
#include <cstdint>
#include <random>

namespace motif = cellshard::compiler::discovery::motif;

int main() {
    motif::typed_motif_node_v1 nodes[] = {
        {{1, 11}, 1, 0},
        {{1, 12}, 2, 0},
        {{1, 13}, 3, 0},
    };
    motif::typed_motif_edge_v1 edges[] = {
        {{2, 21}, 0, 1, 1, motif::motif_edge_direction_v1::directed, {}},
        {{2, 22}, 1, 2, 2, motif::motif_edge_direction_v1::undirected, {}},
    };
    motif::typed_motif_vocabulary_view_v1 view{
        nodes, edges, 3, 2, 4, 4, {3, 31}, {4, 41}, 7};
    assert(motif::validate_typed_motif_vocabulary_v1(view).valid());
    assert(!motif::authorizes_execution(view));

    auto bad = view;
    bad.graph_generation = 0;
    assert(motif::validate_typed_motif_vocabulary_v1(bad).code
           == motif::typed_motif_validation_code_v1::missing_graph_generation);
    bad = view;
    bad.maximum_node_count = 2;
    assert(motif::validate_typed_motif_vocabulary_v1(bad).code
           == motif::typed_motif_validation_code_v1::node_bound_exceeded);
    edges[1].destination_node = 3;
    assert(motif::validate_typed_motif_vocabulary_v1(view).code
           == motif::typed_motif_validation_code_v1::endpoint_out_of_range);
    edges[1].destination_node = 2;
    edges[1] = {{2, 21}, 1, 0, 1,
                motif::motif_edge_direction_v1::directed, {}};
    assert(motif::validate_typed_motif_vocabulary_v1(view).valid());
    edges[1] = edges[0];
    assert(motif::validate_typed_motif_vocabulary_v1(view).code
           == motif::typed_motif_validation_code_v1::duplicate_typed_edge);

    std::mt19937_64 random(0x4d4f544946ULL);
    for (std::uint32_t trial = 0; trial < 1000; ++trial) {
        edges[1] = {{2, 22}, 1, 2, 2,
                    motif::motif_edge_direction_v1::undirected, {}};
        edges[1].source_node = static_cast<std::uint32_t>(random() % 5);
        edges[1].destination_node = static_cast<std::uint32_t>(random() % 5);
        const auto result = motif::validate_typed_motif_vocabulary_v1(view);
        const bool endpoints_valid = edges[1].source_node < view.node_count
            && edges[1].destination_node < view.node_count;
        const bool non_self = edges[1].source_node != edges[1].destination_node;
        assert(result.valid() == (endpoints_valid && non_self));
    }
}
