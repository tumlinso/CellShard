#include <CellShard/compiler/discovery/motif/occurrence_enumeration_v1.hh>

#include <array>
#include <cassert>

namespace motif = cellshard::compiler::discovery::motif;

int main() {
    motif::typed_motif_node_v1 motif_nodes[] = {
        {{1, 1}, 1, 0}, {{1, 2}, 2, 0}};
    motif::typed_motif_edge_v1 motif_edges[] = {
        {{2, 1}, 0, 1, 1, motif::motif_edge_direction_v1::directed, {}}};
    motif::typed_motif_vocabulary_view_v1 pattern{
        motif_nodes, motif_edges, 2, 1, 2, 1, {3, 1}, {4, 1}, 9};
    motif::typed_graph_node_v1 nodes[] = {
        {101, {1, 1}}, {102, {1, 2}}, {103, {1, 1}}, {104, {1, 2}}};
    motif::typed_graph_edge_v1 edges[] = {
        {0, 1, {2, 1}, motif::motif_edge_direction_v1::directed, {}},
        {2, 3, {2, 1}, motif::motif_edge_direction_v1::directed, {}},
        {0, 3, {2, 9}, motif::motif_edge_direction_v1::directed, {}}};
    motif::typed_graph_view_v1 graph{nodes, edges, 4, 3, {4, 1}, 9};
    std::array<std::uint32_t, 2> mapping{};
    std::array<std::uint8_t, 4> used{};
    std::array<std::uint64_t, 8> output_words{};
    motif::motif_occurrence_output_v1 output{
        output_words.data(), output_words.size(), 0};
    const motif::motif_occurrence_workspace_v1 workspace{
        mapping.data(), mapping.size(), used.data(), used.size()};
    auto result = motif::enumerate_motif_occurrences_v1(
        pattern, graph, {16, 4}, workspace, &output);
    assert(result.complete() && result.occurrences == 2);
    assert(output_words[0] == 101 && output_words[1] == 102);
    assert(output_words[2] == 103 && output_words[3] == 104);
    assert(!motif::authorizes_execution(output));
    result = motif::enumerate_motif_occurrences_v1(
        pattern, graph, {16, 1}, workspace, &output);
    assert(result.code
           == motif::motif_occurrence_code_v1::truncated_occurrence_limit);
    graph.graph_generation = 10;
    assert(motif::enumerate_motif_occurrences_v1(
               pattern, graph, {16, 4}, workspace, &output).code
           == motif::motif_occurrence_code_v1::graph_generation_mismatch);
}
