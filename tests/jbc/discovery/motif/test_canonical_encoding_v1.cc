#include <CellShard/compiler/discovery/motif/canonical_encoding_v1.hh>

#include <algorithm>
#include <array>
#include <cassert>

namespace motif = cellshard::compiler::discovery::motif;

int main() {
    motif::typed_motif_node_v1 a_nodes[] = {
        {{1, 2}, 2, 0}, {{1, 1}, 1, 0}, {{1, 2}, 2, 0}};
    motif::typed_motif_edge_v1 a_edges[] = {
        {{2, 1}, 1, 0, 1, motif::motif_edge_direction_v1::directed, {}},
        {{2, 2}, 0, 2, 2, motif::motif_edge_direction_v1::undirected, {}}};
    motif::typed_motif_node_v1 b_nodes[] = {
        {a_nodes[2]}, {a_nodes[1]}, {a_nodes[0]}};
    motif::typed_motif_edge_v1 b_edges[] = {
        {{2, 2}, 2, 0, 2, motif::motif_edge_direction_v1::undirected, {}},
        {{2, 1}, 1, 2, 1, motif::motif_edge_direction_v1::directed, {}}};
    motif::typed_motif_vocabulary_view_v1 a{
        a_nodes, a_edges, 3, 2, 3, 2, {3, 1}, {4, 1}, 1};
    motif::typed_motif_vocabulary_view_v1 b{
        b_nodes, b_edges, 3, 2, 3, 2, {3, 2}, {4, 1}, 1};
    std::array<std::uint32_t, 3> permutation{};
    std::array<std::uint64_t, 24> candidate{};
    std::array<std::uint64_t, 24> a_words{};
    std::array<std::uint64_t, 24> b_words{};
    motif::canonical_motif_encoding_workspace_v1 workspace{
        permutation.data(), permutation.size(), candidate.data(),
        candidate.size()};
    motif::canonical_motif_encoding_output_v1 a_output{
        a_words.data(), a_words.size(), 0};
    auto result = motif::encode_canonical_motif_v1(a, workspace, &a_output);
    assert(result.encoded() && result.permutations_examined == 6);
    motif::canonical_motif_encoding_output_v1 b_output{
        b_words.data(), b_words.size(), 0};
    result = motif::encode_canonical_motif_v1(b, workspace, &b_output);
    assert(result.encoded());
    assert(a_output.word_count == b_output.word_count);
    assert(std::equal(a_words.begin(), a_words.end(), b_words.begin()));
    b_nodes[0].role = 9;
    result = motif::encode_canonical_motif_v1(b, workspace, &b_output);
    assert(result.encoded());
    assert(!std::equal(a_words.begin(), a_words.end(), b_words.begin()));
    workspace.candidate_word_capacity = 1;
    assert(motif::encode_canonical_motif_v1(a, workspace, &a_output).code
           == motif::canonical_motif_encoding_code_v1::
                  insufficient_candidate_words);
}
