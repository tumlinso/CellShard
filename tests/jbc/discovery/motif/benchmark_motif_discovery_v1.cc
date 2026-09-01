#include <CellShard/compiler/discovery/motif/exact_atom_candidate_v1.hh>
#include <CellShard/compiler/discovery/motif/regulatory_baseline_v1.hh>

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

namespace motif = cellshard::compiler::discovery::motif;
namespace atom = cellshard::compiler::atom;

namespace {

constexpr std::uint32_t graph_count = 32;
constexpr std::uint32_t planted_count = 16;

struct benchmark_case {
    std::vector<motif::typed_graph_node_v1> nodes;
    std::vector<motif::typed_graph_edge_v1> edges;
};

benchmark_case make_case(bool planted) {
    benchmark_case result;
    result.nodes.reserve(graph_count * 3);
    result.edges.reserve(graph_count * 3);
    for (std::uint32_t index = 0; index < graph_count; ++index) {
        result.nodes.push_back({1000 + index, {1, 1}});
        result.nodes.push_back({2000 + index, {1, 2}});
        result.nodes.push_back({3000 + index, {1, 3}});
    }
    for (std::uint32_t index = 0; index < graph_count; ++index) {
        const auto a = 3 * index;
        const auto b = a + 1;
        const auto c = a + 2;
        result.edges.push_back(
            {a, b, {2, 1}, motif::motif_edge_direction_v1::directed, {}});
        result.edges.push_back(
            {b, 3 * ((index + 1) % graph_count) + 2, {2, 2},
             motif::motif_edge_direction_v1::directed, {}});
        result.edges.push_back(
            {a, c, {2, 3}, motif::motif_edge_direction_v1::directed, {}});
        if (planted && index < planted_count) {
            result.edges.back() =
                {a, c, {2, 3}, motif::motif_edge_direction_v1::directed, {}};
            result.edges[result.edges.size() - 2] =
                {b, c, {2, 2}, motif::motif_edge_direction_v1::directed, {}};
        }
    }
    return result;
}

} // namespace

int main() {
    atom::atom_persistent_identity_v1 node_types[] = {
        {1, 1}, {1, 2}, {1, 3}};
    atom::atom_persistent_identity_v1 relation_types[] = {
        {2, 1}, {2, 2}, {2, 3}};
    std::array<motif::typed_motif_node_v1, 3> motif_nodes{};
    std::array<motif::typed_motif_edge_v1, 3> motif_edges{};
    const auto baseline = motif::build_regulatory_motif_baseline_v1(
        {motif::regulatory_motif_kind_v1::feed_forward_loop,
         node_types, relation_types, 3, 3, {3, 1}, {4, 1}, 1},
        {motif_nodes.data(), motif_nodes.size(), motif_edges.data(),
         motif_edges.size()});
    assert(baseline.built());

    auto planted = make_case(true);
    auto null_graph = make_case(false);
    std::array<std::uint32_t, 3> mapping{};
    std::array<std::uint8_t, graph_count * 3> used{};
    std::array<std::uint64_t, 192> planted_ids{};
    std::array<std::uint64_t, 192> null_ids{};
    motif::motif_occurrence_workspace_v1 workspace{
        mapping.data(), mapping.size(), used.data(), used.size()};
    motif::motif_occurrence_output_v1 planted_output{
        planted_ids.data(), planted_ids.size(), 0};
    motif::motif_occurrence_output_v1 null_output{
        null_ids.data(), null_ids.size(), 0};

    const auto planted_start = std::chrono::steady_clock::now();
    const auto planted_result = motif::enumerate_motif_occurrences_v1(
        baseline.motif,
        {planted.nodes.data(), planted.edges.data(),
         static_cast<std::uint32_t>(planted.nodes.size()),
         static_cast<std::uint32_t>(planted.edges.size()), {4, 1}, 1},
        {10000, 64}, workspace, &planted_output);
    const auto planted_end = std::chrono::steady_clock::now();
    const auto null_result = motif::enumerate_motif_occurrences_v1(
        baseline.motif,
        {null_graph.nodes.data(), null_graph.edges.data(),
         static_cast<std::uint32_t>(null_graph.nodes.size()),
         static_cast<std::uint32_t>(null_graph.edges.size()), {4, 1}, 1},
        {10000, 64}, workspace, &null_output);
    const auto null_end = std::chrono::steady_clock::now();
    assert(planted_result.complete());
    assert(null_result.complete());
    assert(planted_output.occurrence_count == planted_count);
    assert(null_output.occurrence_count == 0);

    constexpr std::uint64_t promoted = planted_count / 2;
    std::array<atom::atom_persistent_identity_v1, promoted> identities{};
    for (std::uint64_t index = 0; index < promoted; ++index) {
        identities[index] = {6, index + 1};
    }
    std::array<motif::exact_motif_atom_candidate_v1, promoted> candidates{};
    std::array<std::uint64_t, promoted * 3> promoted_ids{};
    const auto promotion_start = std::chrono::steady_clock::now();
    const auto promotion = motif::build_exact_motif_atom_candidates_v1(
        baseline.motif, planted_output,
        {planted_ids.data(), promoted, {5, 1}, 1}, identities.data(),
        {7, 1}, {8, 1},
        {candidates.data(), candidates.size(), promoted_ids.data(),
         promoted_ids.size()});
    const auto promotion_end = std::chrono::steady_clock::now();
    assert(promotion.built());
    assert(promotion.table.candidate_count == promoted);
    assert(!motif::authorizes_execution(promotion.table));

    const auto planted_us = std::chrono::duration_cast<
        std::chrono::microseconds>(planted_end - planted_start).count();
    const auto null_us = std::chrono::duration_cast<
        std::chrono::microseconds>(null_end - planted_end).count();
    const auto promotion_us = std::chrono::duration_cast<
        std::chrono::microseconds>(promotion_end - promotion_start).count();
    std::cout << "case\tgraphs\tedges\toccurrences\tassignments\tlatency_us\n";
    std::cout << "planted\t" << graph_count << '\t' << planted.edges.size()
              << '\t' << planted_output.occurrence_count << '\t'
              << planted_result.assignments_examined << '\t' << planted_us
              << '\n';
    std::cout << "rewired_null\t" << graph_count << '\t'
              << null_graph.edges.size() << '\t' << null_output.occurrence_count
              << '\t' << null_result.assignments_examined << '\t' << null_us
              << '\n';
    std::cout << "exact_promotion\t" << promoted << "\t0\t"
              << promotion.table.candidate_count << "\t0\t" << promotion_us
              << '\n';
}
