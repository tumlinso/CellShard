#include <CellShard/compiler/discovery/motif/exact_atom_candidate_v1.hh>

#include <array>
#include <cassert>

namespace motif = cellshard::compiler::discovery::motif;

int main() {
    motif::typed_motif_node_v1 nodes[] = {
        {{1, 1}, 1, 0}, {{1, 2}, 2, 0}};
    motif::typed_motif_edge_v1 edges[] = {
        {{2, 1}, 0, 1, 1, motif::motif_edge_direction_v1::directed, {}}};
    motif::typed_motif_vocabulary_view_v1 pattern{
        nodes, edges, 2, 1, 2, 1, {3, 1}, {4, 1}, 9};
    std::uint64_t proposed_ids[] = {10, 20, 30, 40};
    motif::motif_occurrence_output_v1 proposal{proposed_ids, 4, 2};
    std::uint64_t rescanned_ids[] = {10, 20};
    motif::exact_motif_rescan_v1 rescan{rescanned_ids, 1, {5, 1}, 7};
    cellshard::compiler::atom::atom_persistent_identity_v1 ids[] = {{6, 1}};
    std::array<motif::exact_motif_atom_candidate_v1, 1> candidates{};
    std::array<std::uint64_t, 2> output_ids{};
    auto result = motif::build_exact_motif_atom_candidates_v1(
        pattern, proposal, rescan, ids, {7, 1}, {8, 1},
        {candidates.data(), candidates.size(), output_ids.data(),
         output_ids.size()});
    assert(result.built() && result.table.candidate_count == 1);
    assert(result.table.candidates[0].global_node_ids[1] == 20);
    assert(!motif::authorizes_execution(result.table));
    rescanned_ids[1] = 99;
    assert(motif::build_exact_motif_atom_candidates_v1(
               pattern, proposal, rescan, ids, {7, 1}, {8, 1},
               {candidates.data(), candidates.size(), output_ids.data(),
                output_ids.size()}).code
           == motif::exact_motif_candidate_code_v1::occurrence_not_proposed);
    rescanned_ids[1] = 20;
    assert(motif::build_exact_motif_atom_candidates_v1(
               pattern, proposal, rescan, ids, {7, 1}, {7, 1},
               {candidates.data(), candidates.size(), output_ids.data(),
                output_ids.size()}).code
           == motif::exact_motif_candidate_code_v1::
                  provider_self_certification);
}
