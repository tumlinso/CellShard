#include <CellShard/compiler/discovery/support_signature/neighborhood_proposal_v1.hh>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

namespace signature =
    cellshard::compiler::discovery::support_signature;
namespace atom = cellshard::compiler::atom;

int main() {
    constexpr std::uint32_t destination_count = 34;
    constexpr std::uint32_t support_size = 32;
    std::vector<signature::destination_source_edge_v1> edges;
    edges.reserve(destination_count * support_size);
    for (std::uint32_t destination = 0;
         destination < destination_count;
         ++destination) {
        for (std::uint32_t source = 0; source < support_size; ++source) {
            std::uint64_t global_source =
                UINT64_C(10000) * destination + source + 1;
            if (destination == 0) global_source = source + 1;
            if (destination == 1) {
                global_source = source < 28 ? source + 1 : 100 + source;
            }
            edges.push_back({1000 + destination, global_source});
        }
    }
    std::reverse(edges.begin(), edges.end());
    std::vector<signature::destination_source_edge_v1> scratch(edges.size());
    std::vector<std::uint64_t> destinations(edges.size());
    std::vector<std::uint64_t> offsets(edges.size() + 1);
    std::vector<std::uint64_t> sources(edges.size());
    const auto support_start = std::chrono::steady_clock::now();
    const auto support = signature::build_exact_destination_support_v1(
        edges.data(), edges.size(), {1, 1}, {2, 1}, {3, 1}, 4,
        {scratch.data(), scratch.size(), destinations.data(),
         destinations.size(), offsets.data(), offsets.size(), sources.data(),
         sources.size()});
    const auto support_end = std::chrono::steady_clock::now();
    assert(support.built());

    constexpr std::uint32_t sketch_size = 64;
    std::vector<std::uint64_t> minima(destination_count * sketch_size);
    const auto sketch = signature::build_minhash_v1(
        support.view, sketch_size, sketch_size, 99,
        minima.data(), minima.size());
    assert(sketch.built());
    constexpr std::uint32_t band_count = 64;
    constexpr std::uint32_t bucket_cap = 8;
    std::vector<signature::deterministic_lsh_entry_v1> entries(
        destination_count * band_count);
    const auto lsh = signature::build_lsh_index_v1(
        sketch.view, band_count, 1, bucket_cap,
        entries.data(), entries.size());
    assert(lsh.built());
    std::uint64_t raw_pair_bound = std::uint64_t(band_count) * bucket_cap
        * (bucket_cap - 1) / 2;
    std::vector<signature::support_candidate_pair_v1> pairs(raw_pair_bound);
    std::vector<std::uint32_t> fan_out(destination_count);
    const auto candidates = signature::build_deduplicated_candidate_pairs_v1(
        lsh.view, 16, pairs.data(), pairs.size(), fan_out.data(),
        fan_out.size());
    assert(candidates.built());
    assert(candidates.required_raw_pairs <= raw_pair_bound);
    assert(lsh.view.entry_count
           == std::uint64_t(destination_count) * band_count);

    std::vector<signature::exact_support_pair_score_v1> scores(
        candidates.view.pair_count);
    const auto exact = signature::rescan_exact_support_pairs_v1(
        support.view, candidates.view, scores.data(), scores.size());
    assert(exact.rescanned());
    std::vector<atom::atom_persistent_identity_v1> proposal_ids(
        scores.size());
    for (std::uint64_t index = 0; index < proposal_ids.size(); ++index)
        proposal_ids[index] = {4, index + 1};
    std::vector<signature::destination_support_neighborhood_proposal_v1>
        proposals(scores.size());
    const auto promoted = signature::build_support_neighborhood_proposals_v1(
        support.view, exact.view, {28, 3, 4, 7, 8},
        proposal_ids.data(), proposal_ids.size(), proposals.data(),
        proposals.size());
    assert(promoted.built());
    assert(promoted.view.proposal_count == 1);
    assert(promoted.view.proposals[0].first_global_destination_id == 1000);
    assert(promoted.view.proposals[0].second_global_destination_id == 1001);
    assert(promoted.view.proposals[0].exact_score.shared_support_count == 28);

    constexpr std::uint64_t high_degree = 4096;
    std::vector<std::uint64_t> high_sources(high_degree);
    for (std::uint64_t index = 0; index < high_degree; ++index)
        high_sources[index] = index + 1;
    const std::uint64_t high_destination[] = {9000};
    const std::uint64_t high_offsets[] = {0, high_degree};
    const signature::exact_destination_support_view_v1 high_support{
        high_destination, high_offsets, high_sources.data(), 1, high_degree,
        {1, 1}, {2, 1}, {3, 1}, 4};
    std::array<std::uint64_t, sketch_size> high_minima{};
    const auto high_start = std::chrono::steady_clock::now();
    const auto high_sketch = signature::build_minhash_v1(
        high_support, sketch_size, sketch_size, 99,
        high_minima.data(), high_minima.size());
    const auto high_end = std::chrono::steady_clock::now();
    assert(high_sketch.built());

    std::array<std::uint64_t, 9 * 8> collision_minima{};
    collision_minima.fill(7);
    const signature::deterministic_minhash_view_v1 collision_sketch{
        collision_minima.data(), 9, 8, 0, 99, {1, 1}, 4};
    std::array<signature::deterministic_lsh_entry_v1, 9> collision_entries{};
    const auto collision = signature::build_lsh_index_v1(
        collision_sketch, 1, 8, bucket_cap,
        collision_entries.data(), collision_entries.size());
    assert(collision.code
           == signature::deterministic_lsh_code_v1::bucket_bound_exceeded);

    const auto support_us = std::chrono::duration_cast<
        std::chrono::microseconds>(support_end - support_start).count();
    const auto high_us = std::chrono::duration_cast<
        std::chrono::microseconds>(high_end - high_start).count();
    std::cout << "case\tdestinations\tedges\tcandidates\tpromoted\tlatency_us\n";
    std::cout << "planted_and_random_null\t" << destination_count << '\t'
              << edges.size() << '\t' << candidates.view.pair_count << '\t'
              << promoted.view.proposal_count << '\t' << support_us << '\n';
    std::cout << "high_degree\t1\t" << high_degree << "\t0\t0\t"
              << high_us << '\n';
    std::cout << "complexity\t" << lsh.view.entry_count << '\t'
              << candidates.required_raw_pairs << '\t' << raw_pair_bound
              << "\t0\t0\n";
}
