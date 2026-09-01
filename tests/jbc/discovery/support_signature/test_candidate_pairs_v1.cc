#include <CellShard/compiler/discovery/support_signature/candidate_pairs_v1.hh>

#include <array>
#include <cassert>

namespace signature =
    cellshard::compiler::discovery::support_signature;

int main() {
    const signature::deterministic_lsh_entry_v1 entries[] = {
        {10, 0, 0}, {10, 0, 1}, {20, 0, 2},
        {30, 1, 0}, {30, 1, 1}, {30, 1, 2}};
    const signature::deterministic_lsh_index_view_v1 index{
        entries, 6, 3, 2, 2, 3, 0, 99, {1, 1}, 2};
    std::array<signature::support_candidate_pair_v1, 4> pairs{};
    std::array<std::uint32_t, 3> fan_out{};
    auto result = signature::build_deduplicated_candidate_pairs_v1(
        index, 2, pairs.data(), pairs.size(), fan_out.data(), fan_out.size());
    assert(result.built() && result.view.pair_count == 3);
    assert(pairs[0].first_destination_index == 0
           && pairs[0].second_destination_index == 1
           && pairs[0].matching_band_count == 2);
    assert(fan_out[0] == 2 && fan_out[1] == 2 && fan_out[2] == 2);
    assert(!signature::authorizes_execution(result.view));
    result = signature::build_deduplicated_candidate_pairs_v1(
        index, 1, pairs.data(), pairs.size(), fan_out.data(), fan_out.size());
    assert(result.code
           == signature::support_candidate_pair_code_v1::fan_out_exceeded);
    result = signature::build_deduplicated_candidate_pairs_v1(
        index, 2, pairs.data(), 1, fan_out.data(), fan_out.size());
    assert(result.code
           == signature::support_candidate_pair_code_v1::insufficient_output);
}
