#include <CellShard/compiler/composition/partial_result_combination_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace composition = cellshard::compiler::composition;

namespace {

composition::partial_result_identity_v1 partial(
    std::uint64_t identity,
    const std::uint64_t *owners,
    std::uint64_t owner_count) {
    return {composition::partial_result_id{identity}, 10,
            cellshard::operator_class_id{11},
            cellshard::scalar_encoding_id{12}, identity,
            {cellshard::structure_id{identity + 20}, cellshard::domain_id{13},
             cellshard::order_id{14}, owners, owner_count}};
}

} // namespace

int main() {
    const std::array<std::uint64_t, 2> left_owners{{1, 5}};
    const std::array<std::uint64_t, 2> right_owners{{2, (std::uint64_t{1} << 54u)}};
    const auto left = partial(1, left_owners.data(), left_owners.size());
    const auto right = partial(2, right_owners.data(), right_owners.size());
    std::array<std::uint64_t, 4> owners{};
    composition::partial_result_combination_v1 output{};
    assert(composition::compose_partial_result_combination_v1(
               composition::composition_production_id{30},
               composition::partial_result_id{3}, 4,
               cellshard::structure_id{31}, left, right,
               {owners.data(), owners.size()}, &output).combined());
    assert(output.combined.contribution_owners.logical_item_count == 4);
    assert(owners[3] == (std::uint64_t{1} << 54u));

    const std::array<std::uint64_t, 1> overlap_owners{{5}};
    const auto overlap = partial(4, overlap_owners.data(), overlap_owners.size());
    assert(composition::compose_partial_result_combination_v1(
               composition::composition_production_id{30},
               composition::partial_result_id{5}, 6,
               cellshard::structure_id{31}, left, overlap,
               {owners.data(), owners.size()}, &output).code
           == composition::partial_result_combination_code_v1::
                  contribution_composition_failed);
}
