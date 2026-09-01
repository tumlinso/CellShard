#include <CellShard/compiler/composition/halo_extension_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<std::uint64_t, 3> owned_ids{{1, 5, 9}};
    const std::array<std::uint64_t, 2> halo_ids{{2, (std::uint64_t{1} << 51u)}};
    const composition::exact_coverage_view_v1 owned{
        cellshard::structure_id{1}, cellshard::domain_id{2},
        cellshard::order_id{3}, owned_ids.data(), owned_ids.size()};
    const composition::exact_coverage_view_v1 halo{
        cellshard::structure_id{4}, cellshard::domain_id{2},
        cellshard::order_id{3}, halo_ids.data(), halo_ids.size()};
    std::array<composition::owned_coverage_item_v1, 5> items{};
    composition::halo_extension_view_v1 output{};
    assert(composition::compose_halo_extension_v1(
               cellshard::structure_id{5}, owned, halo,
               items.data(), items.size(), &output).extended());
    assert(output.owned_count == 3 && output.halo_count == 2);
    assert(items[0].role
           == composition::coverage_ownership_role_v1::contribution_owner);
    assert(items[1].role
           == composition::coverage_ownership_role_v1::halo_read_only);
    assert(items[4].logical_identity == (std::uint64_t{1} << 51u));

    const std::array<std::uint64_t, 1> overlapping_ids{{5}};
    auto overlap = halo;
    overlap.logical_item_ids = overlapping_ids.data();
    overlap.logical_item_count = overlapping_ids.size();
    assert(composition::compose_halo_extension_v1(
               cellshard::structure_id{5}, owned, overlap,
               items.data(), items.size(), &output).code
           == composition::halo_extension_code_v1::ownership_overlap);
}
