#include <CellShard/compiler/certification/read_only_halo_v1.hh>

#include <cassert>
#include <cstdint>

using namespace cellshard::compiler;

int main() {
    const std::uint64_t canonical_ids[]{10, 20, 30, 40, 50, 60};
    const std::uint64_t owned_ids[]{10, 30, 50};
    const std::uint64_t halo_ids[]{20, 40, 60};
    certification::canonical_entity_spine_v1 canonical{
        canonical_ids, 6, {1, 1}, 2};
    certification::atom_entity_coverage_claim_v1 owned{
        owned_ids, 3, {1, 1}};
    certification::atom_entity_coverage_claim_v1 halo{
        halo_ids, 3, {1, 1}};
    assert(certification::validate_read_only_halo_v1(canonical, owned, halo)
               .valid());

    const std::uint64_t overlapping_halo_ids[]{20, 30, 60};
    halo.global_entity_ids = overlapping_halo_ids;
    const auto overlap =
        certification::validate_read_only_halo_v1(canonical, owned, halo);
    assert(overlap.code
           == certification::read_only_halo_validation_code_v1::
               owned_halo_overlap);
    assert(overlap.owned_index == 1);
    assert(overlap.halo_index == 1);

    const std::uint64_t corrupt_halo_ids[]{20, 41, 60};
    halo.global_entity_ids = corrupt_halo_ids;
    assert(certification::validate_read_only_halo_v1(canonical, owned, halo)
               .code
           == certification::read_only_halo_validation_code_v1::
               invalid_halo_coverage);
}
