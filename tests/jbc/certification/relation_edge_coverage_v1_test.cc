#include <CellShard/compiler/certification/relation_edge_coverage_v1.hh>

#include <cassert>
#include <cstdint>
#include <vector>

using namespace cellshard::compiler;

int main() {
    std::vector<std::uint64_t> canonical(5000);
    for (std::uint64_t index = 0; index < canonical.size(); ++index) {
        canonical[index] = (UINT64_C(1) << 48) + index * 5 + 1;
    }
    std::vector<std::uint64_t> claim;
    for (std::uint64_t index = 0; index < canonical.size(); index += 11) {
        claim.push_back(canonical[index]);
    }
    atom::relation_edge_spine_view_v1 spine{
        canonical.data(), canonical.size(), {70, 80}, 9};
    certification::atom_relation_edge_coverage_claim_v1 coverage{
        claim.data(), claim.size(), {70, 80}, 9};
    assert(certification::validate_exact_relation_edge_coverage_v1(
               spine, coverage)
               .valid());

    auto corrupt = claim;
    corrupt[99] += 1;
    coverage.global_edge_ids = corrupt.data();
    assert(certification::validate_exact_relation_edge_coverage_v1(
               spine, coverage)
               .code
           == certification::relation_edge_coverage_validation_code_v1::
               edge_not_in_canonical_relation);

    coverage.global_edge_ids = claim.data();
    coverage.structure_epoch = 10;
    assert(certification::validate_exact_relation_edge_coverage_v1(
               spine, coverage)
               .code
           == certification::relation_edge_coverage_validation_code_v1::
               epoch_mismatch);

    coverage.structure_epoch = 9;
    coverage.relation_identity = {70, 81};
    assert(certification::validate_exact_relation_edge_coverage_v1(
               spine, coverage)
               .code
           == certification::relation_edge_coverage_validation_code_v1::
               relation_mismatch);
}
