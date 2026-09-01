#include <CellShard/compiler/certification/entity_coverage_v1.hh>

#include <cassert>
#include <cstdint>
#include <vector>

using namespace cellshard::compiler;

int main() {
    std::vector<std::uint64_t> canonical(4096);
    for (std::uint64_t index = 0; index < canonical.size(); ++index) {
        canonical[index] = (UINT64_C(1) << 40) + index * 3 + 1;
    }
    std::vector<std::uint64_t> claim;
    for (std::uint64_t index = 0; index < canonical.size(); index += 7) {
        claim.push_back(canonical[index]);
    }

    certification::canonical_entity_spine_v1 spine{
        canonical.data(), canonical.size(), {10, 20}, 3};
    certification::atom_entity_coverage_claim_v1 coverage{
        claim.data(), claim.size(), {10, 20}};
    assert(certification::validate_exact_entity_coverage_v1(spine, coverage)
               .valid());

    auto corrupt = claim;
    corrupt[100] += 1;
    coverage.global_entity_ids = corrupt.data();
    const auto missing =
        certification::validate_exact_entity_coverage_v1(spine, coverage);
    assert(missing.code
           == certification::entity_coverage_validation_code_v1::
               entity_not_in_canonical_domain);
    assert(missing.index == 100);

    corrupt = claim;
    corrupt[200] = corrupt[199];
    coverage.global_entity_ids = corrupt.data();
    assert(certification::validate_exact_entity_coverage_v1(spine, coverage)
               .code
           == certification::entity_coverage_validation_code_v1::
               unordered_or_duplicate_claim_entity);

    coverage.global_entity_ids = claim.data();
    coverage.domain_identity = {10, 21};
    assert(certification::validate_exact_entity_coverage_v1(spine, coverage)
               .code
           == certification::entity_coverage_validation_code_v1::
               domain_mismatch);
}
