#include <CellShard/compiler/discovery/co_support/affinity_evidence_image_v1.hh>

#include <cassert>
#include <cstddef>
#include <vector>

namespace co_support = cellshard::compiler::discovery::co_support;

int main() {
    const co_support::normalized_association_record_v1 associations[] = {
        {0, 1, 1, 2, 3, 2},
    };
    const co_support::source_affinity_edge_v1 edges[] = {
        {0, 1, 0, 0, 1, 2}, {1, 0, 0, 0, 1, 2},
    };
    const co_support::affinity_stability_record_v1 stability[] = {
        {0, 1, 2, 1, 1, 2, 1, 1},
    };
    const co_support::exact_group_rescan_summary_v1 rescans[] = {
        {81, 12, 4, 8, 2, 2},
    };
    const co_support::affinity_evidence_view_v1 evidence{
        71, 72, 3, associations, 1, edges, 2, stability, 1, rescans, 1};
    auto result = co_support::pack_affinity_evidence_image_v1(
        evidence, nullptr, 0);
    assert(result.packed());
    std::vector<std::byte> image(result.required_bytes);
    result = co_support::pack_affinity_evidence_image_v1(
        evidence, image.data(), image.size());
    assert(result.packed());
    assert(co_support::validate_affinity_evidence_image_v1(
        image.data(), image.size()).packed());
    image.back() ^= std::byte{1};
    assert(co_support::validate_affinity_evidence_image_v1(
        image.data(), image.size()).code
        == co_support::affinity_evidence_image_code_v1::checksum_mismatch);
    return 0;
}
