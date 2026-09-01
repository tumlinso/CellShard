#include <CellShard/compiler/certification/duplicate_detection_v1.hh>

#include <cassert>
#include <cstdint>
#include <vector>

using namespace cellshard::compiler;

int main() {
    std::uint64_t entities_a[]{UINT64_C(1) << 40, (UINT64_C(1) << 40) + 2};
    std::uint64_t entities_b[]{(UINT64_C(1) << 40) + 4,
                               (UINT64_C(1) << 40) + 6};
    certification::atom_entity_coverage_claim_v1 claims[]{
        {entities_a, 2, {10, 1}}, {entities_b, 2, {10, 1}}};
    certification::atom_coverage_bundle_v1 bundles[]{
        {&claims[0], nullptr, 1, 0}, {&claims[1], nullptr, 1, 0}};
    std::vector<certification::certification_member_key_v1> workspace(4);
    assert(certification::detect_duplicate_coverage_v1(
               bundles, 2, workspace.data(), workspace.size())
               .unique());

    entities_b[1] = entities_a[0];
    const auto duplicate = certification::detect_duplicate_coverage_v1(
        bundles, 2, workspace.data(), workspace.size());
    assert(duplicate.code
           == certification::duplicate_detection_code_v1::duplicate_entity);
    assert(duplicate.first_atom_index == 0);
    assert(duplicate.duplicate_atom_index == 1);

    assert(certification::detect_duplicate_coverage_v1(
               bundles, 2, workspace.data(), 3)
               .code
           == certification::duplicate_detection_code_v1::
               insufficient_workspace);

    std::uint64_t edges[]{90, 91};
    certification::atom_relation_edge_coverage_claim_v1 edge_claims[]{
        {edges, 1, {20, 1}, 1}, {edges, 1, {20, 1}, 1}};
    bundles[0] = {nullptr, &edge_claims[0], 0, 1};
    bundles[1] = {nullptr, &edge_claims[1], 0, 1};
    const auto duplicate_edge = certification::detect_duplicate_coverage_v1(
        bundles, 2, workspace.data(), workspace.size());
    assert(duplicate_edge.code
           == certification::duplicate_detection_code_v1::
               duplicate_relation_edge);
}
