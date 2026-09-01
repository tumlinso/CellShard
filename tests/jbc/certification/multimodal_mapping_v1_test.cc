#include <CellShard/compiler/certification/multimodal_mapping_v1.hh>

#include <cassert>
#include <cstdint>

using namespace cellshard::compiler;

int main() {
    const std::uint64_t source_ids[]{100, 200, 300};
    const std::uint64_t destination_ids[]{1000, 2000, 3000};
    const certification::canonical_entity_spine_v1 source{
        source_ids, 3, {1, 1}, 1};
    const certification::canonical_entity_spine_v1 destination{
        destination_ids, 3, {2, 1}, 1};
    certification::multimodal_identity_edge_v1 edges[]{
        {100, 3000}, {200, 1000}, {200, 2000}};
    certification::multimodal_identity_mapping_view_v1 mapping{
        edges, 3, {3, 1}, {1, 1}, {2, 1}, 1};
    assert(certification::validate_multimodal_identity_mapping_v1(
               source, destination, mapping)
               .valid());

    edges[2] = edges[1];
    assert(certification::validate_multimodal_identity_mapping_v1(
               source, destination, mapping)
               .code
           == certification::multimodal_mapping_validation_code_v1::
               unordered_or_duplicate_edge);

    edges[2] = {200, 3001};
    assert(certification::validate_multimodal_identity_mapping_v1(
               source, destination, mapping)
               .code
           == certification::multimodal_mapping_validation_code_v1::
               destination_not_canonical);

    edges[2] = {200, 3000};
    mapping.source_domain_identity = {2, 1};
    assert(certification::validate_multimodal_identity_mapping_v1(
               source, destination, mapping)
               .code
           == certification::multimodal_mapping_validation_code_v1::
               source_domain_mismatch);
}
