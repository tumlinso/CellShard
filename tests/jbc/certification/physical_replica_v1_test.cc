#include <CellShard/compiler/certification/physical_replica_v1.hh>

#include <cassert>
#include <cstddef>
#include <cstdint>

using namespace cellshard;
using namespace cellshard::compiler;

int main() {
    std::uint64_t payload = 0;
    atom::atom_physical_extent_v1 extent{{5, 1}, 4, 4, 8};
    atom::atom_physical_view_plane_v1 view{};
    view.payload = &payload;
    view.payload_bytes = sizeof(payload);
    view.extents = &extent;
    view.extent_count = 1;
    view.semantic_family = {{1, 1}};
    view.materialization = {{2, 1}};
    view.physical_view_identity = {3, 1};
    view.encoding_identity = {4, 1};
    view.persistent_order_identity = {5, 1};
    view.projection_abi_identity = {6, 1};
    view.materialization_generation = 1;
    view.payload_alignment = alignof(std::uint64_t);

    atom::common_atom_view_v1 common{};
    common.identities.semantic_family = {{1, 1}};
    common.identities.materialization = {{2, 1}};
    common.identities.replica = {{7, 1}};
    common.identities.resident = {8, 1};
    common.identities.content.digest.algorithm =
        digest_algorithm::legacy_fnv1a64;
    common.identities.content.digest.used_bytes = 8;

    certification::physical_replica_binding_v1 replicas[]{
        {&view, {{7, 1}}, common.identities.content},
        {&view, {{7, 2}}, common.identities.content}};
    assert(certification::validate_physical_replicas_v1(common, replicas, 2)
               .valid());

    replicas[1].replica = replicas[0].replica;
    assert(certification::validate_physical_replicas_v1(common, replicas, 2)
               .code
           == certification::physical_replica_validation_code_v1::
               unordered_or_duplicate_replica);

    replicas[1].replica = {{7, 2}};
    replicas[1].content.digest.bytes[0] = std::byte{1};
    assert(certification::validate_physical_replicas_v1(common, replicas, 2)
               .code
           == certification::physical_replica_validation_code_v1::
               content_mismatch);

    replicas[1].content = common.identities.content;
    view.materialization = {{2, 2}};
    assert(certification::validate_physical_replicas_v1(common, replicas, 2)
               .code
           == certification::physical_replica_validation_code_v1::
               materialization_mismatch);
}
