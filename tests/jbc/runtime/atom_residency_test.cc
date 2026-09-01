#include <CellShard/runtime/v2/atom_residency.hh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    alignas(64) std::array<std::byte, 64> storage{};
    content_digest identity{};
    identity.algorithm = digest_algorithm::legacy_fnv1a64;
    identity.used_bytes = 8;
    atom_plane_resident_instance instance{
        residency_id{1}, storage_object_id{2}, identity,
        atom_plane_kind::mutable_values, residency_space::host,
        placement_epoch_id{3}, 7, 1, -1, storage.data(), storage.size(), 64};
    assert(valid_atom_plane_resident_instance(instance));
    assert(resident_generation_matches(instance, 7));
    assert(!resident_generation_matches(instance, 6));
    instance.plane = atom_plane_kind::immutable_structure;
    assert(!valid_atom_plane_resident_instance(instance));
    instance.value_generation = 0;
    assert(valid_atom_plane_resident_instance(instance));
    instance.data += 1;
    assert(!valid_atom_plane_resident_instance(instance));
}
