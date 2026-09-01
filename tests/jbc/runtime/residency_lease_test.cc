#include <CellShard/runtime/v2/residency_lease.hh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    alignas(64) std::array<std::byte, 64> storage{};
    content_digest identity{};
    identity.algorithm = digest_algorithm::legacy_fnv1a64;
    identity.used_bytes = 8;
    const atom_plane_resident_instance instance{
        residency_id{1}, storage_object_id{2}, identity,
        atom_plane_kind::immutable_structure, residency_space::host,
        placement_epoch_id{3}, 0, 1, -1, storage.data(), storage.size(), 64};
    residency_lease_table table;
    assert(table.initialize(1) == status_code::success);
    assert(table.publish(instance) == status_code::success);
    residency_lease first{};
    residency_lease second{};
    assert(table.acquire(residency_id{1}, &first) == status_code::success);
    assert(table.acquire(residency_id{1}, &second) == status_code::success);
    assert(first.pin_mask != second.pin_mask);
    assert(table.evict(residency_id{1}) == status_code::unsupported_capability);
    assert(table.release(first) == status_code::success);
    assert(table.release(first) == status_code::invalid_input);
    assert(table.release(second) == status_code::success);
    assert(table.evict(residency_id{1}) == status_code::success);
    assert(table.release(second) == status_code::invalid_input);
}
