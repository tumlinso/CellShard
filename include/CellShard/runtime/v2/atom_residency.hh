#pragma once

#include <CellShard/identity.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::runtime_v2 {

enum class atom_plane_kind : std::uint8_t {
    invalid = 0,
    immutable_structure,
    mutable_values,
    index,
    schedule,
};

enum class residency_space : std::uint8_t {
    invalid = 0,
    host,
    cuda_device,
};

struct atom_plane_resident_instance {
    residency_id residency{};
    storage_object_id object{};
    content_digest atom_identity{};
    atom_plane_kind plane = atom_plane_kind::invalid;
    residency_space space = residency_space::invalid;
    placement_epoch_id placement_epoch{};
    std::uint64_t value_generation = 0;
    std::uint32_t logical_node = 0;
    int device_id = -1;
    const std::byte *data = nullptr;
    std::uint64_t bytes = 0;
    std::uint32_t alignment = 0;
};

[[nodiscard]] constexpr bool valid_atom_plane_resident_instance(
    const atom_plane_resident_instance &instance) noexcept {
    const bool power_of_two = instance.alignment != 0
        && (instance.alignment & (instance.alignment - 1)) == 0;
    if (!instance.residency.valid() || !instance.object.valid()
        || instance.atom_identity.algorithm == digest_algorithm::none
        || !valid_content_digest(instance.atom_identity)
        || instance.plane == atom_plane_kind::invalid
        || instance.space == residency_space::invalid
        || !instance.placement_epoch.valid() || instance.logical_node == 0
        || instance.data == nullptr || instance.bytes == 0 || !power_of_two
        || reinterpret_cast<std::uintptr_t>(instance.data) % instance.alignment
               != 0) {
        return false;
    }
    if (instance.space == residency_space::host && instance.device_id != -1) {
        return false;
    }
    if (instance.space == residency_space::cuda_device
        && instance.device_id < 0) {
        return false;
    }
    return instance.plane == atom_plane_kind::mutable_values
        ? instance.value_generation != 0
        : instance.value_generation == 0;
}

[[nodiscard]] constexpr bool resident_generation_matches(
    const atom_plane_resident_instance &instance,
    std::uint64_t required_generation) noexcept {
    return valid_atom_plane_resident_instance(instance)
        && instance.plane == atom_plane_kind::mutable_values
        && required_generation != 0
        && instance.value_generation == required_generation;
}

static_assert(std::is_trivially_copyable_v<atom_plane_resident_instance>);

} // namespace cellshard::runtime_v2
