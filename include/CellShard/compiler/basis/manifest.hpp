#pragma once

#include "CellShard/compiler/basis/input.hpp"

#include <array>
#include <type_traits>

namespace cellshard::compiler::basis {

struct basis_manifest {
    std::uint32_t format_version = 1;
    std::uint32_t reserved = 0;
    global_id basis_id = 0;
    global_id structure_epoch = 0;
    global_id workload_epoch = 0;
    global_id solver_id = 0;
    std::array<std::uint64_t, 4> input_digest{};
    local_index atom_count = 0;
    local_index atom_offset = 0;
};
static_assert(std::is_trivially_copyable_v<basis_manifest>);

enum class manifest_validity : std::uint8_t { valid, bad_version, bad_identity, bad_range, bad_atoms };
enum class manifest_freshness : std::uint8_t { fresh, stale_workload, stale_structure };

inline manifest_validity validate_manifest(const basis_manifest& manifest,
                                           const global_id* atoms,
                                           local_index atom_table_count) noexcept {
    if (manifest.format_version != 1) return manifest_validity::bad_version;
    if (manifest.basis_id == 0 || manifest.structure_epoch == 0 || manifest.solver_id == 0) return manifest_validity::bad_identity;
    const std::uint64_t end = static_cast<std::uint64_t>(manifest.atom_offset) + manifest.atom_count;
    if (end > atom_table_count || (manifest.atom_count != 0 && atoms == nullptr)) return manifest_validity::bad_range;
    global_id previous = 0;
    for (std::uint64_t i = manifest.atom_offset; i < end; ++i) {
        if (atoms[i] == 0 || (i != manifest.atom_offset && atoms[i] <= previous)) return manifest_validity::bad_atoms;
        previous = atoms[i];
    }
    return manifest_validity::valid;
}

inline manifest_freshness freshness(const basis_manifest& manifest,
                                    global_id current_structure_epoch,
                                    global_id current_workload_epoch) noexcept {
    if (manifest.structure_epoch != current_structure_epoch) return manifest_freshness::stale_structure;
    return manifest.workload_epoch == current_workload_epoch ? manifest_freshness::fresh
                                                              : manifest_freshness::stale_workload;
}

}  // namespace cellshard::compiler::basis
