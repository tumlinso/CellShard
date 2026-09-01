#pragma once
#include <CellShard/artifact/atom_store/root_manifest_v1.hh>
#include <cstdint>
namespace cellshard::artifact::atom_store {
enum class recovery_class_v1 : std::uint32_t { active_root, recoverable_successor, orphan, incomplete, corrupt };
struct recovery_candidate_v1 { root_generation_manifest_v1 manifest{}; std::uint32_t object_durable=0; std::uint32_t object_valid=0; std::uint32_t selected_by_root=0; std::uint32_t reserved=0; };
[[nodiscard]] recovery_class_v1 classify_recovery_candidate_v1(const root_generation_manifest_v1 &durable_root, const recovery_candidate_v1 &candidate) noexcept;
}
