#pragma once
#include <CellShard/artifact/atom_store/identity_v1.hh>
#include <CellShard/identity.hh>
#include <cstddef>
#include <cstdint>
namespace cellshard::artifact::atom_store {
struct generation_node_v1 { content_digest_v1 root{}; content_digest_v1 parent{}; std::uint64_t generation=0; };
struct snapshot_pin_v1 { cellshard::snapshot_id snapshot{}; content_digest_v1 root{}; std::uint64_t pin_generation=0; std::uint64_t valid_through_generation=0; };
enum class reachability_status_v1 : std::uint32_t { success, invalid_input, missing_root, cycle };
[[nodiscard]] reachability_status_v1 mark_reachable_generations_v1(const generation_node_v1 *nodes,std::size_t node_count,const content_digest_v1 &active,const snapshot_pin_v1 *pins,std::size_t pin_count,std::uint64_t current_generation,bool *reachable) noexcept;
}
