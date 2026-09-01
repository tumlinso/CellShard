#pragma once
#include <CellShard/artifact/atom_store/identity_v1.hh>
#include <cstddef>
#include <cstdint>
namespace cellshard::artifact::atom_store {
struct consolidation_source_v1 { content_digest_v1 content{}; std::uint64_t source_offset=0; std::uint64_t bytes=0; std::uint64_t alignment=0; std::uint32_t live=0; std::uint32_t reserved=0; };
struct consolidation_copy_v1 { content_digest_v1 content{}; std::uint64_t source_offset=0; std::uint64_t target_offset=0; std::uint64_t bytes=0; };
enum class consolidation_status_v1 : std::uint32_t { success, invalid_input, insufficient_output, overflow };
[[nodiscard]] consolidation_status_v1 plan_consolidation_v1(const consolidation_source_v1 *sources,std::size_t source_count,consolidation_copy_v1 *copies,std::size_t copy_capacity,std::size_t *copy_count,std::uint64_t *output_bytes) noexcept;
}
