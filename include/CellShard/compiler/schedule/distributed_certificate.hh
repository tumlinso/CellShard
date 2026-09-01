#pragma once
#include <CellShard/compiler/schedule/portable_artifact.hh>
#include <cstddef>
#include <cstdint>
#include <type_traits>
namespace cellshard::compiler::schedule {
struct distributed_certificate {
    portable_schedule_id schedule{}; partition_map_id partition_map{};
    route_table_id routes{}; content_digest logical_graph{};
    content_digest exact_coverage{}; std::uint64_t participant_count=0;
    std::uint64_t atom_count=0; std::uint64_t contribution_count=0;
};
struct participant_certificate {
    partition_id partition{}; content_digest contribution_digest{};
    std::uint64_t atom_offset=0; std::uint64_t atom_count=0;
    std::uint64_t contribution_count=0;
};
[[nodiscard]] bool valid_distributed_certificate(const distributed_certificate&certificate,const participant_certificate*participants,std::size_t participant_count)noexcept;
static_assert(std::is_trivially_copyable<distributed_certificate>::value);
static_assert(std::is_trivially_copyable<participant_certificate>::value);
}
