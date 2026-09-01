#pragma once
#include <CellShard/compiler/graph/atom_dependency.hh>
#include <CellShard/compiler/schedule/portable_artifact.hh>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
namespace cellshard::compiler::graph {
inline constexpr std::array<std::byte,8> global_ir_magic{{std::byte{'C'},std::byte{'S'},std::byte{'G'},std::byte{'I'},std::byte{'R'},std::byte{'0'},std::byte{'0'},std::byte{'1'}}};
struct global_ir_header { std::array<std::byte,8> magic=global_ir_magic;std::uint32_t schema=1;std::uint32_t endian=0x01020304;graph_family_id family{};std::uint64_t node_count=0;std::uint64_t port_count=0;std::uint64_t edge_count=0;std::uint64_t node_offset=0;std::uint64_t port_offset=0;std::uint64_t edge_offset=0;std::uint64_t total_bytes=0;content_digest content{}; };
struct profiler_event_identity { std::uint64_t high=0;std::uint64_t low=0;[[nodiscard]] constexpr bool valid()const noexcept{return high!=0||low!=0;} };
enum class global_ir_serialize_status : std::uint32_t { success,invalid_input,insufficient_output,overflow };
[[nodiscard]] std::size_t global_ir_serialized_bytes(std::size_t node_count,std::size_t port_count,std::size_t edge_count)noexcept;
[[nodiscard]] global_ir_serialize_status serialize_global_ir(graph_family_id family,const operation_node_descriptor*nodes,std::size_t node_count,const typed_port_descriptor*ports,std::size_t port_count,const atom_dependency_edge*edges,std::size_t edge_count,std::byte*output,std::size_t capacity)noexcept;
[[nodiscard]] profiler_event_identity emit_profiler_identity(graph_family_id family,schedule::portable_schedule_id schedule,operation_node_id node,std::uint64_t command_ordinal)noexcept;
static_assert(std::is_trivially_copyable<global_ir_header>::value);
}
