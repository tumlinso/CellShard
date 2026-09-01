#pragma once
#include <CellShard/compiler/graph/physical_realization.hh>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
namespace cellshard::compiler::schedule {
struct portable_schedule_id_tag;
using portable_schedule_id=strong_id<portable_schedule_id_tag>;
inline constexpr std::array<std::byte,8> portable_schedule_magic{{std::byte{'C'},std::byte{'S'},std::byte{'S'},std::byte{'C'},std::byte{'H'},std::byte{'0'},std::byte{'0'},std::byte{'1'}}};
struct portable_schedule_header {
    std::array<std::byte,8> magic=portable_schedule_magic; std::uint32_t schema=1; std::uint32_t endian=0x01020304;
    portable_schedule_id id{}; compiler::graph::physical_realization_id realization{};
    content_digest graph_digest{}; std::uint64_t command_count=0; std::uint64_t dependency_count=0;
    std::uint64_t binding_count=0; std::uint64_t transient_bytes=0;
};
enum class portable_command_kind : std::uint32_t { launch=1, copy=2, barrier=3, transform_order=4, publish=5 };
struct portable_schedule_command {
    compiler::graph::operation_node_id node{}; portable_command_kind kind{}; std::uint32_t flags=0;
    std::uint64_t dependency_offset=0; std::uint64_t dependency_count=0;
    std::uint64_t binding_offset=0; std::uint64_t binding_count=0;
    std::uint64_t transient_offset=0; std::uint64_t transient_bytes=0;
};
[[nodiscard]] constexpr bool valid_portable_schedule_header(const portable_schedule_header&h)noexcept{for(std::size_t i=0;i<h.magic.size();++i)if(h.magic[i]!=portable_schedule_magic[i])return false;return h.schema==1&&h.endian==0x01020304&&h.id.valid()&&h.realization.valid()&&valid_content_digest(h.graph_digest)&&h.graph_digest.algorithm!=digest_algorithm::none&&h.command_count!=0;}
[[nodiscard]] constexpr bool valid_portable_schedule_command(const portable_schedule_command&c,std::uint64_t dependencies,std::uint64_t bindings,std::uint64_t transient)noexcept{const bool kind=c.kind==portable_command_kind::launch||c.kind==portable_command_kind::copy||c.kind==portable_command_kind::barrier||c.kind==portable_command_kind::transform_order||c.kind==portable_command_kind::publish;return c.node.valid()&&kind&&c.dependency_offset<=dependencies&&c.dependency_count<=dependencies-c.dependency_offset&&c.binding_offset<=bindings&&c.binding_count<=bindings-c.binding_offset&&c.transient_offset<=transient&&c.transient_bytes<=transient-c.transient_offset;}
static_assert(std::is_trivially_copyable<portable_schedule_header>::value);
static_assert(std::is_trivially_copyable<portable_schedule_command>::value);
}
