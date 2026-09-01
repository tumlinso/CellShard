#pragma once
#include <CellShard/compiler/graph/operation_provider.hh>
#include <CellShard/identity/strong_id.hh>
#include <cstdint>
#include <type_traits>
namespace cellshard::compiler::graph {
struct operation_node_id_tag;
struct operation_port_id_tag;
using operation_node_id = strong_id<operation_node_id_tag>;
using operation_port_id = strong_id<operation_port_id_tag>;
enum class port_direction : std::uint32_t { input=1, output=2 };
enum class port_payload_kind : std::uint32_t { structure=1, value_plane=2, event_stream=3, control=4 };
struct typed_port_descriptor {
    operation_port_id id{}; operation_node_id node{}; domain_id domain{};
    order_id order{}; scalar_encoding_id encoding{}; std::uint32_t ordinal=0;
    port_direction direction{}; port_payload_kind payload{}; std::uint32_t reserved=0;
};
struct operation_node_descriptor {
    operation_node_id id{}; producer_abi_id provider{}; operator_class_id operation{};
    std::uint64_t port_offset=0; std::uint64_t port_count=0; std::uint64_t structure_epoch=0;
};
[[nodiscard]] constexpr bool valid_typed_port_descriptor(const typed_port_descriptor&p)noexcept{
    const bool typed=p.payload==port_payload_kind::control||(p.domain.valid()&&p.order.valid()&&p.encoding.valid());
    return p.id.valid()&&p.node.valid()&&typed&&(p.direction==port_direction::input||p.direction==port_direction::output)
        && (p.payload==port_payload_kind::structure||p.payload==port_payload_kind::value_plane||p.payload==port_payload_kind::event_stream||p.payload==port_payload_kind::control);
}
[[nodiscard]] constexpr bool valid_operation_node_descriptor(const operation_node_descriptor&n)noexcept{return n.id.valid()&&n.provider.valid()&&n.operation.valid()&&n.port_count!=0&&n.structure_epoch!=0&&n.port_offset<=UINT64_MAX-n.port_count;}
static_assert(std::is_trivially_copyable<typed_port_descriptor>::value);
static_assert(std::is_trivially_copyable<operation_node_descriptor>::value);
}
