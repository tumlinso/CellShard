#pragma once
#include <CellShard/compiler/graph/operation_node.hh>
#include <cstdint>
#include <type_traits>
namespace cellshard::compiler::graph {
enum class access_mode : std::uint32_t { read=1, write=2, read_write=3, consume=4 };
enum class residency_kind : std::uint32_t { host=1, device=2, portable=3 };
enum operation_effect : std::uint64_t {
    effect_none=0, effect_create_structure=UINT64_C(1)<<0,
    effect_advance_values=UINT64_C(1)<<1, effect_transform_order=UINT64_C(1)<<2,
    effect_publish_artifact=UINT64_C(1)<<3,
};
struct port_access_descriptor {
    operation_port_id port{}; access_mode mode{}; residency_kind residency{};
    std::uint32_t reserved=0; std::uint64_t structure_epoch=0;
    std::uint64_t value_generation=0; std::uint64_t effects=effect_none;
};
[[nodiscard]] constexpr bool valid_port_access_descriptor(const port_access_descriptor&a)noexcept{
    constexpr auto known=effect_create_structure|effect_advance_values|effect_transform_order|effect_publish_artifact;
    const bool mode_valid=a.mode==access_mode::read||a.mode==access_mode::write||a.mode==access_mode::read_write||a.mode==access_mode::consume;
    const bool residency_valid=a.residency==residency_kind::host||a.residency==residency_kind::device||a.residency==residency_kind::portable;
    if(!a.port.valid()||!mode_valid||!residency_valid||a.structure_epoch==0||(a.effects&~known)!=0)return false;
    if(a.mode==access_mode::read&&a.effects!=effect_none)return false;
    if((a.effects&effect_advance_values)!=0&&a.value_generation==0)return false;
    if((a.effects&effect_create_structure)!=0&&a.mode==access_mode::consume)return false;
    return true;
}
static_assert(std::is_trivially_copyable<port_access_descriptor>::value);
}
