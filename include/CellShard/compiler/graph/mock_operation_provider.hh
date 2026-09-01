#pragma once
#include <CellShard/compiler/graph/operation_provider.hh>
#include <CellShard/compiler/graph/physical_realization.hh>
#include <CellShard/compiler/schedule/portable_artifact.hh>
namespace cellshard::compiler::graph {
enum class mock_provider_status : std::uint32_t { success,invalid_node,unsupported_target,invalid_projection };
[[nodiscard]] operation_provider_descriptor mock_non_cellerator_provider_descriptor()noexcept;
[[nodiscard]] mock_provider_status lower_mock_non_cellerator_operation(const operation_node_descriptor&node,const target_capabilities&target,image_id projection,physical_node_binding*binding,schedule::portable_schedule_command*command)noexcept;
}
