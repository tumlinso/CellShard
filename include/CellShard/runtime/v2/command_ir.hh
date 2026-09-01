#pragma once

#include <CellShard/identity.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::runtime_v2 {

enum class runtime_command_kind : std::uint8_t {
    invalid = 0,
    read_atoms,
    stage_host,
    transport,
    publish_residency,
    pin_residency,
    evict_residency,
    barrier,
};

struct runtime_command {
    std::uint32_t id = 0;
    runtime_command_kind kind = runtime_command_kind::invalid;
    std::uint32_t logical_node = 0;
    storage_object_id object{};
    residency_id residency{};
    std::uint64_t bytes = 0;
    std::uint32_t dependency_begin = 0;
    std::uint32_t dependency_count = 0;
    std::uint64_t provider_cookie = 0;
};

struct runtime_command_program {
    array_view<runtime_command> commands{};
    array_view<std::uint32_t> dependencies{};
};

enum class runtime_command_state : std::uint8_t {
    pending = 0,
    running,
    complete,
    failed,
};

[[nodiscard]] bool valid_runtime_command_program(
    const runtime_command_program &program) noexcept;

[[nodiscard]] status_code claim_ready_commands(
    const runtime_command_program &program, runtime_command_state *states,
    std::size_t state_count, std::uint32_t *ready_indices,
    std::size_t ready_capacity, std::size_t *ready_count) noexcept;

[[nodiscard]] status_code finish_runtime_command(
    runtime_command_state *states, std::size_t state_count,
    std::uint32_t command_index, bool success) noexcept;

static_assert(std::is_trivially_copyable_v<runtime_command>);
static_assert(std::is_trivially_copyable_v<runtime_command_program>);

} // namespace cellshard::runtime_v2
