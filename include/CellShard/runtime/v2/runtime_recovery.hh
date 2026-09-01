#pragma once

#include <CellShard/runtime/v2/command_ir.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::runtime_v2 {

enum class runtime_journal_phase : std::uint8_t {
    invalid = 0,
    started,
    committed,
    rolled_back,
};

struct runtime_journal_record {
    std::uint64_t transaction_id = 0;
    std::uint64_t sequence = 0;
    std::uint32_t command_index = 0;
    runtime_journal_phase phase = runtime_journal_phase::invalid;
    content_digest topology_identity{};
};

struct runtime_recovery_plan {
    array_view<std::uint32_t> compensate_commands{};
    std::uint64_t last_sequence = 0;
};

[[nodiscard]] status_code plan_runtime_recovery(
    const runtime_command_program &program,
    array_view<runtime_journal_record> journal, std::uint64_t transaction_id,
    content_digest topology_identity, runtime_command_state *state_storage,
    std::size_t state_capacity, std::uint32_t *compensation_storage,
    std::size_t compensation_capacity, runtime_recovery_plan *out) noexcept;

static_assert(std::is_trivially_copyable_v<runtime_journal_record>);
static_assert(std::is_trivially_copyable_v<runtime_recovery_plan>);

} // namespace cellshard::runtime_v2
