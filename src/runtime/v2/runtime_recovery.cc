#include <CellShard/runtime/v2/runtime_recovery.hh>

#include <algorithm>

namespace cellshard::runtime_v2 {
namespace {
bool requires_compensation(runtime_command_kind kind) noexcept {
    return kind == runtime_command_kind::publish_residency
        || kind == runtime_command_kind::pin_residency
        || kind == runtime_command_kind::evict_residency;
}
} // namespace

status_code plan_runtime_recovery(
    const runtime_command_program &program,
    array_view<runtime_journal_record> journal, std::uint64_t transaction_id,
    content_digest topology_identity, runtime_command_state *states,
    std::size_t state_capacity, std::uint32_t *compensations,
    std::size_t compensation_capacity, runtime_recovery_plan *out) noexcept {
    if (!valid_runtime_command_program(program) || journal.empty()
        || transaction_id == 0
        || topology_identity.algorithm == digest_algorithm::none
        || !valid_content_digest(topology_identity) || states == nullptr
        || state_capacity < program.commands.size || compensations == nullptr
        || out == nullptr) {
        return status_code::invalid_input;
    }
    std::fill(states, states + program.commands.size,
              runtime_command_state::pending);
    std::size_t compensation_count = 0;
    std::uint64_t last_sequence = 0;
    for (const runtime_journal_record &record : journal) {
        if (record.transaction_id != transaction_id || record.sequence == 0
            || record.sequence <= last_sequence
            || record.command_index >= program.commands.size
            || record.phase == runtime_journal_phase::invalid
            || record.topology_identity != topology_identity) {
            return status_code::corruption;
        }
        last_sequence = record.sequence;
        runtime_command_state &state = states[record.command_index];
        switch (record.phase) {
        case runtime_journal_phase::started:
            if (state != runtime_command_state::pending) {
                return status_code::corruption;
            }
            state = runtime_command_state::running;
            break;
        case runtime_journal_phase::committed:
            if (state != runtime_command_state::running) {
                return status_code::corruption;
            }
            state = runtime_command_state::complete;
            break;
        case runtime_journal_phase::rolled_back:
            if (state != runtime_command_state::running) {
                return status_code::corruption;
            }
            state = runtime_command_state::pending;
            break;
        case runtime_journal_phase::invalid:
            return status_code::corruption;
        }
    }
    for (std::size_t i = 0; i < program.commands.size; ++i) {
        if (states[i] != runtime_command_state::running) {
            continue;
        }
        if (requires_compensation(program.commands[i].kind)) {
            if (compensation_count == compensation_capacity) {
                return status_code::allocation_failure;
            }
            compensations[compensation_count++] = static_cast<std::uint32_t>(i);
        }
        states[i] = runtime_command_state::pending;
    }
    *out = runtime_recovery_plan{{compensations, compensation_count},
                                 last_sequence};
    return status_code::success;
}

} // namespace cellshard::runtime_v2
