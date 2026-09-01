#include <CellShard/runtime/v2/command_ir.hh>

namespace cellshard::runtime_v2 {

bool valid_runtime_command_program(
    const runtime_command_program &program) noexcept {
    if (program.commands.empty()) {
        return false;
    }
    for (std::size_t i = 0; i < program.commands.size; ++i) {
        const auto &command = program.commands[i];
        if (command.id == 0 || command.kind == runtime_command_kind::invalid
            || command.logical_node == 0
            || command.dependency_begin > program.dependencies.size
            || command.dependency_count
                   > program.dependencies.size - command.dependency_begin) {
            return false;
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (program.commands[j].id == command.id) {
                return false;
            }
        }
        for (std::uint32_t d = 0; d < command.dependency_count; ++d) {
            const std::uint32_t dependency =
                program.dependencies[command.dependency_begin + d];
            if (dependency >= i) {
                return false;
            }
            for (std::uint32_t previous = 0; previous < d; ++previous) {
                if (program.dependencies[command.dependency_begin + previous]
                    == dependency) {
                    return false;
                }
            }
        }
    }
    return true;
}

status_code claim_ready_commands(
    const runtime_command_program &program, runtime_command_state *states,
    std::size_t state_count, std::uint32_t *ready_indices,
    std::size_t ready_capacity, std::size_t *ready_count) noexcept {
    if (!valid_runtime_command_program(program) || states == nullptr
        || state_count != program.commands.size || ready_indices == nullptr
        || ready_capacity == 0 || ready_count == nullptr) {
        return status_code::invalid_input;
    }
    *ready_count = 0;
    for (std::size_t i = 0; i < program.commands.size
         && *ready_count < ready_capacity; ++i) {
        if (states[i] != runtime_command_state::pending) {
            continue;
        }
        const auto &command = program.commands[i];
        bool ready = true;
        for (std::uint32_t d = 0; d < command.dependency_count; ++d) {
            const runtime_command_state dependency_state =
                states[program.dependencies[command.dependency_begin + d]];
            if (dependency_state == runtime_command_state::failed) {
                states[i] = runtime_command_state::failed;
                ready = false;
                break;
            }
            ready &= dependency_state == runtime_command_state::complete;
        }
        if (ready) {
            states[i] = runtime_command_state::running;
            ready_indices[(*ready_count)++] = static_cast<std::uint32_t>(i);
        }
    }
    return status_code::success;
}

status_code finish_runtime_command(
    runtime_command_state *states, std::size_t state_count,
    std::uint32_t command_index, bool success) noexcept {
    if (states == nullptr || command_index >= state_count
        || states[command_index] != runtime_command_state::running) {
        return status_code::invalid_input;
    }
    states[command_index] = success ? runtime_command_state::complete
                                    : runtime_command_state::failed;
    return status_code::success;
}

} // namespace cellshard::runtime_v2
