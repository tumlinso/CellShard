#include <CellShard/runtime/v2/command_ir.hh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    const std::array dependencies{0U, 0U, 1U, 2U};
    const std::array commands{
        runtime_command{1, runtime_command_kind::read_atoms, 1, {}, {}, 8, 0, 0, 1},
        runtime_command{2, runtime_command_kind::stage_host, 1, {}, {}, 8, 0, 1, 2},
        runtime_command{3, runtime_command_kind::transport, 2, {}, {}, 8, 1, 1, 3},
        runtime_command{4, runtime_command_kind::barrier, 2, {}, {}, 0, 2, 2, 0},
    };
    const runtime_command_program program{{commands.data(), commands.size()},
                                          {dependencies.data(),
                                           dependencies.size()}};
    assert(valid_runtime_command_program(program));
    std::array<runtime_command_state, 4> states{};
    std::array<std::uint32_t, 4> ready{};
    std::size_t count = 0;
    assert(claim_ready_commands(program, states.data(), states.size(),
                                ready.data(), ready.size(), &count)
           == status_code::success);
    assert(count == 1 && ready[0] == 0);
    assert(finish_runtime_command(states.data(), states.size(), 0, true)
           == status_code::success);
    assert(claim_ready_commands(program, states.data(), states.size(),
                                ready.data(), ready.size(), &count)
           == status_code::success);
    assert(count == 2 && ready[0] == 1 && ready[1] == 2);
    assert(finish_runtime_command(states.data(), states.size(), 1, false)
           == status_code::success);
    assert(finish_runtime_command(states.data(), states.size(), 2, true)
           == status_code::success);
    assert(claim_ready_commands(program, states.data(), states.size(),
                                ready.data(), ready.size(), &count)
           == status_code::success);
    assert(count == 0 && states[3] == runtime_command_state::failed);
}
