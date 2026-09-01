#include <CellShard/runtime/v2/runtime_recovery.hh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    const std::array dependencies{0U};
    const std::array commands{
        runtime_command{1, runtime_command_kind::read_atoms, 1, {}, {}, 8, 0, 0, 1},
        runtime_command{2, runtime_command_kind::publish_residency, 1, {}, residency_id{2}, 8, 0, 1, 2},
    };
    const runtime_command_program program{{commands.data(), commands.size()},
                                          {dependencies.data(), dependencies.size()}};
    content_digest identity{};
    identity.algorithm = digest_algorithm::legacy_fnv1a64;
    identity.used_bytes = 8;
    std::array journal{
        runtime_journal_record{7, 1, 0, runtime_journal_phase::started, identity},
        runtime_journal_record{7, 2, 0, runtime_journal_phase::committed, identity},
        runtime_journal_record{7, 3, 1, runtime_journal_phase::started, identity},
    };
    std::array<runtime_command_state, 2> states{};
    std::array<std::uint32_t, 2> compensations{};
    runtime_recovery_plan recovery{};
    assert(plan_runtime_recovery(
               program, {journal.data(), journal.size()}, 7, identity,
               states.data(), states.size(), compensations.data(),
               compensations.size(), &recovery) == status_code::success);
    assert(states[0] == runtime_command_state::complete);
    assert(states[1] == runtime_command_state::pending);
    assert(recovery.compensate_commands.size == 1
           && recovery.compensate_commands[0] == 1);
    journal[2].sequence = 2;
    assert(plan_runtime_recovery(
               program, {journal.data(), journal.size()}, 7, identity,
               states.data(), states.size(), compensations.data(),
               compensations.size(), &recovery) == status_code::corruption);
}
