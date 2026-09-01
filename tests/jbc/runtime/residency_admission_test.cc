#include <CellShard/runtime/v2/residency_admission.hh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    const std::array candidates{
        eviction_candidate{residency_id{1}, 20, 100, 7, 0},
        eviction_candidate{residency_id{2}, 40, 10, 9, 1},
        eviction_candidate{residency_id{3}, 30, 20, 5, 0},
    };
    std::array<residency_id, 3> victims{};
    std::array<bool, 3> selected{};
    residency_admission_plan plan{};
    const residency_admission_request request{
        100, 90, 45, {candidates.data(), candidates.size()}};
    assert(plan_residency_admission(request, victims.data(), victims.size(),
                                    selected.data(), selected.size(), &plan)
           == status_code::success);
    assert(plan.evictions.size == 2);
    assert(plan.evictions[0] == residency_id{3});
    assert(plan.evictions[1] == residency_id{1});
    assert(plan.evicted_bytes == 50 && plan.reconstruction_nanoseconds == 120);

    auto pinned = candidates;
    pinned[0].active_pins = 1;
    const residency_admission_request blocked{
        100, 90, 45, {pinned.data(), pinned.size()}};
    assert(plan_residency_admission(blocked, victims.data(), victims.size(),
                                    selected.data(), selected.size(), &plan)
           == status_code::unsupported_capability);
}
