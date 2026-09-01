#include <CellShard/runtime/v2/read_plan.hh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    const std::array ranges{
        atom_range{storage_object_id{1}, 0, 4, 8},
        atom_range{storage_object_id{1}, 8, 4, 0},
        atom_range{storage_object_id{2}, 0, 2, 12},
    };
    std::array<read_span, 3> spans{};
    std::array<read_copy, 3> copies{};
    read_plan plan{};
    assert(build_read_plan({ranges.data(), ranges.size()}, 4, 16, spans.data(),
                           spans.size(), copies.data(), copies.size(), &plan)
           == status_code::success);
    assert(plan.spans.size == 2 && plan.copies.size == 3);
    assert(plan.spans[0].bytes == 12 && plan.staging_bytes == 14);
    assert(plan.requested_bytes == 10);
    assert(plan.copies[1].span_offset == 8
           && plan.copies[1].destination_offset == 0);

    assert(build_read_plan({ranges.data(), ranges.size()}, 3, 16, spans.data(),
                           spans.size(), copies.data(), copies.size(), &plan)
           == status_code::success);
    assert(plan.spans.size == 3);
    assert(build_read_plan({ranges.data(), ranges.size()}, 4, 8, spans.data(),
                           1, copies.data(), copies.size(), &plan)
           == status_code::allocation_failure);
}
