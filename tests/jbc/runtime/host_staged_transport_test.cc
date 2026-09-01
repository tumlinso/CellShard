#include <CellShard/runtime/v2/host_staged_transport.cuh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    const std::array source{std::byte{1}, std::byte{2}, std::byte{3}};
    std::array<std::byte, 3> destination{};
    numa_transfer_record record{};
    assert(numa_copy_exact(source.data(), source.size(), 0, destination.data(),
                           destination.size(), 1, source.size(), &record)
           == status_code::success);
    assert(destination == source && record.crosses_numa_fabric);
    assert(numa_copy_exact(source.data(), source.size(), 0, destination.data(),
                           2, 0, source.size(), &record)
           == status_code::invalid_input);
}
