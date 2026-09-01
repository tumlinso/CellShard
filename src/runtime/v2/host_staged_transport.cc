#include <CellShard/runtime/v2/host_staged_transport.cuh>

#include <cstring>

namespace cellshard::runtime_v2 {

status_code numa_copy_exact(
    const std::byte *source, std::uint64_t source_bytes,
    std::uint32_t source_numa, std::byte *destination,
    std::uint64_t destination_bytes, std::uint32_t destination_numa,
    std::uint64_t bytes, numa_transfer_record *record) noexcept {
    if (source == nullptr || destination == nullptr || bytes == 0
        || bytes > source_bytes || bytes > destination_bytes
        || record == nullptr) {
        return status_code::invalid_input;
    }
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
    *record = numa_transfer_record{source_numa, destination_numa, bytes,
                                   source_numa != destination_numa};
    return status_code::success;
}

} // namespace cellshard::runtime_v2
