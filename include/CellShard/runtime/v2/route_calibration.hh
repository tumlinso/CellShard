#pragma once

#include <CellShard/identity.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::runtime_v2 {

struct route_calibration {
    route_table_id route{};
    content_digest topology_identity{};
    std::uint32_t source_node = 0;
    std::uint32_t destination_node = 0;
    std::uint64_t minimum_bytes = 0;
    std::uint64_t maximum_bytes = 0;
    std::uint64_t fixed_nanoseconds = 0;
    std::uint64_t picoseconds_per_byte = 0;
    std::uint64_t p99_fixed_nanoseconds = 0;
    std::uint32_t sample_count = 0;
};

[[nodiscard]] constexpr bool valid_route_calibration(
    const route_calibration &record) noexcept {
    return record.route.valid()
        && record.topology_identity.algorithm != digest_algorithm::none
        && valid_content_digest(record.topology_identity)
        && record.source_node != 0 && record.destination_node != 0
        && record.source_node != record.destination_node
        && record.minimum_bytes != 0
        && record.minimum_bytes <= record.maximum_bytes
        && record.fixed_nanoseconds != 0
        && record.picoseconds_per_byte != 0
        && record.p99_fixed_nanoseconds >= record.fixed_nanoseconds
        && record.sample_count >= 2;
}

[[nodiscard]] constexpr std::uint64_t calibrated_route_nanoseconds(
    const route_calibration &record, std::uint64_t bytes) noexcept {
    if (!valid_route_calibration(record) || bytes < record.minimum_bytes
        || bytes > record.maximum_bytes) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    const std::uint64_t transfer_ns = bytes >
            (std::numeric_limits<std::uint64_t>::max() / record.picoseconds_per_byte)
        ? std::numeric_limits<std::uint64_t>::max()
        : (bytes * record.picoseconds_per_byte + 999) / 1000;
    return transfer_ns > std::numeric_limits<std::uint64_t>::max()
                               - record.fixed_nanoseconds
        ? std::numeric_limits<std::uint64_t>::max()
        : record.fixed_nanoseconds + transfer_ns;
}

static_assert(std::is_trivially_copyable_v<route_calibration>);

} // namespace cellshard::runtime_v2
