#pragma once
#include "fixtures.hpp"
#include <array>
namespace cellshard::jbc::validation {
struct node_segment {
    global_id logical_node_id = 0;
    global_id object_id = 0;
    global_id mount_fingerprint = 0;
    std::uint64_t byte_begin = 0;
    std::uint64_t byte_end = 0;
    std::array<std::uint64_t, 4> object_digest{};
    bool mounted_at_block = false;
};
inline bool valid_two_node_slice(const node_segment (&segments)[2],
                                 std::uint64_t object_size) noexcept {
    return segments[0].logical_node_id != 0 && segments[1].logical_node_id != 0 &&
           segments[0].logical_node_id != segments[1].logical_node_id &&
           segments[0].object_id != 0 && segments[0].object_id == segments[1].object_id &&
           segments[0].mount_fingerprint != 0 && segments[1].mount_fingerprint != 0 &&
           segments[0].mounted_at_block && segments[1].mounted_at_block &&
           segments[0].byte_begin == 0 && segments[0].byte_end == segments[1].byte_begin &&
           segments[1].byte_end == object_size && segments[0].object_digest == segments[1].object_digest;
}
}  // namespace cellshard::jbc::validation
