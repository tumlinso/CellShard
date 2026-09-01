#pragma once

#include <CellShard/identity.hh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::runtime_v2 {

struct atom_range {
    storage_object_id object{};
    std::uint64_t object_offset = 0;
    std::uint64_t bytes = 0;
    std::uint64_t destination_offset = 0;
};

struct atom_source_request {
    array_view<atom_range> ranges{};
    std::byte *destination = nullptr;
    std::uint64_t destination_bytes = 0;
};

struct atom_request_token {
    std::uint64_t value = 0;
    [[nodiscard]] constexpr bool valid() const noexcept { return value != 0; }
};

enum class atom_request_state : std::uint8_t {
    invalid = 0,
    pending,
    complete,
    failed,
    cancelled,
};

struct atom_source_ops {
    status_code (*submit)(void *context, const atom_source_request &request,
                          atom_request_token *token) noexcept = nullptr;
    atom_request_state (*query)(void *context,
                                atom_request_token token) noexcept = nullptr;
    status_code (*cancel)(void *context,
                          atom_request_token token) noexcept = nullptr;
};

struct atom_source_ref {
    void *context = nullptr;
    const atom_source_ops *ops = nullptr;
};

[[nodiscard]] inline bool valid_atom_source_request(
    const atom_source_request &request) noexcept {
    if (request.ranges.empty() || request.destination == nullptr
        || request.destination_bytes == 0) {
        return false;
    }
    for (std::size_t i = 0; i < request.ranges.size; ++i) {
        const auto &range = request.ranges[i];
        if (!range.object.valid() || range.bytes == 0
            || range.object_offset > std::numeric_limits<std::uint64_t>::max()
                                         - range.bytes
            || range.destination_offset > request.destination_bytes
            || range.bytes > request.destination_bytes - range.destination_offset) {
            return false;
        }
        const std::uint64_t begin = range.destination_offset;
        const std::uint64_t end = begin + range.bytes;
        for (std::size_t j = 0; j < i; ++j) {
            const std::uint64_t previous_begin =
                request.ranges[j].destination_offset;
            const std::uint64_t previous_end =
                previous_begin + request.ranges[j].bytes;
            if (begin < previous_end && previous_begin < end) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] constexpr bool valid_atom_source(
    atom_source_ref source) noexcept {
    return source.ops != nullptr && source.ops->submit != nullptr
        && source.ops->query != nullptr && source.ops->cancel != nullptr;
}

static_assert(std::is_trivially_copyable_v<atom_range>);
static_assert(std::is_trivially_copyable_v<atom_source_request>);
static_assert(std::is_trivially_copyable_v<atom_source_ref>);

} // namespace cellshard::runtime_v2
