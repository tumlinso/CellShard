#pragma once

#include <cstddef>
#include <cstdint>

#include <CellShard/artifact/extent.hh>

namespace cellshard {

struct exact_read_request {
    storage_object_id object{};
    std::uint64_t object_offset = 0;
    std::uint64_t byte_count = 0;
    std::byte *destination = nullptr;
    std::size_t destination_bytes = 0;
};

struct payload_source_ops {
    status_code (*read_exact)(void *context,
                              const exact_read_request &request) noexcept = nullptr;
};

// Non-owning type-erased boundary. The provider and its context outlive this
// value; no allocation, inheritance, or std::function is hidden here.
struct payload_source_ref {
    void *context = nullptr;
    const payload_source_ops *ops = nullptr;
    source_provider_id provider{};
    source_location_id location{};
    storage_object_id object{};
    std::uint64_t object_bytes = 0;
    source_capabilities capabilities = 0;
};

[[nodiscard]] constexpr bool valid_payload_source_ref(
    const payload_source_ref &source) noexcept {
    return source.context != nullptr && source.ops != nullptr
        && source.ops->read_exact != nullptr && source.provider.valid()
        && source.location.valid() && source.object.valid()
        && source.object_bytes != 0
        && has_source_capability(source.capabilities,
                                 source_capability::exact_range_read);
}

[[nodiscard]] constexpr bool valid_exact_read_request(
    const exact_read_request &request,
    const payload_source_ref &source) noexcept {
    return valid_payload_source_ref(source) && request.object == source.object
        && request.byte_count != 0 && request.destination != nullptr
        && request.byte_count <= request.destination_bytes
        && request.object_offset <= source.object_bytes
        && request.byte_count <= source.object_bytes - request.object_offset;
}

[[nodiscard]] inline status_code read_exact(
    const payload_source_ref &source,
    const exact_read_request &request) noexcept {
    if (!valid_exact_read_request(request, source)) {
        return status_code::invalid_input;
    }
    return source.ops->read_exact(source.context, request);
}

[[nodiscard]] inline status_code read_extent_exact(
    const payload_source_ref &source,
    const extent_descriptor &extent,
    const storage_object_descriptor &object,
    std::byte *destination,
    std::size_t destination_bytes) noexcept {
    if (!valid_extent_descriptor(extent, object) || source.object != object.id) {
        return status_code::invalid_input;
    }
    return read_exact(source, exact_read_request{
        object.id,
        extent.object_offset,
        extent.byte_count,
        destination,
        destination_bytes,
    });
}

} // namespace cellshard
