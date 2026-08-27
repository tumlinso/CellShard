#pragma once

#include <cstdint>
#include <string>

#include <CellShard/identity.hh>

namespace cellshard {

struct storage_object_descriptor {
    storage_object_id id{};
    std::uint64_t byte_count = 0;
    content_digest object_digest{};
};

struct extent_descriptor {
    extent_id id{};
    storage_object_id object{};
    std::uint64_t object_offset = 0;
    std::uint64_t byte_count = 0;
    std::uint32_t required_alignment = 0;
    content_digest payload_digest{};
};

enum class source_capability : std::uint32_t {
    none = 0,
    exact_range_read = 1u << 0,
    stable_size = 1u << 1,
};

using source_capabilities = std::uint32_t;

[[nodiscard]] constexpr source_capabilities capability_bit(
    source_capability capability) noexcept {
    return static_cast<source_capabilities>(capability);
}

[[nodiscard]] constexpr bool has_source_capability(
    source_capabilities capabilities, source_capability capability) noexcept {
    return (capabilities & capability_bit(capability)) != 0;
}

// Cold mutable catalog metadata. Provider-specific locator text is deliberately
// separate from storage-object and extent identity.
struct source_location_descriptor {
    source_location_id id{};
    source_provider_id provider{};
    storage_object_id object{};
    source_capabilities capabilities = 0;
    std::string locator{};
};

[[nodiscard]] constexpr bool valid_power_of_two(
    std::uint32_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

[[nodiscard]] constexpr bool valid_storage_object_descriptor(
    const storage_object_descriptor &object) noexcept {
    return object.id.valid() && object.byte_count != 0
        && valid_content_digest(object.object_digest);
}

[[nodiscard]] constexpr bool valid_extent_descriptor(
    const extent_descriptor &extent,
    const storage_object_descriptor &object) noexcept {
    return valid_storage_object_descriptor(object) && extent.id.valid()
        && extent.object == object.id && extent.byte_count != 0
        && extent.object_offset <= object.byte_count
        && extent.byte_count <= object.byte_count - extent.object_offset
        && valid_power_of_two(extent.required_alignment)
        && extent.payload_digest.algorithm != digest_algorithm::none
        && valid_content_digest(extent.payload_digest);
}

[[nodiscard]] inline bool valid_source_location_descriptor(
    const source_location_descriptor &location,
    const storage_object_descriptor &object) noexcept {
    return valid_storage_object_descriptor(object) && location.id.valid()
        && location.provider.valid() && location.object == object.id
        && has_source_capability(location.capabilities,
                                 source_capability::exact_range_read)
        && !location.locator.empty();
}

} // namespace cellshard
