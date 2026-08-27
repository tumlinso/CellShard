#include <CellShard/runtime/residency/host.hh>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <utility>

namespace cellshard {
namespace {
constexpr std::uint64_t fnv_offset = 1469598103934665603ull;
constexpr std::uint64_t fnv_prime = 1099511628211ull;

void *allocate_default(void *, std::size_t bytes, std::size_t alignment) noexcept {
    void *result = nullptr;
    const std::size_t effective_alignment =
        std::max(alignment, sizeof(void *));
    return ::posix_memalign(&result, effective_alignment, bytes) == 0
        ? result : nullptr;
}
void deallocate_default(void *, void *allocation) noexcept { std::free(allocation); }
const host_allocator_ops default_ops{&allocate_default, &deallocate_default};

content_digest digest_bytes(const std::byte *bytes, std::size_t count) noexcept {
    std::uint64_t hash = fnv_offset;
    for (std::size_t index = 0; index < count; ++index) {
        hash ^= std::to_integer<unsigned char>(bytes[index]);
        hash *= fnv_prime;
    }
    if (hash == 0) hash = 1;
    content_digest result{};
    result.algorithm = digest_algorithm::legacy_fnv1a64;
    result.used_bytes = sizeof(hash);
    for (unsigned shift = 0; shift < 64; shift += 8) {
        result.bytes[shift / 8] = std::byte((hash >> shift) & 0xffu);
    }
    return result;
}
}

host_allocator_ref default_host_allocator() noexcept {
    return {nullptr, &default_ops};
}

host_residency::~host_residency() noexcept { reset(); }
host_residency::host_residency(host_residency &&other) noexcept { *this = std::move(other); }
host_residency &host_residency::operator=(host_residency &&other) noexcept {
    if (this != &other) {
        reset();
        image_ = other.image_;
        allocation_ = other.allocation_;
        payload_bytes_ = other.payload_bytes_;
        alignment_ = other.alignment_;
        digest_ = other.digest_;
        allocator_ = other.allocator_;
        other.allocation_ = nullptr;
        other.payload_bytes_ = 0;
    }
    return *this;
}
void host_residency::reset() noexcept {
    if (allocation_ != nullptr && allocator_.ops != nullptr) {
        allocator_.ops->deallocate(allocator_.context, allocation_);
    }
    allocation_ = nullptr;
    payload_bytes_ = 0;
}
host_residency_view host_residency::view() const noexcept {
    return {image_, static_cast<const std::byte *>(allocation_), payload_bytes_,
            alignment_, digest_};
}

status_code load_host_residency(
    const payload_source_ref &source, const storage_object_descriptor &object,
    const extent_descriptor &extent, const image_descriptor_view &image,
    host_allocator_ref allocator, host_residency *out) noexcept {
    if (out == nullptr || !valid_image_descriptor(image)
        || !valid_extent_descriptor(extent, object)
        || source.object != object.id || extent.byte_count != image.stored_bytes
        || extent.required_alignment != image.required_alignment
        || extent.payload_digest != image.payload_digest || allocator.ops == nullptr
        || allocator.ops->allocate == nullptr || allocator.ops->deallocate == nullptr
        || extent.byte_count > std::numeric_limits<std::size_t>::max()) {
        return status_code::invalid_input;
    }
    out->reset();
    void *allocation = allocator.ops->allocate(
        allocator.context, static_cast<std::size_t>(extent.byte_count),
        extent.required_alignment);
    if (allocation == nullptr) return status_code::allocation_failure;
    const status_code read_status = read_extent_exact(
        source, extent, object, static_cast<std::byte *>(allocation),
        static_cast<std::size_t>(extent.byte_count));
    if (read_status != status_code::success) {
        allocator.ops->deallocate(allocator.context, allocation);
        return read_status;
    }
    if (digest_bytes(static_cast<const std::byte *>(allocation),
                     static_cast<std::size_t>(extent.byte_count))
        != image.payload_digest) {
        allocator.ops->deallocate(allocator.context, allocation);
        return status_code::corruption;
    }
    out->image_ = image.id;
    out->allocation_ = allocation;
    out->payload_bytes_ = static_cast<std::size_t>(extent.byte_count);
    out->alignment_ = extent.required_alignment;
    out->digest_ = image.payload_digest;
    out->allocator_ = allocator;
    return status_code::success;
}

} // namespace cellshard
