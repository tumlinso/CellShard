#pragma once

#include <CellShard/artifact/image.hh>
#include <CellShard/runtime/source/payload_source.hh>

namespace cellshard {

struct host_allocator_ops {
    void *(*allocate)(void *context, std::size_t bytes,
                      std::size_t alignment) noexcept = nullptr;
    void (*deallocate)(void *context, void *allocation) noexcept = nullptr;
};

struct host_allocator_ref {
    void *context = nullptr;
    const host_allocator_ops *ops = nullptr;
};

[[nodiscard]] host_allocator_ref default_host_allocator() noexcept;

struct host_residency_view {
    image_id image{};
    const std::byte *payload = nullptr;
    std::size_t payload_bytes = 0;
    std::uint32_t alignment = 0;
    content_digest payload_digest{};
};

class host_residency {
public:
    host_residency() noexcept = default;
    ~host_residency() noexcept;
    host_residency(const host_residency &) = delete;
    host_residency &operator=(const host_residency &) = delete;
    host_residency(host_residency &&other) noexcept;
    host_residency &operator=(host_residency &&other) noexcept;

    [[nodiscard]] host_residency_view view() const noexcept;
    [[nodiscard]] bool valid() const noexcept { return allocation_ != nullptr; }
    void reset() noexcept;

private:
    image_id image_{};
    void *allocation_ = nullptr;
    std::size_t payload_bytes_ = 0;
    std::uint32_t alignment_ = 0;
    content_digest digest_{};
    host_allocator_ref allocator_{};

    friend status_code load_host_residency(
        const payload_source_ref &, const storage_object_descriptor &,
        const extent_descriptor &, const image_descriptor_view &,
        host_allocator_ref, host_residency *) noexcept;
};

[[nodiscard]] status_code load_host_residency(
    const payload_source_ref &source,
    const storage_object_descriptor &object,
    const extent_descriptor &extent,
    const image_descriptor_view &image,
    host_allocator_ref allocator,
    host_residency *out) noexcept;

} // namespace cellshard
