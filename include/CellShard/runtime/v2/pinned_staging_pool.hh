#pragma once

#include <CellShard/identity.hh>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace cellshard::runtime_v2 {

struct pinned_staging_allocator_ops {
    status_code (*allocate)(void *context, std::uint32_t numa_id,
                            std::size_t bytes, std::size_t alignment,
                            void **out) noexcept = nullptr;
    void (*deallocate)(void *context, void *allocation) noexcept = nullptr;
};

struct pinned_staging_allocator_ref {
    void *context = nullptr;
    const pinned_staging_allocator_ops *ops = nullptr;
};

[[nodiscard]] pinned_staging_allocator_ref cuda_pinned_staging_allocator() noexcept;

struct pinned_staging_lease {
    std::byte *data = nullptr;
    std::size_t bytes = 0;
    std::uint32_t numa_id = 0;
    std::uint32_t slot = 0;
};

class pinned_staging_pool {
public:
    pinned_staging_pool() noexcept = default;
    ~pinned_staging_pool() noexcept;
    pinned_staging_pool(const pinned_staging_pool &) = delete;
    pinned_staging_pool &operator=(const pinned_staging_pool &) = delete;

    [[nodiscard]] status_code initialize(
        std::uint32_t numa_id, std::size_t buffer_bytes,
        std::uint32_t buffer_count, std::size_t alignment,
        pinned_staging_allocator_ref allocator) noexcept;
    [[nodiscard]] status_code acquire(std::size_t minimum_bytes,
                                      pinned_staging_lease *out) noexcept;
    [[nodiscard]] status_code release(pinned_staging_lease lease) noexcept;
    void reset() noexcept;

private:
    struct slot_state {
        void *allocation = nullptr;
        std::atomic<bool> in_use{false};
    };

    std::unique_ptr<slot_state[]> slots_{};
    std::uint32_t slot_count_ = 0;
    std::uint32_t numa_id_ = 0;
    std::size_t buffer_bytes_ = 0;
    pinned_staging_allocator_ref allocator_{};
};

} // namespace cellshard::runtime_v2
