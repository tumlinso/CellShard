#include <CellShard/runtime/v2/pinned_staging_pool.hh>

namespace cellshard::runtime_v2 {

pinned_staging_pool::~pinned_staging_pool() noexcept { reset(); }

status_code pinned_staging_pool::initialize(
    std::uint32_t numa_id, std::size_t buffer_bytes,
    std::uint32_t buffer_count, std::size_t alignment,
    pinned_staging_allocator_ref allocator) noexcept {
    if (slots_ || buffer_bytes == 0 || buffer_count == 0 || alignment == 0
        || (alignment & (alignment - 1)) != 0 || allocator.ops == nullptr
        || allocator.ops->allocate == nullptr
        || allocator.ops->deallocate == nullptr) {
        return status_code::invalid_input;
    }
    std::unique_ptr<slot_state[]> slots;
    try {
        slots = std::make_unique<slot_state[]>(buffer_count);
    } catch (...) {
        return status_code::allocation_failure;
    }
    std::uint32_t initialized = 0;
    for (; initialized < buffer_count; ++initialized) {
        const status_code status = allocator.ops->allocate(
            allocator.context, numa_id, buffer_bytes, alignment,
            &slots[initialized].allocation);
        if (!status_ok(status) || slots[initialized].allocation == nullptr) {
            for (std::uint32_t index = 0; index < initialized; ++index) {
                allocator.ops->deallocate(allocator.context,
                                          slots[index].allocation);
            }
            return status_ok(status) ? status_code::allocation_failure : status;
        }
    }
    slots_ = std::move(slots);
    slot_count_ = buffer_count;
    numa_id_ = numa_id;
    buffer_bytes_ = buffer_bytes;
    allocator_ = allocator;
    return status_code::success;
}

status_code pinned_staging_pool::acquire(
    std::size_t minimum_bytes, pinned_staging_lease *out) noexcept {
    if (!slots_ || out == nullptr || minimum_bytes == 0
        || minimum_bytes > buffer_bytes_) {
        return status_code::invalid_input;
    }
    for (std::uint32_t index = 0; index < slot_count_; ++index) {
        bool expected = false;
        if (slots_[index].in_use.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            *out = pinned_staging_lease{
                static_cast<std::byte *>(slots_[index].allocation),
                buffer_bytes_, numa_id_, index};
            return status_code::success;
        }
    }
    return status_code::allocation_failure;
}

status_code pinned_staging_pool::release(pinned_staging_lease lease) noexcept {
    if (!slots_ || lease.slot >= slot_count_
        || lease.data != slots_[lease.slot].allocation
        || lease.bytes != buffer_bytes_ || lease.numa_id != numa_id_) {
        return status_code::invalid_input;
    }
    bool expected = true;
    if (!slots_[lease.slot].in_use.compare_exchange_strong(
            expected, false, std::memory_order_acq_rel)) {
        return status_code::invalid_input;
    }
    return status_code::success;
}

void pinned_staging_pool::reset() noexcept {
    if (slots_) {
        for (std::uint32_t index = 0; index < slot_count_; ++index) {
            allocator_.ops->deallocate(allocator_.context,
                                       slots_[index].allocation);
        }
    }
    slots_.reset();
    slot_count_ = 0;
    numa_id_ = 0;
    buffer_bytes_ = 0;
    allocator_ = {};
}

} // namespace cellshard::runtime_v2
