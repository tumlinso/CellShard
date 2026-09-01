#include <CellShard/runtime/v2/residency_lease.hh>

namespace cellshard::runtime_v2 {

status_code residency_lease_table::initialize(std::uint32_t capacity) noexcept {
    if (entries_ || capacity == 0) {
        return status_code::invalid_input;
    }
    try {
        entries_ = std::make_unique<entry[]>(capacity);
    } catch (...) {
        return status_code::allocation_failure;
    }
    capacity_ = capacity;
    return status_code::success;
}

status_code residency_lease_table::publish(
    atom_plane_resident_instance instance) noexcept {
    if (!entries_ || !valid_atom_plane_resident_instance(instance)) {
        return status_code::invalid_input;
    }
    const std::lock_guard<std::mutex> lock(mutation_mutex_);
    entry *free_entry = nullptr;
    for (std::uint32_t i = 0; i < capacity_; ++i) {
        if (entries_[i].occupied
            && entries_[i].instance.residency == instance.residency) {
            return status_code::invalid_input;
        }
        if (!entries_[i].occupied && free_entry == nullptr) {
            free_entry = &entries_[i];
        }
    }
    if (free_entry == nullptr) {
        return status_code::allocation_failure;
    }
    free_entry->instance = instance;
    free_entry->pins.store(0, std::memory_order_release);
    free_entry->incarnation = next_incarnation_++;
    if (free_entry->incarnation == 0) {
        free_entry->incarnation = next_incarnation_++;
    }
    free_entry->occupied = true;
    return status_code::success;
}

status_code residency_lease_table::acquire(
    residency_id residency, residency_lease *out) noexcept {
    if (!entries_ || !residency.valid() || out == nullptr) {
        return status_code::invalid_input;
    }
    const std::lock_guard<std::mutex> lock(mutation_mutex_);
    for (std::uint32_t i = 0; i < capacity_; ++i) {
        entry &candidate = entries_[i];
        if (!candidate.occupied || candidate.instance.residency != residency) {
            continue;
        }
        std::uint64_t pins = candidate.pins.load(std::memory_order_acquire);
        while (pins != ~std::uint64_t{0}) {
            const std::uint64_t free_bit = (~pins) & (pins + 1);
            if (candidate.pins.compare_exchange_weak(
                    pins, pins | free_bit, std::memory_order_acq_rel)) {
                *out = residency_lease{candidate.instance, i, free_bit,
                                       candidate.incarnation};
                return status_code::success;
            }
        }
        return status_code::allocation_failure;
    }
    return status_code::missing_object;
}

status_code residency_lease_table::release(residency_lease lease) noexcept {
    if (!entries_ || lease.slot >= capacity_ || lease.pin_mask == 0
        || (lease.pin_mask & (lease.pin_mask - 1)) != 0) {
        return status_code::invalid_input;
    }
    const std::lock_guard<std::mutex> lock(mutation_mutex_);
    entry &candidate = entries_[lease.slot];
    if (!candidate.occupied || candidate.incarnation != lease.incarnation
        || candidate.instance.residency != lease.instance.residency) {
        return status_code::invalid_input;
    }
    std::uint64_t pins = candidate.pins.load(std::memory_order_acquire);
    while ((pins & lease.pin_mask) != 0) {
        if (candidate.pins.compare_exchange_weak(
                pins, pins & ~lease.pin_mask, std::memory_order_acq_rel)) {
            return status_code::success;
        }
    }
    return status_code::invalid_input;
}

status_code residency_lease_table::evict(residency_id residency) noexcept {
    if (!entries_ || !residency.valid()) {
        return status_code::invalid_input;
    }
    const std::lock_guard<std::mutex> lock(mutation_mutex_);
    for (std::uint32_t i = 0; i < capacity_; ++i) {
        entry &candidate = entries_[i];
        if (candidate.occupied && candidate.instance.residency == residency) {
            if (candidate.pins.load(std::memory_order_acquire) != 0) {
                return status_code::unsupported_capability;
            }
            candidate.occupied = false;
            candidate.instance = {};
            return status_code::success;
        }
    }
    return status_code::missing_object;
}

} // namespace cellshard::runtime_v2
