#pragma once

#include <CellShard/runtime/v2/atom_source.hh>

#include <cstddef>
#include <cstdint>

namespace cellshard::runtime_v2 {

struct transport_provider {
    source_provider_id source_provider{};
    source_location_id source_location{};
    std::uint32_t destination_node = 0;
    std::uint32_t priority = 0;
    atom_source_ref source{};
};

[[nodiscard]] constexpr bool valid_transport_provider(
    const transport_provider &provider) noexcept {
    return provider.source_provider.valid() && provider.source_location.valid()
        && provider.destination_node != 0 && provider.priority != 0
        && valid_atom_source(provider.source);
}

class transport_provider_registry {
public:
    constexpr transport_provider_registry(
        transport_provider *storage, std::size_t capacity) noexcept
        : storage_(storage), capacity_(capacity) {}

    [[nodiscard]] status_code add(transport_provider provider) noexcept {
        if (!valid_transport_provider(provider) || storage_ == nullptr) {
            return status_code::invalid_input;
        }
        for (std::size_t i = 0; i < size_; ++i) {
            if (storage_[i].source_provider == provider.source_provider
                && storage_[i].source_location == provider.source_location
                && storage_[i].destination_node == provider.destination_node
                && storage_[i].priority == provider.priority) {
                return status_code::invalid_input;
            }
        }
        if (size_ == capacity_) {
            return status_code::allocation_failure;
        }
        storage_[size_++] = provider;
        return status_code::success;
    }

    [[nodiscard]] const transport_provider *resolve(
        source_provider_id provider, source_location_id location,
        std::uint32_t destination_node) const noexcept {
        const transport_provider *selected = nullptr;
        for (std::size_t i = 0; i < size_; ++i) {
            const auto &candidate = storage_[i];
            if (candidate.source_provider == provider
                && candidate.source_location == location
                && candidate.destination_node == destination_node
                && (selected == nullptr
                    || candidate.priority > selected->priority)) {
                selected = &candidate;
            }
        }
        return selected;
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }

private:
    transport_provider *storage_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t size_ = 0;
};

} // namespace cellshard::runtime_v2
