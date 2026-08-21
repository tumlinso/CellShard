#pragma once

#include <cstddef>
#include <cstdint>

#include "identity/digest.hh"
#include "identity/strong_id.hh"

namespace cellshard {

template<typename T>
struct array_view {
    const T *data = nullptr;
    std::size_t size = 0;

    [[nodiscard]] constexpr bool empty() const noexcept { return size == 0; }
    [[nodiscard]] constexpr const T *begin() const noexcept { return data; }
    [[nodiscard]] constexpr const T *end() const noexcept {
        return size == 0 ? data : data + size;
    }
    [[nodiscard]] constexpr const T &operator[](std::size_t index) const noexcept {
        return data[index];
    }
};

enum class status_code : std::uint32_t {
    success = 0,
    invalid_input,
    missing_object,
    short_read,
    corruption,
    incompatible_image,
    unsupported_capability,
    allocation_failure,
    cuda_failure,
};

[[nodiscard]] constexpr bool status_ok(status_code status) noexcept {
    return status == status_code::success;
}

} // namespace cellshard
