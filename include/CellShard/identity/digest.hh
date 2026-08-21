#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace cellshard {

enum class digest_algorithm : std::uint32_t {
    none = 0,
    // Compatibility-only corruption detection; this is not cryptographic.
    legacy_fnv1a64 = 1,
    // Values for stronger algorithms are intentionally reserved for later work.
};

struct content_digest {
    digest_algorithm algorithm = digest_algorithm::none;
    std::uint32_t used_bytes = 0;
    std::array<std::byte, 32> bytes{};
};

[[nodiscard]] constexpr bool operator==(const content_digest &lhs,
                                        const content_digest &rhs) noexcept {
    if (lhs.algorithm != rhs.algorithm || lhs.used_bytes != rhs.used_bytes) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.bytes.size(); ++index) {
        if (lhs.bytes[index] != rhs.bytes[index]) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] constexpr bool operator!=(const content_digest &lhs,
                                        const content_digest &rhs) noexcept {
    return !(lhs == rhs);
}

[[nodiscard]] constexpr bool valid_content_digest(const content_digest &digest) noexcept {
    if (digest.used_bytes > digest.bytes.size()) {
        return false;
    }

    std::uint32_t required_bytes = 0;
    switch (digest.algorithm) {
    case digest_algorithm::none:
        required_bytes = 0;
        break;
    case digest_algorithm::legacy_fnv1a64:
        required_bytes = sizeof(std::uint64_t);
        break;
    default:
        return false;
    }

    if (digest.used_bytes != required_bytes) {
        return false;
    }
    for (std::size_t index = digest.used_bytes; index < digest.bytes.size(); ++index) {
        if (digest.bytes[index] != std::byte{0}) {
            return false;
        }
    }
    return true;
}

} // namespace cellshard
