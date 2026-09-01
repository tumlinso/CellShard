#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::artifact::atom_store {

enum class digest_algorithm_v1 : std::uint32_t {
    sha256 = 1,
};

struct content_digest_v1 {
    digest_algorithm_v1 algorithm = digest_algorithm_v1::sha256;
    std::uint32_t digest_bytes = 32;
    std::array<std::byte, 32> bytes{};
};

[[nodiscard]] constexpr bool valid_content_digest_v1(
    const content_digest_v1 &digest) noexcept {
    return digest.algorithm == digest_algorithm_v1::sha256
        && digest.digest_bytes == digest.bytes.size();
}

namespace detail {

[[nodiscard]] constexpr std::uint32_t rotate_right(std::uint32_t value,
                                                    std::uint32_t count) noexcept {
    return (value >> count) | (value << (32u - count));
}

inline void sha256_compress(const std::byte *block,
                            std::uint32_t state[8]) noexcept {
    static constexpr std::uint32_t constants[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    std::uint32_t words[64]{};
    for (std::uint32_t index = 0; index < 16; ++index) {
        const auto offset = index * 4;
        words[index] = static_cast<std::uint32_t>(block[offset]) << 24u
            | static_cast<std::uint32_t>(block[offset + 1]) << 16u
            | static_cast<std::uint32_t>(block[offset + 2]) << 8u
            | static_cast<std::uint32_t>(block[offset + 3]);
    }
    for (std::uint32_t index = 16; index < 64; ++index) {
        const auto s0 = rotate_right(words[index - 15], 7)
            ^ rotate_right(words[index - 15], 18)
            ^ (words[index - 15] >> 3u);
        const auto s1 = rotate_right(words[index - 2], 17)
            ^ rotate_right(words[index - 2], 19)
            ^ (words[index - 2] >> 10u);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    auto a = state[0]; auto b = state[1]; auto c = state[2]; auto d = state[3];
    auto e = state[4]; auto f = state[5]; auto g = state[6]; auto h = state[7];
    for (std::uint32_t index = 0; index < 64; ++index) {
        const auto upper = rotate_right(e, 6) ^ rotate_right(e, 11)
            ^ rotate_right(e, 25);
        const auto choice = (e & f) ^ (~e & g);
        const auto temporary1 = h + upper + choice + constants[index] + words[index];
        const auto lower = rotate_right(a, 2) ^ rotate_right(a, 13)
            ^ rotate_right(a, 22);
        const auto majority = (a & b) ^ (a & c) ^ (b & c);
        const auto temporary2 = lower + majority;
        h = g; g = f; f = e; e = d + temporary1;
        d = c; c = b; b = a; a = temporary1 + temporary2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

} // namespace detail

[[nodiscard]] inline content_digest_v1 sha256_digest_v1(
    const std::byte *data, std::size_t size) noexcept {
    std::uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                              0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                              0x1f83d9abu, 0x5be0cd19u};
    std::size_t offset = 0;
    while (size - offset >= 64) {
        detail::sha256_compress(data + offset, state);
        offset += 64;
    }
    std::array<std::byte, 128> tail{};
    const auto remaining = size - offset;
    for (std::size_t index = 0; index < remaining; ++index)
        tail[index] = data[offset + index];
    tail[remaining] = std::byte{0x80};
    const auto tail_bytes = remaining < 56 ? 64u : 128u;
    const auto bit_count = static_cast<std::uint64_t>(size) * 8u;
    for (std::uint32_t index = 0; index < 8; ++index)
        tail[tail_bytes - 1u - index]
            = static_cast<std::byte>(bit_count >> (index * 8u));
    detail::sha256_compress(tail.data(), state);
    if (tail_bytes == 128) detail::sha256_compress(tail.data() + 64, state);
    content_digest_v1 digest{};
    for (std::uint32_t word = 0; word < 8; ++word)
        for (std::uint32_t byte = 0; byte < 4; ++byte)
            digest.bytes[word * 4 + byte]
                = static_cast<std::byte>(state[word] >> (24u - byte * 8u));
    return digest;
}

static_assert(std::is_standard_layout<content_digest_v1>::value);
static_assert(std::is_trivially_copyable<content_digest_v1>::value);

} // namespace cellshard::artifact::atom_store
