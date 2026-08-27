#pragma once

#include <cstddef>
#include <cstdint>

#include <CellShard/artifact/image.hh>

namespace cellshard {

inline constexpr std::uint32_t image_envelope_schema_version = 2;
inline constexpr std::uint32_t image_envelope_endian_marker = 0x01020304u;
inline constexpr std::size_t image_envelope_fixed_header_bytes = 192;
inline constexpr std::uint32_t image_envelope_max_alignment = 1u << 20;

struct decoded_image_envelope {
    image_descriptor descriptor{};
    array_view<std::byte> payload{};
    std::uint64_t payload_offset = 0;
    std::uint64_t envelope_checksum = 0;
};

[[nodiscard]] status_code encoded_image_envelope_size(
    const image_descriptor_view &descriptor,
    std::size_t payload_bytes,
    std::size_t *out_bytes) noexcept;

[[nodiscard]] status_code encode_image_envelope(
    const image_descriptor_view &descriptor,
    array_view<std::byte> payload,
    std::byte *output,
    std::size_t output_bytes) noexcept;

[[nodiscard]] status_code decode_image_envelope(
    const std::byte *input,
    std::size_t input_bytes,
    decoded_image_envelope *out);

} // namespace cellshard
