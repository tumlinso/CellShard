#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <CellShard/artifact/extent.hh>
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

struct image_cspack_entry_source {
    extent_id extent{};
    image_descriptor_view descriptor{};
    array_view<std::byte> payload{};
};

struct published_image_cspack {
    storage_object_descriptor object{};
    std::vector<extent_descriptor> payload_extents{};
};

struct image_cspack_inspection {
    std::uint64_t shard_id = 0;
    std::uint64_t partition_index = 0;
    std::uint64_t envelope_offset = 0;
    std::uint64_t envelope_bytes = 0;
    std::uint64_t inspected_bytes = 0;
    std::uint64_t envelope_checksum = 0;
    image_descriptor descriptor{};
    extent_descriptor payload_extent{};
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

[[nodiscard]] status_code store_image_cspack(
    const char *path,
    std::uint64_t shard_id,
    storage_object_id object,
    const image_cspack_entry_source *entries,
    std::size_t entry_count,
    published_image_cspack *out);

// Reads the CSPACK table and the selected CPEXEC02 metadata prefix only. The
// payload extent is described without allocating or reading the payload bytes.
[[nodiscard]] status_code inspect_image_cspack_partition(
    const char *path,
    std::uint64_t expected_shard_id,
    std::uint64_t partition_index,
    storage_object_id object,
    extent_id extent,
    image_cspack_inspection *out);

} // namespace cellshard
