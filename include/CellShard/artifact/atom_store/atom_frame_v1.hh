#pragma once

#include <CellShard/artifact/atom_store/format_v1.hh>
#include <CellShard/artifact/atom_store/identity_v1.hh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::artifact::atom_store {

inline constexpr std::array<std::byte, 8> atom_frame_magic_v1{
    std::byte{'C'}, std::byte{'S'}, std::byte{'A'}, std::byte{'T'},
    std::byte{'M'}, std::byte{'F'}, std::byte{'R'}, std::byte{'1'}};
inline constexpr std::uint32_t atom_frame_header_bytes_v1 = 128;

enum class atom_frame_codec_v1 : std::uint32_t {
    raw = 0,
    provider_defined = 0xffff'ffffu,
};

enum atom_frame_flags_v1 : std::uint32_t {
    atom_frame_exact_bytes_v1 = 1u << 0,
};

struct atom_frame_header_v1 {
    std::array<std::byte, 8> magic = atom_frame_magic_v1;
    std::uint32_t schema_version = schema_version_v1;
    std::uint32_t header_bytes = atom_frame_header_bytes_v1;
    semantic_identity_v1 atom{};
    materialization_identity_v1 materialization{};
    content_digest_v1 content{};
    std::uint64_t logical_bytes = 0;
    std::uint64_t encoded_bytes = 0;
    std::uint64_t payload_offset = 0;
    atom_frame_codec_v1 codec = atom_frame_codec_v1::raw;
    std::uint32_t flags = atom_frame_exact_bytes_v1;
    std::uint32_t payload_alignment = 0;
    std::uint32_t reserved = 0;
};

[[nodiscard]] constexpr bool valid_atom_frame_header_v1(
    const atom_frame_header_v1 &header, std::uint64_t available_bytes) noexcept {
    for (std::size_t index = 0; index < atom_frame_magic_v1.size(); ++index) {
        if (header.magic[index] != atom_frame_magic_v1[index]) return false;
    }
    if (header.schema_version != schema_version_v1
        || header.header_bytes != atom_frame_header_bytes_v1
        || !header.atom.valid() || !header.materialization.valid()
        || !valid_content_digest_v1(header.content)
        || header.logical_bytes == 0 || header.encoded_bytes == 0
        || header.flags != atom_frame_exact_bytes_v1
        || header.payload_alignment == 0
        || (header.payload_alignment & (header.payload_alignment - 1u)) != 0
        || header.payload_offset < header.header_bytes
        || (header.payload_offset % header.payload_alignment) != 0
        || header.payload_offset > available_bytes
        || header.encoded_bytes > available_bytes - header.payload_offset) return false;
    bool digest_nonzero = false;
    for (auto byte : header.content.bytes) digest_nonzero = digest_nonzero || byte != std::byte{0};
    return digest_nonzero
        && (header.codec != atom_frame_codec_v1::raw
            || header.logical_bytes == header.encoded_bytes);
}

static_assert(sizeof(atom_frame_header_v1) == atom_frame_header_bytes_v1);
static_assert(std::is_trivially_copyable<atom_frame_header_v1>::value);

} // namespace cellshard::artifact::atom_store
