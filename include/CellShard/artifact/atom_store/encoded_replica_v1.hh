#pragma once

#include <CellShard/artifact/atom_store/identity_v1.hh>
#include <CellShard/identity.hh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::artifact::atom_store {

enum class replica_encoding_v1 : std::uint32_t {
    identity = 1,
    zstd = 2,
    lz4 = 3,
    provider_defined = 0xffff'ffffu,
};

struct encoded_replica_descriptor_v1 {
    replica_identity_v1 replica{};
    semantic_identity_v1 atom{};
    materialization_identity_v1 materialization{};
    content_digest_v1 decoded_content{};
    content_digest_v1 encoded_content{};
    cellshard::storage_object_id object{};
    std::uint64_t encoded_offset = 0;
    std::uint64_t encoded_bytes = 0;
    std::uint64_t decoded_bytes = 0;
    std::uint64_t first_extent_slice = 0;
    std::uint64_t extent_slice_count = 0;
    replica_encoding_v1 encoding = replica_encoding_v1::identity;
    std::uint32_t reserved = 0;
};

[[nodiscard]] constexpr bool nonzero_replica_digest_v1(
    const content_digest_v1 &digest) noexcept {
    for (auto byte : digest.bytes) if (byte != std::byte{0}) return true;
    return false;
}

[[nodiscard]] constexpr bool valid_encoded_replica_descriptor_v1(
    const encoded_replica_descriptor_v1 &descriptor) noexcept {
    if (!descriptor.replica.valid() || !descriptor.atom.valid()
        || !descriptor.materialization.valid() || !descriptor.object.valid()
        || !valid_content_digest_v1(descriptor.decoded_content)
        || !valid_content_digest_v1(descriptor.encoded_content)
        || !nonzero_replica_digest_v1(descriptor.decoded_content)
        || !nonzero_replica_digest_v1(descriptor.encoded_content)
        || descriptor.encoded_bytes == 0 || descriptor.decoded_bytes == 0
        || descriptor.encoded_offset > std::numeric_limits<std::uint64_t>::max()
            - descriptor.encoded_bytes
        || descriptor.extent_slice_count == 0
        || descriptor.first_extent_slice > std::numeric_limits<std::uint64_t>::max()
            - descriptor.extent_slice_count) return false;
    if (descriptor.encoding == replica_encoding_v1::identity) {
        return descriptor.encoded_bytes == descriptor.decoded_bytes
            && descriptor.encoded_content.bytes == descriptor.decoded_content.bytes;
    }
    return descriptor.encoding == replica_encoding_v1::zstd
        || descriptor.encoding == replica_encoding_v1::lz4
        || descriptor.encoding == replica_encoding_v1::provider_defined;
}

static_assert(std::is_trivially_copyable<encoded_replica_descriptor_v1>::value);

} // namespace cellshard::artifact::atom_store
