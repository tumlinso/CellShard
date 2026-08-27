#include <CellShard/io/pack/image_envelope.hh>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "cellShardImageEnvelopeTest: %s\n", message);
        std::exit(1);
    }
}

cellshard::content_digest digest() {
    cellshard::content_digest result{};
    result.algorithm = cellshard::digest_algorithm::legacy_fnv1a64;
    result.used_bytes = sizeof(std::uint64_t);
    result.bytes[0] = std::byte{0xd2};
    return result;
}

cellshard::image_descriptor descriptor(std::size_t payload_bytes) {
    using namespace cellshard;
    image_descriptor result{};
    result.id = image_id{1};
    result.projection = {producer_abi_id{2}, structure_id{3}, geometry_id{4},
                         operator_class_id{5}, scalar_encoding_id{6},
                         {execution_backend::cuda, 7, 0, 0x40}};
    result.stored_bytes = payload_bytes;
    result.device_bytes = payload_bytes + 64;
    result.required_alignment = 256;
    result.reuse = image_reuse_class::durable_reuse;
    result.payload_digest = digest();
    result.domains = {
        {domain_binding_role::primary, domain_id{10}, partition_map_id{11},
         partition_id{12}, order_id{13}},
        {domain_binding_role::secondary, domain_id{20}, partition_map_id{21},
         partition_id{22}, order_id{23}},
    };
    result.dependencies = {image_id{30}, image_id{31}};
    result.routes = {route_table_id{40}};
    return result;
}

std::uint32_t little_u32(const std::vector<std::byte> &bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) {
        value |= std::uint32_t(std::to_integer<unsigned char>(
                     bytes[offset + shift / 8]))
            << shift;
    }
    return value;
}

std::uint64_t little_u64(const std::vector<std::byte> &bytes, std::size_t offset) {
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        value |= std::uint64_t(std::to_integer<unsigned char>(
                     bytes[offset + shift / 8]))
            << shift;
    }
    return value;
}

void set_u32(std::vector<std::byte> &bytes, std::size_t offset,
             std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        bytes[offset + shift / 8] = std::byte((value >> shift) & 0xffu);
    }
}

} // namespace

int main() {
    using namespace cellshard;
    const std::array<std::byte, 9> payload{{
        std::byte{9}, std::byte{8}, std::byte{7}, std::byte{6}, std::byte{5},
        std::byte{4}, std::byte{3}, std::byte{2}, std::byte{1},
    }};
    const auto image = descriptor(payload.size());
    std::size_t encoded_bytes = 0;
    require(encoded_image_envelope_size(view_of(image), payload.size(),
                                        &encoded_bytes) == status_code::success,
            "calculate encoded size");
    require(encoded_bytes > image_envelope_fixed_header_bytes + payload.size(),
            "aligned envelope includes metadata and padding");

    std::vector<std::byte> first(encoded_bytes);
    std::vector<std::byte> second(encoded_bytes);
    require(encode_image_envelope(view_of(image), {payload.data(), payload.size()},
                                  first.data(), first.size()) == status_code::success,
            "encode first envelope");
    require(encode_image_envelope(view_of(image), {payload.data(), payload.size()},
                                  second.data(), second.size()) == status_code::success,
            "encode second envelope");
    require(first == second, "deterministic encoding");
    require(little_u32(first, 8) == image_envelope_schema_version,
            "explicit schema version");
    require(little_u32(first, 16) == image_envelope_endian_marker,
            "explicit little-endian marker");
    const auto payload_offset = static_cast<std::size_t>(little_u64(first, 32));
    require(payload_offset % image.required_alignment == 0,
            "payload offset alignment");
    for (std::size_t index = image_envelope_fixed_header_bytes
             + image.domains.size() * 40
             + image.dependencies.size() * 8 + image.routes.size() * 8;
         index < payload_offset; ++index) {
        require(first[index] == std::byte{0}, "deterministic zero padding");
    }

    decoded_image_envelope decoded{};
    require(decode_image_envelope(first.data(), first.size(), &decoded)
                == status_code::success,
            "decode envelope");
    require(decoded.descriptor.id == image.id
            && decoded.descriptor.projection.producer == image.projection.producer
            && decoded.descriptor.projection.target.backend
                == image.projection.target.backend
            && decoded.descriptor.domains.size() == image.domains.size()
            && decoded.descriptor.dependencies == image.dependencies
            && decoded.descriptor.routes == image.routes,
            "exact descriptor round trip");
    require(decoded.payload.data == first.data() + payload_offset
            && decoded.payload.size == payload.size()
            && std::equal(payload.begin(), payload.end(), decoded.payload.begin()),
            "opaque payload byte preservation");

    auto malformed = first;
    malformed.back() ^= std::byte{0x80};
    require(decode_image_envelope(malformed.data(), malformed.size(), &decoded)
                == status_code::corruption,
            "checksum tamper rejection");
    malformed = first;
    set_u32(malformed, 16, 0x04030201u);
    require(decode_image_envelope(malformed.data(), malformed.size(), &decoded)
                == status_code::corruption,
            "endian marker rejection");
    malformed = first;
    set_u32(malformed, 140, 0xffffffffu);
    require(decode_image_envelope(malformed.data(), malformed.size(), &decoded)
                == status_code::corruption,
            "oversized domain table rejection");
    malformed = first;
    const std::size_t padding_index = image_envelope_fixed_header_bytes
        + image.domains.size() * 40 + image.dependencies.size() * 8
        + image.routes.size() * 8;
    require(padding_index < payload_offset, "test envelope has padding");
    malformed[padding_index] = std::byte{1};
    require(decode_image_envelope(malformed.data(), malformed.size(), &decoded)
                == status_code::corruption,
            "nonzero padding rejection");
    malformed.assign(first.begin(), first.end() - 1);
    require(decode_image_envelope(malformed.data(), malformed.size(), &decoded)
                == status_code::corruption,
            "truncated payload rejection");

    auto huge_alignment = image;
    huge_alignment.required_alignment = image_envelope_max_alignment << 1;
    require(encoded_image_envelope_size(view_of(huge_alignment), payload.size(),
                                        &encoded_bytes) == status_code::invalid_input,
            "excessive alignment rejected before allocation");
    require(encode_image_envelope(view_of(image), {payload.data(), payload.size()},
                                  first.data(), first.size() - 1)
                == status_code::invalid_input,
            "wrong output size rejection");

    std::puts("cellShardImageEnvelopeTest: passed");
    return 0;
}
