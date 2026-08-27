#include <CellShard/io/pack/image_envelope.hh>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

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

std::string temporary_path(const char *suffix) {
    return std::string("/tmp/cellshard-image-envelope-")
        + std::to_string(static_cast<unsigned long long>(::getpid())) + suffix;
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

    const std::string pack_path = temporary_path(".cspack");
    std::remove(pack_path.c_str());
    std::remove((pack_path + ".tmp").c_str());
    std::vector<std::byte> large_payload(8192, std::byte{0x5a});
    auto second_image = descriptor(large_payload.size());
    second_image.id = image_id{2};
    second_image.dependencies.clear();
    const std::array<image_cspack_entry_source, 2> sources{{
        {extent_id{101}, view_of(image), {payload.data(), payload.size()}},
        {extent_id{102}, view_of(second_image),
         {large_payload.data(), large_payload.size()}},
    }};
    published_image_cspack published{};
    require(store_image_cspack(pack_path.c_str(), 77, storage_object_id{88},
                               sources.data(), sources.size(), &published)
                == status_code::success,
            "publish image cspack");
    require(published.object.id == storage_object_id{88}
            && published.object.byte_count > large_payload.size()
            && valid_storage_object_descriptor(published.object)
            && published.payload_extents.size() == sources.size(),
            "published object descriptor");
    require(published.payload_extents[0].id == extent_id{101}
            && published.payload_extents[1].id == extent_id{102}
            && valid_extent_descriptor(published.payload_extents[0],
                                       published.object)
            && valid_extent_descriptor(published.payload_extents[1],
                                       published.object),
            "published payload extents");
    require(published.payload_extents[0].object_offset
                    % published.payload_extents[0].required_alignment == 0
            && published.payload_extents[1].object_offset
                    % published.payload_extents[1].required_alignment == 0,
            "absolute payload alignment");
    require(::access((pack_path + ".tmp").c_str(), F_OK) != 0,
            "temporary file removed after publication");

    image_cspack_inspection inspection{};
    require(inspect_image_cspack_partition(
                pack_path.c_str(), 77, 1, storage_object_id{88}, extent_id{102},
                &inspection) == status_code::success,
            "inspect selected CPEXEC02 metadata");
    require(inspection.shard_id == 77 && inspection.partition_index == 1
            && inspection.descriptor.id == second_image.id
            && inspection.descriptor.projection.producer
                == second_image.projection.producer
            && inspection.descriptor.projection.structure
                == second_image.projection.structure
            && inspection.descriptor.projection.geometry
                == second_image.projection.geometry
            && inspection.descriptor.projection.operation
                == second_image.projection.operation
            && inspection.descriptor.projection.encoding
                == second_image.projection.encoding
            && inspection.descriptor.projection.target.backend
                == second_image.projection.target.backend
            && inspection.descriptor.projection.target.capability_major
                == second_image.projection.target.capability_major
            && inspection.payload_extent.id == extent_id{102}
            && inspection.payload_extent.object == storage_object_id{88}
            && inspection.payload_extent.object_offset
                == published.payload_extents[1].object_offset
            && inspection.payload_extent.byte_count == large_payload.size()
            && inspection.envelope_checksum != 0,
            "inspection preserves image identity and exposes extent");
    require(inspection.inspected_bytes < published.object.byte_count
            && inspection.inspected_bytes
                < published.object.byte_count - large_payload.size() / 2,
            "inspection does not load selected payload");

    std::FILE *pack = std::fopen(pack_path.c_str(), "rb");
    require(pack != nullptr, "open published pack");
    std::array<unsigned char, 8> top_magic{};
    std::uint64_t top_shard = 0, top_count = 0;
    require(std::fread(top_magic.data(), 1, top_magic.size(), pack)
                    == top_magic.size()
            && std::memcmp(top_magic.data(), "CSPACK01", top_magic.size()) == 0
            && std::fread(&top_shard, sizeof(top_shard), 1, pack) == 1
            && std::fread(&top_count, sizeof(top_count), 1, pack) == 1
            && top_shard == 77 && top_count == 2,
            "unchanged CSPACK01 top-level header");
    std::fclose(pack);

    pack = std::fopen(pack_path.c_str(), "rb+");
    require(pack != nullptr
            && std::fseek(pack, 8 + 2 * sizeof(std::uint64_t), SEEK_SET) == 0,
            "open offset table for corruption case");
    const std::uint64_t invalid_offset = 1;
    require(std::fwrite(&invalid_offset, sizeof(invalid_offset), 1, pack) == 1
            && std::fclose(pack) == 0,
            "corrupt offset table");
    require(inspect_image_cspack_partition(
                pack_path.c_str(), 77, 0, storage_object_id{88}, extent_id{101},
                &inspection) == status_code::corruption,
            "invalid top-level offset rejected");

    require(store_image_cspack(pack_path.c_str(), 77, storage_object_id{88},
                               sources.data(), sources.size(), &published)
                == status_code::success,
            "republish after corruption");
    require(::truncate(pack_path.c_str(),
                       static_cast<off_t>(published.object.byte_count - 1)) == 0,
            "truncate published pack");
    require(inspect_image_cspack_partition(
                pack_path.c_str(), 77, 1, storage_object_id{88}, extent_id{102},
                &inspection) == status_code::corruption,
            "truncated selected envelope rejected");
    std::remove(pack_path.c_str());

    const std::string blocked_path = temporary_path("-directory");
    std::remove((blocked_path + ".tmp").c_str());
    ::rmdir(blocked_path.c_str());
    require(::mkdir(blocked_path.c_str(), 0700) == 0,
            "create rename blocker directory");
    require(store_image_cspack(blocked_path.c_str(), 77, storage_object_id{88},
                               sources.data(), sources.size(), &published)
                == status_code::corruption,
            "rename failure rejects publication");
    require(::access((blocked_path + ".tmp").c_str(), F_OK) != 0,
            "temporary file cleaned after publication failure");
    require(::rmdir(blocked_path.c_str()) == 0,
            "remove rename blocker directory");

    std::puts("cellShardImageEnvelopeTest: passed");
    return 0;
}
