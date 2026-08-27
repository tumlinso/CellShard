#include <CellShard/io/pack/image_envelope.hh>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <new>

namespace cellshard {
namespace {

constexpr std::array<std::byte, 8> envelope_magic{{
    std::byte{'C'}, std::byte{'P'}, std::byte{'E'}, std::byte{'X'},
    std::byte{'E'}, std::byte{'C'}, std::byte{'0'}, std::byte{'2'},
}};
constexpr std::size_t checksum_offset = 48;
constexpr std::size_t digest_bytes_offset = 160;
constexpr std::size_t domain_record_bytes = 40;
constexpr std::uint64_t fnv1a_offset = 1469598103934665603ull;
constexpr std::uint64_t fnv1a_prime = 1099511628211ull;

void put_u32(std::byte *output, std::size_t offset, std::uint32_t value) noexcept {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        output[offset + shift / 8] = std::byte((value >> shift) & 0xffu);
    }
}

void put_u64(std::byte *output, std::size_t offset, std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output[offset + shift / 8] = std::byte((value >> shift) & 0xffu);
    }
}

[[nodiscard]] std::uint32_t get_u32(const std::byte *input,
                                    std::size_t offset) noexcept {
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) {
        value |= std::uint32_t(std::to_integer<unsigned char>(
                     input[offset + shift / 8]))
            << shift;
    }
    return value;
}

[[nodiscard]] std::uint64_t get_u64(const std::byte *input,
                                    std::size_t offset) noexcept {
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        value |= std::uint64_t(std::to_integer<unsigned char>(
                     input[offset + shift / 8]))
            << shift;
    }
    return value;
}

[[nodiscard]] bool add_size(std::size_t *value, std::size_t increment) noexcept {
    if (value == nullptr || increment > std::numeric_limits<std::size_t>::max() - *value) {
        return false;
    }
    *value += increment;
    return true;
}

[[nodiscard]] bool multiply_size(std::size_t count, std::size_t width,
                                 std::size_t *out) noexcept {
    if (out == nullptr || (count != 0
        && width > std::numeric_limits<std::size_t>::max() / count)) {
        return false;
    }
    *out = count * width;
    return true;
}

[[nodiscard]] bool align_size(std::size_t value, std::uint32_t alignment,
                              std::size_t *out) noexcept {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0
        || alignment > image_envelope_max_alignment
        || out == nullptr) {
        return false;
    }
    const std::size_t mask = alignment - 1u;
    if (value > std::numeric_limits<std::size_t>::max() - mask) {
        return false;
    }
    *out = (value + mask) & ~mask;
    return true;
}

[[nodiscard]] std::uint64_t checksum(const std::byte *bytes,
                                     std::size_t size) noexcept {
    std::uint64_t hash = fnv1a_offset;
    for (std::size_t index = 0; index < size; ++index) {
        const unsigned char value = index >= checksum_offset
                && index < checksum_offset + sizeof(std::uint64_t)
            ? 0
            : std::to_integer<unsigned char>(bytes[index]);
        hash ^= value;
        hash *= fnv1a_prime;
    }
    return hash == 0 ? 1 : hash;
}

[[nodiscard]] status_code layout_size(const image_descriptor_view &descriptor,
                                      std::size_t payload_bytes,
                                      std::size_t *metadata_end,
                                      std::size_t *payload_offset,
                                      std::size_t *total_bytes) noexcept {
    if (metadata_end == nullptr || payload_offset == nullptr || total_bytes == nullptr
        || payload_bytes == 0 || descriptor.stored_bytes != payload_bytes
        || descriptor.domains.size > std::numeric_limits<std::uint32_t>::max()
        || descriptor.dependencies.size > std::numeric_limits<std::uint32_t>::max()
        || descriptor.routes.size > std::numeric_limits<std::uint32_t>::max()) {
        return status_code::invalid_input;
    }
    std::size_t size = image_envelope_fixed_header_bytes;
    std::size_t table_bytes = 0;
    if (!multiply_size(descriptor.domains.size, domain_record_bytes, &table_bytes)
        || !add_size(&size, table_bytes)
        || !multiply_size(descriptor.dependencies.size, sizeof(std::uint64_t),
                          &table_bytes)
        || !add_size(&size, table_bytes)
        || !multiply_size(descriptor.routes.size, sizeof(std::uint64_t), &table_bytes)
        || !add_size(&size, table_bytes)) {
        return status_code::invalid_input;
    }
    *metadata_end = size;
    if (!align_size(size, descriptor.required_alignment, payload_offset)) {
        return status_code::invalid_input;
    }
    size = *payload_offset;
    if (!add_size(&size, payload_bytes)) {
        return status_code::invalid_input;
    }
    *total_bytes = size;
    return status_code::success;
}

} // namespace

status_code encoded_image_envelope_size(const image_descriptor_view &descriptor,
                                        std::size_t payload_bytes,
                                        std::size_t *out_bytes) noexcept {
    if (out_bytes == nullptr || !valid_image_descriptor(descriptor)) {
        return status_code::invalid_input;
    }
    std::size_t metadata_end = 0;
    std::size_t payload_offset = 0;
    return layout_size(descriptor, payload_bytes, &metadata_end, &payload_offset,
                       out_bytes);
}

status_code encode_image_envelope(const image_descriptor_view &descriptor,
                                  array_view<std::byte> payload,
                                  std::byte *output,
                                  std::size_t output_bytes) noexcept {
    std::size_t metadata_end = 0;
    std::size_t payload_offset = 0;
    std::size_t required_bytes = 0;
    if (!valid_image_descriptor(descriptor) || payload.data == nullptr
        || output == nullptr
        || layout_size(descriptor, payload.size, &metadata_end, &payload_offset,
                       &required_bytes) != status_code::success
        || output_bytes != required_bytes) {
        return status_code::invalid_input;
    }

    std::memset(output, 0, output_bytes);
    std::copy(envelope_magic.begin(), envelope_magic.end(), output);
    put_u32(output, 8, image_envelope_schema_version);
    put_u32(output, 12, image_envelope_fixed_header_bytes);
    put_u32(output, 16, image_envelope_endian_marker);
    put_u64(output, 24, output_bytes);
    put_u64(output, 32, payload_offset);
    put_u64(output, 40, payload.size);
    put_u64(output, 56, descriptor.id.value());
    put_u64(output, 64, descriptor.projection.producer.value());
    put_u64(output, 72, descriptor.projection.structure.value());
    put_u64(output, 80, descriptor.projection.geometry.value());
    put_u64(output, 88, descriptor.projection.operation.value());
    put_u64(output, 96, descriptor.projection.encoding.value());
    put_u32(output, 104,
            static_cast<std::uint32_t>(descriptor.projection.target.backend));
    put_u32(output, 108, descriptor.projection.target.capability_major);
    put_u32(output, 112, descriptor.projection.target.capability_minor);
    put_u32(output, 116, descriptor.required_alignment);
    put_u64(output, 120, descriptor.projection.target.capability_flags);
    put_u64(output, 128, descriptor.device_bytes);
    put_u32(output, 136, static_cast<std::uint32_t>(descriptor.reuse));
    put_u32(output, 140, static_cast<std::uint32_t>(descriptor.domains.size));
    put_u32(output, 144, static_cast<std::uint32_t>(descriptor.dependencies.size));
    put_u32(output, 148, static_cast<std::uint32_t>(descriptor.routes.size));
    put_u32(output, 152,
            static_cast<std::uint32_t>(descriptor.payload_digest.algorithm));
    put_u32(output, 156, descriptor.payload_digest.used_bytes);
    std::memcpy(output + digest_bytes_offset, descriptor.payload_digest.bytes.data(),
                descriptor.payload_digest.bytes.size());

    std::size_t cursor = image_envelope_fixed_header_bytes;
    for (std::size_t index = 0; index < descriptor.domains.size; ++index) {
        const auto &binding = descriptor.domains[index];
        put_u32(output, cursor, static_cast<std::uint32_t>(binding.role));
        put_u64(output, cursor + 8, binding.domain.value());
        put_u64(output, cursor + 16, binding.map.value());
        put_u64(output, cursor + 24, binding.partition.value());
        put_u64(output, cursor + 32, binding.order.value());
        cursor += domain_record_bytes;
    }
    for (std::size_t index = 0; index < descriptor.dependencies.size; ++index) {
        put_u64(output, cursor, descriptor.dependencies[index].value());
        cursor += sizeof(std::uint64_t);
    }
    for (std::size_t index = 0; index < descriptor.routes.size; ++index) {
        put_u64(output, cursor, descriptor.routes[index].value());
        cursor += sizeof(std::uint64_t);
    }
    if (cursor != metadata_end) {
        return status_code::invalid_input;
    }
    std::memcpy(output + payload_offset, payload.data, payload.size);
    put_u64(output, checksum_offset, checksum(output, output_bytes));
    return status_code::success;
}

status_code decode_image_envelope(const std::byte *input, std::size_t input_bytes,
                                  decoded_image_envelope *out) {
    if (out == nullptr) {
        return status_code::invalid_input;
    }
    *out = decoded_image_envelope{};
    if (input == nullptr || input_bytes < image_envelope_fixed_header_bytes
        || !std::equal(envelope_magic.begin(), envelope_magic.end(), input)
        || get_u32(input, 8) != image_envelope_schema_version
        || get_u32(input, 12) != image_envelope_fixed_header_bytes
        || get_u32(input, 16) != image_envelope_endian_marker
        || get_u32(input, 20) != 0 || get_u64(input, 24) != input_bytes
        || get_u64(input, checksum_offset) == 0
        || get_u64(input, checksum_offset) != checksum(input, input_bytes)) {
        return status_code::corruption;
    }

    const std::uint64_t payload_offset_u64 = get_u64(input, 32);
    const std::uint64_t payload_bytes_u64 = get_u64(input, 40);
    if (payload_offset_u64 > input_bytes || payload_bytes_u64 > input_bytes
        || payload_bytes_u64 != input_bytes - payload_offset_u64
        || payload_bytes_u64 > std::numeric_limits<std::size_t>::max()) {
        return status_code::corruption;
    }

    image_descriptor descriptor{};
    descriptor.id = image_id{get_u64(input, 56)};
    descriptor.projection.producer = producer_abi_id{get_u64(input, 64)};
    descriptor.projection.structure = structure_id{get_u64(input, 72)};
    descriptor.projection.geometry = geometry_id{get_u64(input, 80)};
    descriptor.projection.operation = operator_class_id{get_u64(input, 88)};
    descriptor.projection.encoding = scalar_encoding_id{get_u64(input, 96)};
    descriptor.projection.target.backend =
        static_cast<execution_backend>(get_u32(input, 104));
    descriptor.projection.target.capability_major = get_u32(input, 108);
    descriptor.projection.target.capability_minor = get_u32(input, 112);
    descriptor.required_alignment = get_u32(input, 116);
    descriptor.projection.target.capability_flags = get_u64(input, 120);
    descriptor.stored_bytes = payload_bytes_u64;
    descriptor.device_bytes = get_u64(input, 128);
    descriptor.reuse = static_cast<image_reuse_class>(get_u32(input, 136));
    const std::size_t domain_count = get_u32(input, 140);
    const std::size_t dependency_count = get_u32(input, 144);
    const std::size_t route_count = get_u32(input, 148);
    descriptor.payload_digest.algorithm =
        static_cast<digest_algorithm>(get_u32(input, 152));
    descriptor.payload_digest.used_bytes = get_u32(input, 156);
    std::memcpy(descriptor.payload_digest.bytes.data(), input + digest_bytes_offset,
                descriptor.payload_digest.bytes.size());

    std::size_t metadata_end = image_envelope_fixed_header_bytes;
    std::size_t table_bytes = 0;
    if (!multiply_size(domain_count, domain_record_bytes, &table_bytes)
        || !add_size(&metadata_end, table_bytes)
        || !multiply_size(dependency_count, sizeof(std::uint64_t), &table_bytes)
        || !add_size(&metadata_end, table_bytes)
        || !multiply_size(route_count, sizeof(std::uint64_t), &table_bytes)
        || !add_size(&metadata_end, table_bytes)) {
        return status_code::corruption;
    }
    std::size_t expected_payload_offset = 0;
    if (!align_size(metadata_end, descriptor.required_alignment,
                    &expected_payload_offset)
        || expected_payload_offset != payload_offset_u64) {
        return status_code::corruption;
    }
    for (std::size_t index = metadata_end; index < expected_payload_offset; ++index) {
        if (input[index] != std::byte{0}) {
            return status_code::corruption;
        }
    }

    try {
        descriptor.domains.reserve(domain_count);
        descriptor.dependencies.reserve(dependency_count);
        descriptor.routes.reserve(route_count);
    } catch (const std::bad_alloc &) {
        return status_code::allocation_failure;
    }
    std::size_t cursor = image_envelope_fixed_header_bytes;
    for (std::size_t index = 0; index < domain_count; ++index) {
        if (get_u32(input, cursor + 4) != 0) {
            return status_code::corruption;
        }
        descriptor.domains.push_back({
            static_cast<domain_binding_role>(get_u32(input, cursor)),
            domain_id{get_u64(input, cursor + 8)},
            partition_map_id{get_u64(input, cursor + 16)},
            partition_id{get_u64(input, cursor + 24)},
            order_id{get_u64(input, cursor + 32)},
        });
        cursor += domain_record_bytes;
    }
    for (std::size_t index = 0; index < dependency_count; ++index) {
        descriptor.dependencies.push_back(image_id{get_u64(input, cursor)});
        cursor += sizeof(std::uint64_t);
    }
    for (std::size_t index = 0; index < route_count; ++index) {
        descriptor.routes.push_back(route_table_id{get_u64(input, cursor)});
        cursor += sizeof(std::uint64_t);
    }
    if (cursor != metadata_end || !valid_image_descriptor(descriptor)) {
        return status_code::corruption;
    }

    out->descriptor = std::move(descriptor);
    out->payload = {input + expected_payload_offset,
                    static_cast<std::size_t>(payload_bytes_u64)};
    out->payload_offset = payload_offset_u64;
    out->envelope_checksum = get_u64(input, checksum_offset);
    return status_code::success;
}

} // namespace cellshard
