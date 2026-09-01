#pragma once

#include <CellShard/compiler/partial/dependency_freshness_v1.hh>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace cellshard::compiler::partial {

inline constexpr std::uint32_t partial_image_schema_version_v1 = 1;
inline constexpr std::uint32_t partial_image_header_bytes_v1 = 272;
inline constexpr std::uint32_t partial_image_dependency_bytes_v1 = 48;
inline constexpr std::uint64_t partial_image_checksum_offset_v1 = 56;
inline constexpr char partial_image_magic_v1[8] = {
    'C', 'S', 'P', 'A', 'R', 'T', '0', '1'};

enum class partial_image_code_v1 : std::uint32_t {
    valid = 0,
    invalid_partial,
    invalid_dependency_closure,
    closure_binding_mismatch,
    size_overflow,
    capacity_overflow,
    invalid_magic,
    unsupported_schema,
    invalid_header_bytes,
    truncated_image,
    invalid_offsets,
    checksum_mismatch,
    invalid_header,
    invalid_dependency,
};

struct partial_image_result_v1 {
    partial_image_code_v1 code = partial_image_code_v1::valid;
    std::uint64_t bytes = 0;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == partial_image_code_v1::valid;
    }
};

struct materialized_partial_image_v1 {
    const std::byte *image = nullptr;
    std::uint64_t image_bytes = 0;
    const std::byte *payload = nullptr;
    std::uint64_t payload_bytes = 0;
    partial_atom_header_v1 header{};
    atom_persistent_identity_v1 exact_dependency_certification_identity{};
    std::uint64_t dependency_offset = 0;
    std::uint64_t dependency_count = 0;
};

inline void partial_image_write_u32_v1(std::byte *destination,
                                       std::uint32_t value) noexcept {
    for (std::uint32_t index = 0; index < 4; ++index) {
        destination[index] = std::byte((value >> (index * 8)) & 0xffU);
    }
}

inline void partial_image_write_u64_v1(std::byte *destination,
                                       std::uint64_t value) noexcept {
    for (std::uint32_t index = 0; index < 8; ++index) {
        destination[index] = std::byte((value >> (index * 8)) & 0xffU);
    }
}

[[nodiscard]] inline std::uint32_t partial_image_read_u32_v1(
    const std::byte *source) noexcept {
    std::uint32_t value = 0;
    for (std::uint32_t index = 0; index < 4; ++index) {
        value |= std::uint32_t(std::to_integer<std::uint8_t>(source[index]))
            << (index * 8);
    }
    return value;
}

[[nodiscard]] inline std::uint64_t partial_image_read_u64_v1(
    const std::byte *source) noexcept {
    std::uint64_t value = 0;
    for (std::uint32_t index = 0; index < 8; ++index) {
        value |= std::uint64_t(std::to_integer<std::uint8_t>(source[index]))
            << (index * 8);
    }
    return value;
}

inline void partial_image_write_id_v1(
    std::byte *destination, atom_persistent_identity_v1 identity) noexcept {
    partial_image_write_u64_v1(destination, identity.producer_namespace);
    partial_image_write_u64_v1(destination + 8, identity.local_identity);
}

[[nodiscard]] inline atom_persistent_identity_v1 partial_image_read_id_v1(
    const std::byte *source) noexcept {
    return {partial_image_read_u64_v1(source),
            partial_image_read_u64_v1(source + 8)};
}

[[nodiscard]] inline std::uint64_t partial_image_checksum_v1(
    const std::byte *image, std::uint64_t image_bytes) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::uint64_t index = 0; index < image_bytes; ++index) {
        if (index >= partial_image_checksum_offset_v1
            && index < partial_image_checksum_offset_v1 + 8) {
            continue;
        }
        hash ^= std::to_integer<std::uint8_t>(image[index]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

[[nodiscard]] constexpr std::uint64_t partial_image_align8_v1(
    std::uint64_t value) noexcept {
    return (value + 7U) & ~std::uint64_t{7};
}

[[nodiscard]] inline partial_image_result_v1 serialize_partial_image_v1(
    const partial_atom_view_v1 &partial,
    const partial_dependency_closure_view_v1 &closure,
    std::byte *destination, std::uint64_t capacity) noexcept {
    if (!validate_partial_atom_envelope_v1(partial).valid()) {
        return {partial_image_code_v1::invalid_partial, 0, 0};
    }
    const auto closure_result = validate_partial_dependency_closure_v1(
        closure, partial.header.partial_identity);
    if (!closure_result.valid()) {
        return {partial_image_code_v1::invalid_dependency_closure, 0,
                closure_result.index};
    }
    if (closure.closure_identity != partial.header.dependency_closure_identity) {
        return {partial_image_code_v1::closure_binding_mismatch, 0, 0};
    }
    if (closure.dependency_count
        > (std::numeric_limits<std::uint64_t>::max()
               - partial_image_header_bytes_v1)
              / partial_image_dependency_bytes_v1) {
        return {partial_image_code_v1::size_overflow, 0, 0};
    }
    const std::uint64_t dependency_offset = partial_image_header_bytes_v1;
    const std::uint64_t payload_offset = partial_image_align8_v1(
        dependency_offset
        + closure.dependency_count * partial_image_dependency_bytes_v1);
    if (partial.payload_bytes
        > std::numeric_limits<std::uint64_t>::max() - payload_offset) {
        return {partial_image_code_v1::size_overflow, 0, 0};
    }
    const std::uint64_t image_bytes = payload_offset + partial.payload_bytes;
    if (destination == nullptr || capacity < image_bytes) {
        return {partial_image_code_v1::capacity_overflow, image_bytes, 0};
    }
    std::memset(destination, 0, static_cast<std::size_t>(image_bytes));
    for (std::uint32_t index = 0; index < 8; ++index) {
        destination[index] = std::byte(partial_image_magic_v1[index]);
    }
    partial_image_write_u32_v1(destination + 8, partial_image_schema_version_v1);
    partial_image_write_u32_v1(destination + 12, partial_image_header_bytes_v1);
    partial_image_write_u64_v1(destination + 16, image_bytes);
    partial_image_write_u64_v1(destination + 24, closure.dependency_count);
    partial_image_write_u64_v1(destination + 32, dependency_offset);
    partial_image_write_u64_v1(destination + 40, payload_offset);
    partial_image_write_u64_v1(destination + 48, partial.payload_bytes);
    const atom_persistent_identity_v1 identities[]{
        partial.header.partial_identity,
        partial.header.source_atom_semantic_identity,
        partial.header.partial_kind_identity,
        partial.header.payload_schema_identity,
        partial.header.contribution_coverage_identity,
        partial.header.dependency_closure_identity,
        partial.header.reconstruction_algebra_identity,
        partial.header.numerical_policy_identity,
        partial.header.complete_cost_evidence_identity};
    for (std::uint32_t index = 0; index < 9; ++index) {
        partial_image_write_id_v1(destination + 64 + index * 16,
                                  identities[index]);
    }
    const std::uint64_t generations[]{
        partial.header.structure_generation, partial.header.value_generation,
        partial.header.state_generation,
        partial.header.materialization_generation,
        partial.header.cost_model_generation};
    for (std::uint32_t index = 0; index < 5; ++index) {
        partial_image_write_u64_v1(destination + 208 + index * 8,
                                   generations[index]);
    }
    partial_image_write_u32_v1(
        destination + 248,
        static_cast<std::uint32_t>(partial.header.persistence_class));
    partial_image_write_u32_v1(destination + 252, partial.header.reserved);
    partial_image_write_id_v1(destination + 256,
                              closure.exact_certification_identity);
    for (std::uint64_t index = 0; index < closure.dependency_count; ++index) {
        std::byte *record = destination + dependency_offset
            + index * partial_image_dependency_bytes_v1;
        const auto &dependency = closure.dependencies[index];
        partial_image_write_id_v1(record, dependency.dependency_identity);
        partial_image_write_id_v1(record + 16,
                                  dependency.generation_namespace);
        partial_image_write_u64_v1(record + 32,
                                   dependency.captured_generation);
        partial_image_write_u32_v1(
            record + 40,
            static_cast<std::uint32_t>(dependency.generation_kind));
        partial_image_write_u32_v1(
            record + 44, static_cast<std::uint32_t>(dependency.role));
    }
    std::memcpy(destination + payload_offset, partial.payload,
                static_cast<std::size_t>(partial.payload_bytes));
    partial_image_write_u64_v1(destination + partial_image_checksum_offset_v1,
                               partial_image_checksum_v1(destination, image_bytes));
    return {partial_image_code_v1::valid, image_bytes,
            closure.dependency_count};
}

[[nodiscard]] inline partial_dependency_requirement_v1
partial_image_dependency_v1(const materialized_partial_image_v1 &image,
                            std::uint64_t index) noexcept {
    if (image.image == nullptr || index >= image.dependency_count) return {};
    const std::byte *record = image.image + image.dependency_offset
        + index * partial_image_dependency_bytes_v1;
    return {partial_image_read_id_v1(record),
            partial_image_read_id_v1(record + 16),
            partial_image_read_u64_v1(record + 32),
            static_cast<atom_dependency_generation_kind_v1>(
                partial_image_read_u32_v1(record + 40)),
            static_cast<partial_dependency_role_v1>(
                partial_image_read_u32_v1(record + 44))};
}

[[nodiscard]] inline partial_image_result_v1 materialize_partial_image_v1(
    const std::byte *source, std::uint64_t source_bytes,
    materialized_partial_image_v1 *output) noexcept {
    if (source == nullptr || output == nullptr
        || source_bytes < partial_image_header_bytes_v1) {
        return {partial_image_code_v1::truncated_image, 0, 0};
    }
    *output = {};
    for (std::uint32_t index = 0; index < 8; ++index) {
        if (std::to_integer<char>(source[index]) != partial_image_magic_v1[index]) {
            return {partial_image_code_v1::invalid_magic, 0, index};
        }
    }
    if (partial_image_read_u32_v1(source + 8)
        != partial_image_schema_version_v1) {
        return {partial_image_code_v1::unsupported_schema, 0, 0};
    }
    if (partial_image_read_u32_v1(source + 12)
        != partial_image_header_bytes_v1) {
        return {partial_image_code_v1::invalid_header_bytes, 0, 0};
    }
    const std::uint64_t image_bytes = partial_image_read_u64_v1(source + 16);
    const std::uint64_t dependency_count = partial_image_read_u64_v1(source + 24);
    const std::uint64_t dependency_offset = partial_image_read_u64_v1(source + 32);
    const std::uint64_t payload_offset = partial_image_read_u64_v1(source + 40);
    const std::uint64_t payload_bytes = partial_image_read_u64_v1(source + 48);
    if (image_bytes > source_bytes || image_bytes < partial_image_header_bytes_v1) {
        return {partial_image_code_v1::truncated_image, image_bytes, 0};
    }
    if (dependency_count
            > (std::numeric_limits<std::uint64_t>::max()
                   - partial_image_header_bytes_v1)
                  / partial_image_dependency_bytes_v1
        || dependency_offset != partial_image_header_bytes_v1
        || payload_offset != partial_image_align8_v1(
               dependency_offset
               + dependency_count * partial_image_dependency_bytes_v1)
        || payload_offset > image_bytes
        || payload_bytes > image_bytes - payload_offset
        || payload_offset + payload_bytes != image_bytes) {
        return {partial_image_code_v1::invalid_offsets, 0, 0};
    }
    if (partial_image_read_u64_v1(source + partial_image_checksum_offset_v1)
        != partial_image_checksum_v1(source, image_bytes)) {
        return {partial_image_code_v1::checksum_mismatch, 0, 0};
    }
    partial_atom_header_v1 header{};
    header.schema_version = partial_atom_schema_version_v1;
    header.record_bytes = sizeof(partial_atom_header_v1);
    atom_persistent_identity_v1 *identities[]{
        &header.partial_identity, &header.source_atom_semantic_identity,
        &header.partial_kind_identity, &header.payload_schema_identity,
        &header.contribution_coverage_identity,
        &header.dependency_closure_identity,
        &header.reconstruction_algebra_identity,
        &header.numerical_policy_identity,
        &header.complete_cost_evidence_identity};
    for (std::uint32_t index = 0; index < 9; ++index) {
        *identities[index] = partial_image_read_id_v1(source + 64 + index * 16);
        if (!atom::validate_atom_persistent_identity_v1(*identities[index]).valid()) {
            return {partial_image_code_v1::invalid_header, 0, index};
        }
    }
    std::uint64_t *generations[]{
        &header.structure_generation, &header.value_generation,
        &header.state_generation, &header.materialization_generation,
        &header.cost_model_generation};
    for (std::uint32_t index = 0; index < 5; ++index) {
        *generations[index] = partial_image_read_u64_v1(source + 208 + index * 8);
        if (*generations[index] == 0) {
            return {partial_image_code_v1::invalid_header, 0, 9 + index};
        }
    }
    header.persistence_class = static_cast<partial_persistence_class_v1>(
        partial_image_read_u32_v1(source + 248));
    header.reserved = partial_image_read_u32_v1(source + 252);
    if (header.persistence_class
            != partial_persistence_class_v1::exact_reconstructible
        || header.reserved != 0) {
        return {partial_image_code_v1::invalid_header, 0, 14};
    }
    const auto certification = partial_image_read_id_v1(source + 256);
    if (!atom::validate_atom_persistent_identity_v1(certification).valid()) {
        return {partial_image_code_v1::invalid_dependency, 0, 0};
    }
    output->image = source;
    output->image_bytes = image_bytes;
    output->payload = source + payload_offset;
    output->payload_bytes = payload_bytes;
    output->header = header;
    output->exact_dependency_certification_identity = certification;
    output->dependency_offset = dependency_offset;
    output->dependency_count = dependency_count;
    partial_dependency_requirement_v1 previous{};
    for (std::uint64_t index = 0; index < dependency_count; ++index) {
        const auto dependency = partial_image_dependency_v1(*output, index);
        if (!atom::validate_atom_persistent_identity_v1(
                 dependency.dependency_identity).valid()
            || !atom::validate_atom_persistent_identity_v1(
                    dependency.generation_namespace).valid()
            || dependency.captured_generation == 0
            || !atom::valid_atom_dependency_generation_kind_v1(
                dependency.generation_kind)
            || !valid_partial_dependency_role_v1(dependency.role)
            || (index != 0
                && !partial_dependency_key_less_v1(previous, dependency))) {
            *output = {};
            return {partial_image_code_v1::invalid_dependency, 0, index};
        }
        previous = dependency;
    }
    if (dependency_count == 0) {
        *output = {};
        return {partial_image_code_v1::invalid_dependency, 0, 0};
    }
    return {partial_image_code_v1::valid, image_bytes, dependency_count};
}

} // namespace cellshard::compiler::partial
