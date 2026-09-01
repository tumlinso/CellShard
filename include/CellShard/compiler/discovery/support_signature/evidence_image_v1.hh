#pragma once

#include <CellShard/compiler/discovery/support_signature/lsh_index_v1.hh>
#include <CellShard/compiler/evidence/negative_evidence_v1.hh>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellshard::compiler::discovery::support_signature {

inline constexpr std::uint64_t support_evidence_magic_v1 =
    UINT64_C(0x3156454749535353); // "SSSIGEV1" in little-endian bytes.
inline constexpr std::uint32_t support_evidence_schema_v1 = 1;
inline constexpr std::uint32_t support_evidence_header_bytes_v1 = 104;
inline constexpr std::uint32_t negative_evidence_record_bytes_v1 = 80;

enum class support_evidence_image_code_v1 : std::uint32_t {
    built = 0,
    valid = 1,
    invalid_support,
    invalid_sketch,
    context_mismatch,
    missing_negative_evidence,
    invalid_negative_evidence,
    stale_negative_generation,
    unordered_or_duplicate_negative_evidence,
    size_overflow,
    missing_output,
    insufficient_output,
    truncated_image,
    invalid_magic,
    unsupported_schema,
    invalid_header,
    invalid_layout,
    invalid_destination_ids,
};

struct support_evidence_image_result_v1 {
    support_evidence_image_code_v1 code = support_evidence_image_code_v1::built;
    std::uint64_t image_bytes = 0;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool built() const noexcept {
        return code == support_evidence_image_code_v1::built;
    }
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == support_evidence_image_code_v1::valid;
    }
};

namespace detail {

constexpr void put_u32_v1(std::byte *output, std::uint64_t offset,
                          std::uint32_t value) noexcept {
    for (std::uint32_t byte = 0; byte < 4; ++byte)
        output[offset + byte] = std::byte((value >> (byte * 8)) & 0xffu);
}

constexpr void put_u64_v1(std::byte *output, std::uint64_t offset,
                          std::uint64_t value) noexcept {
    for (std::uint32_t byte = 0; byte < 8; ++byte)
        output[offset + byte] = std::byte((value >> (byte * 8)) & 0xffu);
}

[[nodiscard]] constexpr std::uint32_t get_u32_v1(
    const std::byte *input, std::uint64_t offset) noexcept {
    std::uint32_t value = 0;
    for (std::uint32_t byte = 0; byte < 4; ++byte)
        value |= std::uint32_t(input[offset + byte]) << (byte * 8);
    return value;
}

[[nodiscard]] constexpr std::uint64_t get_u64_v1(
    const std::byte *input, std::uint64_t offset) noexcept {
    std::uint64_t value = 0;
    for (std::uint32_t byte = 0; byte < 8; ++byte)
        value |= std::uint64_t(input[offset + byte]) << (byte * 8);
    return value;
}

[[nodiscard]] constexpr bool checked_add_v1(
    std::uint64_t lhs, std::uint64_t rhs, std::uint64_t *output) noexcept {
    if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs) return false;
    *output = lhs + rhs;
    return true;
}

[[nodiscard]] constexpr bool checked_multiply_v1(
    std::uint64_t lhs, std::uint64_t rhs, std::uint64_t *output) noexcept {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        return false;
    *output = lhs * rhs;
    return true;
}

constexpr void put_identity_v1(
    std::byte *output, std::uint64_t offset,
    evidence::evidence_identity_v1 identity) noexcept {
    put_u64_v1(output, offset, identity.producer_namespace);
    put_u64_v1(output, offset + 8, identity.local_identity);
}

[[nodiscard]] constexpr evidence::evidence_identity_v1 get_identity_v1(
    const std::byte *input, std::uint64_t offset) noexcept {
    return {get_u64_v1(input, offset), get_u64_v1(input, offset + 8)};
}

constexpr void put_negative_v1(
    std::byte *output, std::uint64_t offset,
    const evidence::negative_evidence_v1 &record) noexcept {
    put_identity_v1(output, offset, record.evidence_identity);
    put_identity_v1(output, offset + 16, record.subject_identity);
    put_identity_v1(output, offset + 32, record.observation_scope_identity);
    put_u64_v1(output, offset + 48, record.observation_generation);
    put_u64_v1(output, offset + 56, record.attempted_observations);
    put_u64_v1(output, offset + 64, record.contradictory_observations);
    put_u32_v1(output, offset + 72, static_cast<std::uint32_t>(record.reason));
    put_u32_v1(output, offset + 76, record.reserved);
}

[[nodiscard]] constexpr evidence::negative_evidence_v1 get_negative_v1(
    const std::byte *input, std::uint64_t offset) noexcept {
    return {get_identity_v1(input, offset),
            get_identity_v1(input, offset + 16),
            get_identity_v1(input, offset + 32),
            get_u64_v1(input, offset + 48),
            get_u64_v1(input, offset + 56),
            get_u64_v1(input, offset + 64),
            static_cast<evidence::negative_evidence_reason_v1>(
                get_u32_v1(input, offset + 72)),
            get_u32_v1(input, offset + 76)};
}

} // namespace detail

[[nodiscard]] constexpr support_evidence_image_result_v1
build_support_evidence_image_v1(
    exact_destination_support_view_v1 support,
    deterministic_minhash_view_v1 sketch,
    const evidence::negative_evidence_v1 *negative_evidence,
    std::uint64_t negative_count,
    std::byte *output,
    std::uint64_t output_capacity) noexcept {
    if (!validate_exact_destination_support_view_v1(support))
        return {support_evidence_image_code_v1::invalid_support};
    if (!valid_minhash_view_v1(sketch))
        return {support_evidence_image_code_v1::invalid_sketch};
    if (sketch.destination_count != support.destination_count
        || sketch.relation_identity != support.relation_identity
        || sketch.relation_generation != support.relation_generation)
        return {support_evidence_image_code_v1::context_mismatch};
    if (negative_count != 0 && negative_evidence == nullptr)
        return {support_evidence_image_code_v1::missing_negative_evidence};
    for (std::uint64_t index = 0; index < negative_count; ++index) {
        if (!evidence::validate_negative_evidence_v1(
                negative_evidence[index]).valid())
            return {support_evidence_image_code_v1::invalid_negative_evidence,
                    0, index};
        if (negative_evidence[index].observation_generation
            != support.relation_generation)
            return {support_evidence_image_code_v1::stale_negative_generation,
                    0, index};
        if (index != 0
            && !evidence::evidence_identity_less_v1(
                negative_evidence[index - 1].evidence_identity,
                negative_evidence[index].evidence_identity))
            return {support_evidence_image_code_v1::
                        unordered_or_duplicate_negative_evidence,
                    0, index};
    }
    std::uint64_t destination_bytes = 0;
    std::uint64_t minima_count = 0;
    std::uint64_t minima_bytes = 0;
    std::uint64_t negative_bytes = 0;
    std::uint64_t minima_offset = 0;
    std::uint64_t negative_offset = 0;
    std::uint64_t image_bytes = 0;
    if (!detail::checked_multiply_v1(
            support.destination_count, 8, &destination_bytes)
        || !detail::checked_multiply_v1(
            support.destination_count, sketch.sketch_size, &minima_count)
        || !detail::checked_multiply_v1(minima_count, 8, &minima_bytes)
        || !detail::checked_multiply_v1(
            negative_count, negative_evidence_record_bytes_v1, &negative_bytes)
        || !detail::checked_add_v1(
            support_evidence_header_bytes_v1, destination_bytes,
            &minima_offset)
        || !detail::checked_add_v1(
            minima_offset, minima_bytes, &negative_offset)
        || !detail::checked_add_v1(
            negative_offset, negative_bytes, &image_bytes))
        return {support_evidence_image_code_v1::size_overflow};
    if (output == nullptr)
        return {support_evidence_image_code_v1::missing_output, image_bytes};
    if (output_capacity < image_bytes)
        return {support_evidence_image_code_v1::insufficient_output,
                image_bytes};
    detail::put_u64_v1(output, 0, support_evidence_magic_v1);
    detail::put_u32_v1(output, 8, support_evidence_schema_v1);
    detail::put_u32_v1(output, 12, support_evidence_header_bytes_v1);
    detail::put_u64_v1(output, 16,
                       support.relation_identity.producer_namespace);
    detail::put_u64_v1(output, 24, support.relation_identity.local_identity);
    detail::put_u64_v1(output, 32, support.relation_generation);
    detail::put_u64_v1(output, 40, sketch.seed_namespace);
    detail::put_u64_v1(output, 48, support.destination_count);
    detail::put_u32_v1(output, 56, sketch.sketch_size);
    detail::put_u32_v1(output, 60, 0);
    detail::put_u64_v1(output, 64, negative_count);
    detail::put_u64_v1(output, 72, support_evidence_header_bytes_v1);
    detail::put_u64_v1(output, 80, minima_offset);
    detail::put_u64_v1(output, 88, negative_offset);
    detail::put_u64_v1(output, 96, image_bytes);
    for (std::uint64_t index = 0; index < support.destination_count; ++index)
        detail::put_u64_v1(output, support_evidence_header_bytes_v1 + 8 * index,
                           support.global_destination_ids[index]);
    for (std::uint64_t index = 0; index < minima_count; ++index)
        detail::put_u64_v1(output, minima_offset + 8 * index,
                           sketch.minima[index]);
    for (std::uint64_t index = 0; index < negative_count; ++index)
        detail::put_negative_v1(
            output, negative_offset + negative_evidence_record_bytes_v1 * index,
            negative_evidence[index]);
    return {support_evidence_image_code_v1::built, image_bytes,
            negative_count};
}

[[nodiscard]] constexpr support_evidence_image_result_v1
validate_support_evidence_image_v1(
    const std::byte *image, std::uint64_t available_bytes) noexcept {
    if (image == nullptr || available_bytes < support_evidence_header_bytes_v1)
        return {support_evidence_image_code_v1::truncated_image};
    if (detail::get_u64_v1(image, 0) != support_evidence_magic_v1)
        return {support_evidence_image_code_v1::invalid_magic};
    if (detail::get_u32_v1(image, 8) != support_evidence_schema_v1)
        return {support_evidence_image_code_v1::unsupported_schema};
    if (detail::get_u32_v1(image, 12) != support_evidence_header_bytes_v1
        || detail::get_u32_v1(image, 60) != 0)
        return {support_evidence_image_code_v1::invalid_header};
    const atom::atom_persistent_identity_v1 relation{
        detail::get_u64_v1(image, 16), detail::get_u64_v1(image, 24)};
    const auto generation = detail::get_u64_v1(image, 32);
    const auto seed = detail::get_u64_v1(image, 40);
    const auto destinations = detail::get_u64_v1(image, 48);
    const auto sketch_size = detail::get_u32_v1(image, 56);
    const auto negative_count = detail::get_u64_v1(image, 64);
    const auto destination_offset = detail::get_u64_v1(image, 72);
    const auto minima_offset = detail::get_u64_v1(image, 80);
    const auto negative_offset = detail::get_u64_v1(image, 88);
    const auto image_bytes = detail::get_u64_v1(image, 96);
    if (!atom::validate_atom_persistent_identity_v1(relation).valid()
        || generation == 0 || seed == 0 || destinations == 0
        || sketch_size == 0 || image_bytes > available_bytes)
        return {support_evidence_image_code_v1::invalid_header};
    std::uint64_t expected_minima = 0;
    std::uint64_t minima_count = 0;
    std::uint64_t expected_negative = 0;
    std::uint64_t expected_image = 0;
    if (!detail::checked_multiply_v1(destinations, 8, &expected_minima)
        || !detail::checked_add_v1(
            support_evidence_header_bytes_v1, expected_minima,
            &expected_minima)
        || !detail::checked_multiply_v1(
            destinations, sketch_size, &minima_count)
        || !detail::checked_multiply_v1(
            minima_count, 8, &expected_negative)
        || !detail::checked_add_v1(
            expected_minima, expected_negative, &expected_negative)
        || !detail::checked_multiply_v1(
            negative_count, negative_evidence_record_bytes_v1, &expected_image)
        || !detail::checked_add_v1(
            expected_negative, expected_image, &expected_image)
        || destination_offset != support_evidence_header_bytes_v1
        || minima_offset != expected_minima
        || negative_offset != expected_negative || image_bytes != expected_image)
        return {support_evidence_image_code_v1::invalid_layout};
    std::uint64_t previous_destination = 0;
    for (std::uint64_t index = 0; index < destinations; ++index) {
        const auto value = detail::get_u64_v1(
            image, destination_offset + 8 * index);
        if (value == 0 || (index != 0 && previous_destination >= value))
            return {support_evidence_image_code_v1::invalid_destination_ids,
                    image_bytes, index};
        previous_destination = value;
    }
    evidence::evidence_identity_v1 previous_identity{};
    for (std::uint64_t index = 0; index < negative_count; ++index) {
        const auto record = detail::get_negative_v1(
            image, negative_offset + negative_evidence_record_bytes_v1 * index);
        if (!evidence::validate_negative_evidence_v1(record).valid()
            || record.observation_generation != generation)
            return {support_evidence_image_code_v1::invalid_negative_evidence,
                    image_bytes, index};
        if (index != 0
            && !evidence::evidence_identity_less_v1(
                previous_identity, record.evidence_identity))
            return {support_evidence_image_code_v1::
                        unordered_or_duplicate_negative_evidence,
                    image_bytes, index};
        previous_identity = record.evidence_identity;
    }
    return {support_evidence_image_code_v1::valid, image_bytes,
            negative_count};
}

[[nodiscard]] constexpr bool authorizes_execution(
    support_evidence_image_result_v1) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::support_signature
