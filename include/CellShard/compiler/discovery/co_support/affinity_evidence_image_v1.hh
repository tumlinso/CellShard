#pragma once

#include <CellShard/compiler/discovery/co_support/affinity_stability_v1.hh>
#include <CellShard/compiler/discovery/co_support/exact_group_rescan_v1.hh>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::co_support {

inline constexpr std::uint64_t affinity_evidence_magic_v1
    = UINT64_C(0x3145564646415343);

struct affinity_evidence_section_v1 {
    std::uint64_t offset = 0;
    std::uint64_t count = 0;
    std::uint64_t stride = 0;
};

struct affinity_evidence_header_v1 {
    std::uint64_t magic = affinity_evidence_magic_v1;
    std::uint32_t schema_version = 1;
    std::uint32_t header_bytes = sizeof(affinity_evidence_header_v1);
    std::uint64_t total_bytes = 0;
    std::uint64_t evidence_identity = 0;
    std::uint64_t relation_identity = 0;
    std::uint64_t structure_epoch = 0;
    std::uint64_t payload_checksum = 0;
    affinity_evidence_section_v1 associations{};
    affinity_evidence_section_v1 affinity_edges{};
    affinity_evidence_section_v1 stability{};
    affinity_evidence_section_v1 rescans{};
};

struct affinity_evidence_view_v1 {
    std::uint64_t evidence_identity = 0;
    std::uint64_t relation_identity = 0;
    std::uint64_t structure_epoch = 0;
    const normalized_association_record_v1 *associations = nullptr;
    std::uint64_t association_count = 0;
    const source_affinity_edge_v1 *affinity_edges = nullptr;
    std::uint64_t affinity_edge_count = 0;
    const affinity_stability_record_v1 *stability = nullptr;
    std::uint64_t stability_count = 0;
    const exact_group_rescan_summary_v1 *rescans = nullptr;
    std::uint64_t rescan_count = 0;
};

enum class affinity_evidence_image_code_v1 : std::uint32_t {
    packed = 0,
    invalid_identity,
    missing_input,
    size_overflow,
    missing_output,
    insufficient_capacity,
    invalid_image,
    checksum_mismatch,
};

struct affinity_evidence_image_result_v1 {
    affinity_evidence_image_code_v1 code
        = affinity_evidence_image_code_v1::packed;
    std::uint64_t required_bytes = 0;
    [[nodiscard]] constexpr bool packed() const noexcept {
        return code == affinity_evidence_image_code_v1::packed;
    }
};

[[nodiscard]] inline bool checked_section_end_v1(
    std::uint64_t begin,
    std::uint64_t count,
    std::uint64_t stride,
    std::uint64_t *end) noexcept {
    if (count != 0 && stride > std::numeric_limits<std::uint64_t>::max() / count)
        return false;
    const auto bytes = count * stride;
    if (begin > std::numeric_limits<std::uint64_t>::max() - bytes) return false;
    *end = begin + bytes;
    return true;
}

[[nodiscard]] inline std::uint64_t affinity_payload_checksum_v1(
    const std::byte *data, std::uint64_t size) noexcept {
    std::uint64_t hash = UINT64_C(1469598103934665603);
    for (std::uint64_t index = 0; index < size; ++index) {
        hash ^= static_cast<std::uint8_t>(data[index]);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

[[nodiscard]] inline affinity_evidence_image_result_v1
pack_affinity_evidence_image_v1(
    affinity_evidence_view_v1 evidence,
    std::byte *output,
    std::uint64_t output_capacity) noexcept {
    if (evidence.evidence_identity == 0 || evidence.relation_identity == 0
        || evidence.structure_epoch == 0)
        return {affinity_evidence_image_code_v1::invalid_identity};
    if ((evidence.association_count != 0 && evidence.associations == nullptr)
        || (evidence.affinity_edge_count != 0 && evidence.affinity_edges == nullptr)
        || (evidence.stability_count != 0 && evidence.stability == nullptr)
        || (evidence.rescan_count != 0 && evidence.rescans == nullptr))
        return {affinity_evidence_image_code_v1::missing_input};

    affinity_evidence_header_v1 header{};
    header.evidence_identity = evidence.evidence_identity;
    header.relation_identity = evidence.relation_identity;
    header.structure_epoch = evidence.structure_epoch;
    std::uint64_t cursor = sizeof(header);
#define CS_JBC_ASSIGN_SECTION(field, count_value, record_type)                 \
    header.field = {cursor, count_value, sizeof(record_type)};                 \
    if (!checked_section_end_v1(cursor, count_value, sizeof(record_type),       \
                                &cursor))                                       \
        return {affinity_evidence_image_code_v1::size_overflow}
    CS_JBC_ASSIGN_SECTION(associations, evidence.association_count,
                          normalized_association_record_v1);
    CS_JBC_ASSIGN_SECTION(affinity_edges, evidence.affinity_edge_count,
                          source_affinity_edge_v1);
    CS_JBC_ASSIGN_SECTION(stability, evidence.stability_count,
                          affinity_stability_record_v1);
    CS_JBC_ASSIGN_SECTION(rescans, evidence.rescan_count,
                          exact_group_rescan_summary_v1);
#undef CS_JBC_ASSIGN_SECTION
    header.total_bytes = cursor;
    if (output == nullptr)
        return {output_capacity == 0 ? affinity_evidence_image_code_v1::packed
                                     : affinity_evidence_image_code_v1::missing_output,
                cursor};
    if (output_capacity < cursor)
        return {affinity_evidence_image_code_v1::insufficient_capacity, cursor};
    std::memcpy(output, &header, sizeof(header));
#define CS_JBC_COPY_SECTION(field, pointer_value)                              \
    if (header.field.count != 0)                                                \
        std::memcpy(output + header.field.offset, pointer_value,                \
                    header.field.count * header.field.stride)
    CS_JBC_COPY_SECTION(associations, evidence.associations);
    CS_JBC_COPY_SECTION(affinity_edges, evidence.affinity_edges);
    CS_JBC_COPY_SECTION(stability, evidence.stability);
    CS_JBC_COPY_SECTION(rescans, evidence.rescans);
#undef CS_JBC_COPY_SECTION
    header.payload_checksum = affinity_payload_checksum_v1(
        output + sizeof(header), cursor - sizeof(header));
    std::memcpy(output, &header, sizeof(header));
    return {affinity_evidence_image_code_v1::packed, cursor};
}

[[nodiscard]] inline affinity_evidence_image_result_v1
validate_affinity_evidence_image_v1(
    const std::byte *image, std::uint64_t image_bytes) noexcept {
    if (image == nullptr || image_bytes < sizeof(affinity_evidence_header_v1))
        return {affinity_evidence_image_code_v1::invalid_image};
    affinity_evidence_header_v1 header{};
    std::memcpy(&header, image, sizeof(header));
    if (header.magic != affinity_evidence_magic_v1 || header.schema_version != 1
        || header.header_bytes != sizeof(header)
        || header.total_bytes != image_bytes || header.evidence_identity == 0
        || header.relation_identity == 0 || header.structure_epoch == 0)
        return {affinity_evidence_image_code_v1::invalid_image};
    const affinity_evidence_section_v1 sections[] = {
        header.associations, header.affinity_edges, header.stability,
        header.rescans};
    const std::uint64_t strides[] = {
        sizeof(normalized_association_record_v1), sizeof(source_affinity_edge_v1),
        sizeof(affinity_stability_record_v1), sizeof(exact_group_rescan_summary_v1)};
    std::uint64_t expected_offset = sizeof(header);
    for (std::uint32_t index = 0; index < 4; ++index) {
        std::uint64_t end = 0;
        if (sections[index].offset != expected_offset
            || sections[index].stride != strides[index]
            || !checked_section_end_v1(sections[index].offset,
                                       sections[index].count,
                                       sections[index].stride, &end)
            || end > image_bytes)
            return {affinity_evidence_image_code_v1::invalid_image};
        expected_offset = end;
    }
    if (expected_offset != image_bytes)
        return {affinity_evidence_image_code_v1::invalid_image};
    const auto checksum = affinity_payload_checksum_v1(
        image + sizeof(header), image_bytes - sizeof(header));
    if (checksum != header.payload_checksum)
        return {affinity_evidence_image_code_v1::checksum_mismatch};
    return {affinity_evidence_image_code_v1::packed, image_bytes};
}

static_assert(std::is_standard_layout<affinity_evidence_header_v1>::value);
static_assert(std::is_trivially_copyable<affinity_evidence_header_v1>::value);

} // namespace cellshard::compiler::discovery::co_support
