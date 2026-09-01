#pragma once

#include <CellShard/compiler/discovery/multimodal/identity_spine_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::multimodal {

enum class multimodal_payload_kind_v1 : std::uint32_t {
    sparse_counts = 1,
    continuous_values = 2,
    genomic_intervals = 3,
    sequence_events = 4,
    spatial_coordinates = 5,
    missingness_mask = 6,
};

struct multimodal_payload_descriptor_v1 {
    std::uint64_t modality_identity = 0;
    std::uint64_t payload_identity = 0;
    std::uint64_t value_generation = 0;
    std::uint64_t byte_offset = 0;
    std::uint64_t byte_count = 0;
    std::uint64_t element_count = 0;
    multimodal_payload_kind_v1 kind
        = multimodal_payload_kind_v1::continuous_values;
    std::uint32_t alignment = 1;
};

struct multi_payload_atom_v1 {
    std::uint64_t atom_identity = 0;
    std::uint64_t evidence_identity = 0;
    std::uint64_t spine_identity = 0;
    std::uint64_t structure_epoch = 0;
    std::uint64_t subject_begin = 0;
    std::uint64_t subject_count = 0;
    std::uint64_t payload_offset = 0;
    std::uint32_t payload_count = 0;
    std::uint32_t reserved = 0;
};

enum class multi_payload_atom_code_v1 : std::uint32_t {
    valid = 0,
    invalid_spine,
    invalid_atom,
    missing_payloads,
    payload_range_overflow,
    invalid_payload,
    unknown_modality,
    duplicate_modality,
    payload_out_of_bounds,
};

struct multi_payload_atom_result_v1 {
    multi_payload_atom_code_v1 code = multi_payload_atom_code_v1::valid;
    std::uint32_t payload_index = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == multi_payload_atom_code_v1::valid;
    }
};

[[nodiscard]] inline multi_payload_atom_result_v1
validate_multi_payload_atom_v1(
    multimodal_identity_spine_view_v1 spine,
    multi_payload_atom_v1 atom,
    const multimodal_payload_descriptor_v1 *payloads,
    std::uint64_t payload_descriptor_count,
    std::uint64_t payload_bytes) noexcept {
    if (!validate_multimodal_identity_spine_v1(spine).valid()
        || atom.spine_identity != spine.spine_identity
        || atom.structure_epoch != spine.structure_epoch)
        return {multi_payload_atom_code_v1::invalid_spine};
    if (atom.atom_identity == 0 || atom.evidence_identity == 0
        || atom.subject_count == 0 || atom.payload_count < 2)
        return {multi_payload_atom_code_v1::invalid_atom};
    if (payloads == nullptr)
        return {multi_payload_atom_code_v1::missing_payloads};
    if (atom.payload_offset > payload_descriptor_count
        || atom.payload_count
            > payload_descriptor_count - atom.payload_offset)
        return {multi_payload_atom_code_v1::payload_range_overflow};
    for (std::uint32_t index = 0; index < atom.payload_count; ++index) {
        const auto &payload = payloads[atom.payload_offset + index];
        if (payload.payload_identity == 0 || payload.value_generation == 0
            || payload.byte_count == 0 || payload.element_count == 0
            || payload.alignment == 0
            || (payload.alignment & (payload.alignment - 1u)) != 0
            || payload.byte_offset % payload.alignment != 0)
            return {multi_payload_atom_code_v1::invalid_payload, index};
        bool known = false;
        for (std::uint32_t modality = 0; modality < spine.modality_count;
             ++modality)
            if (spine.modalities[modality].modality_identity
                == payload.modality_identity) {
                known = true;
                break;
            }
        if (!known)
            return {multi_payload_atom_code_v1::unknown_modality, index};
        for (std::uint32_t previous = 0; previous < index; ++previous)
            if (payloads[atom.payload_offset + previous].modality_identity
                == payload.modality_identity)
                return {multi_payload_atom_code_v1::duplicate_modality, index};
        if (payload.byte_offset > payload_bytes
            || payload.byte_count > payload_bytes - payload.byte_offset)
            return {multi_payload_atom_code_v1::payload_out_of_bounds, index};
    }
    return {};
}

static_assert(std::is_standard_layout<multimodal_payload_descriptor_v1>::value);
static_assert(std::is_trivially_copyable<multimodal_payload_descriptor_v1>::value);
static_assert(std::is_standard_layout<multi_payload_atom_v1>::value);
static_assert(std::is_trivially_copyable<multi_payload_atom_v1>::value);

} // namespace cellshard::compiler::discovery::multimodal
