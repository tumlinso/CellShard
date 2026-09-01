#pragma once

#include <CellShard/compiler/discovery/multimodal/identity_spine_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::multimodal {

enum class cross_modal_relation_kind_v1 : std::uint32_t {
    subject_correspondence = 1,
    regulatory = 2,
    spatial_proximity = 3,
    sequence_annotation = 4,
    statistical_proposal = 5,
};

struct cross_modal_relation_atom_v1 {
    std::uint64_t atom_identity = 0;
    std::uint64_t relation_identity = 0;
    std::uint64_t evidence_identity = 0;
    std::uint64_t source_modality_identity = 0;
    std::uint64_t source_entity_identity = 0;
    std::uint64_t destination_modality_identity = 0;
    std::uint64_t destination_entity_identity = 0;
    std::int64_t weight_numerator = 0;
    std::uint64_t weight_denominator = 1;
    cross_modal_relation_kind_v1 kind
        = cross_modal_relation_kind_v1::statistical_proposal;
    std::uint32_t directed = 1;
};

enum class cross_modal_atom_code_v1 : std::uint32_t {
    constructed = 0,
    invalid_spine,
    missing_candidates,
    missing_output,
    insufficient_capacity,
    invalid_atom,
    unknown_modality,
    not_cross_modal,
    duplicate_atom_identity,
};

struct cross_modal_atom_result_v1 {
    cross_modal_atom_code_v1 code = cross_modal_atom_code_v1::constructed;
    std::uint64_t atom_count = 0;
    std::uint64_t candidate_index = 0;
    [[nodiscard]] constexpr bool constructed() const noexcept {
        return code == cross_modal_atom_code_v1::constructed;
    }
};

[[nodiscard]] inline bool spine_has_modality_v1(
    multimodal_identity_spine_view_v1 spine,
    std::uint64_t modality_identity) noexcept {
    for (std::uint32_t index = 0; index < spine.modality_count; ++index)
        if (spine.modalities[index].modality_identity == modality_identity)
            return true;
    return false;
}

[[nodiscard]] inline cross_modal_atom_result_v1
construct_cross_modal_atoms_v1(
    multimodal_identity_spine_view_v1 spine,
    const cross_modal_relation_atom_v1 *candidates,
    std::uint64_t candidate_count,
    cross_modal_relation_atom_v1 *output,
    std::uint64_t output_capacity) noexcept {
    if (!validate_multimodal_identity_spine_v1(spine).valid())
        return {cross_modal_atom_code_v1::invalid_spine};
    if (candidate_count != 0 && candidates == nullptr)
        return {cross_modal_atom_code_v1::missing_candidates};
    if (output_capacity != 0 && output == nullptr)
        return {cross_modal_atom_code_v1::missing_output};
    if (output_capacity < candidate_count)
        return {cross_modal_atom_code_v1::insufficient_capacity};
    cross_modal_atom_result_v1 result{};
    for (std::uint64_t index = 0; index < candidate_count; ++index) {
        const auto &atom = candidates[index];
        if (atom.atom_identity == 0 || atom.relation_identity == 0
            || atom.evidence_identity == 0 || atom.source_entity_identity == 0
            || atom.destination_entity_identity == 0
            || atom.weight_denominator == 0 || atom.directed > 1)
            return {cross_modal_atom_code_v1::invalid_atom,
                    result.atom_count, index};
        if (!spine_has_modality_v1(spine, atom.source_modality_identity)
            || !spine_has_modality_v1(spine,
                                      atom.destination_modality_identity))
            return {cross_modal_atom_code_v1::unknown_modality,
                    result.atom_count, index};
        if (atom.source_modality_identity == atom.destination_modality_identity)
            return {cross_modal_atom_code_v1::not_cross_modal,
                    result.atom_count, index};
        std::uint64_t position = 0;
        while (position < result.atom_count
               && output[position].atom_identity < atom.atom_identity)
            ++position;
        if (position < result.atom_count
            && output[position].atom_identity == atom.atom_identity)
            return {cross_modal_atom_code_v1::duplicate_atom_identity,
                    result.atom_count, index};
        for (auto move = result.atom_count; move > position; --move)
            output[move] = output[move - 1];
        output[position] = atom;
        ++result.atom_count;
        result.candidate_index = index + 1;
    }
    return result;
}

static_assert(std::is_standard_layout<cross_modal_relation_atom_v1>::value);
static_assert(std::is_trivially_copyable<cross_modal_relation_atom_v1>::value);

} // namespace cellshard::compiler::discovery::multimodal
